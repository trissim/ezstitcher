#!/usr/bin/env python3
"""
Full workflow test for the EZStitcher package.

This script will:
1. Check if the input folder has Z-stacks
2. Organize Z-stack images with proper filenames
3. Detect the best focus plane for each site using the specified wavelength
4. Create a new folder with only best-focused images
5. Stitch the images together using the found positions

Usage:
    python test_full_workflow.py input_folder [--wavelength WAVELENGTH] [--focus-method METHOD]
"""

import os
import sys
import time
import shutil
import argparse
import logging
import numpy as np
import cv2
from pathlib import Path
import re
import pandas as pd

# Add parent directory to path so we can import from ezstitcher
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from ezstitcher modules
from ezstitcher.core.z_stack_handler import organize_zstack_folders
from ezstitcher.core.focus_detect import (
    combined_focus_measure, 
    normalized_variance,
    laplacian_energy,
    tenengrad_variance
)
from ezstitcher.core.stitcher import (
    process_plate_folder,
    find_HTD_file,
    auto_detect_patterns,
    generate_positions_df,
    ashlar_stitch_v2,
    assemble_image_subpixel,
    compute_stitched_name,
    parse_filename,
    folder_to_df,
    path_list_from_pattern,
    clean_folder
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def calculate_focus_score(image, method='combined'):
    """Calculate focus score for an image using the specified method."""
    # Convert BGR to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if method == 'combined':
        return combined_focus_measure(image)
    elif method == 'laplacian':
        # Compute variance of Laplacian
        return laplacian_energy(image)
    elif method == 'normalized_variance':
        return normalized_variance(image)
    elif method == 'tenengrad':
        return tenengrad_variance(image)
    else:
        # Default to Laplacian
        return laplacian_energy(image)

def find_best_focus_images(input_dir, output_dir, focus_wavelength='1', focus_method='combined'):
    """
    Find best focused images for each site and create a directory with links to them.
    (Files will have z-indices removed for consistent stitching patterns)
    
    Args:
        input_dir: Directory with Z-stack images 
        output_dir: Directory to create with best focus images
        focus_wavelength: Wavelength to use for focus detection
        focus_method: Focus detection method to use
        
    Returns:
        dict: Mapping of (well, site, wavelength) to best focus Z-index
    """
    logger.info(f"Finding best focused images using wavelength {focus_wavelength}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # First make sure all files are standardized
    logger.info(f"Processing filenames in {input_dir}...")
    df = folder_to_df(input_dir)
    
    # Check if we have z_step information
    if 'z_step' not in df.columns or df['z_step'].isna().all():
        logger.warning("No Z-step information found in the files. Make sure Z-stacks are properly organized.")
        return {}
    
    # Filter out rows with missing data
    df = df[df['well'].notna() & df['site'].notna() & df['wavelength'].notna() & df['z_step'].notna()]
    
    # Group files by well and site
    site_groups = df.groupby(['well', 'site'])
    best_focus_z = {}
    
    # Process each site
    for (well, site), site_df in site_groups:
        logger.info(f"Processing focus for well {well}, site {site}")
        
        # Filter to only the focus wavelength for determining best z-plane
        focus_df = site_df[site_df['wavelength'] == focus_wavelength]
        
        if focus_df.empty:
            logger.warning(f"No files with wavelength {focus_wavelength} found for {well}, site {site}")
            continue
        
        # Calculate focus scores for each image in the focus wavelength
        focus_scores = []
        for idx, row in focus_df.iterrows():
            try:
                img = cv2.imread(row['filepath'])
                if img is None:
                    logger.warning(f"Failed to load image: {row['filepath']}")
                    focus_scores.append((row['z_step'], -1))
                    continue
                    
                score = calculate_focus_score(img, method=focus_method)
                focus_scores.append((row['z_step'], score))
                logger.debug(f"Focus score for {os.path.basename(row['filepath'])}: {score}")
            except Exception as e:
                logger.error(f"Error calculating focus for {os.path.basename(row['filepath'])}: {e}")
                focus_scores.append((row['z_step'], -1))
        
        if not focus_scores:
            logger.warning(f"No valid focus scores for {well}, site {site}")
            continue
        
        # Find z-index with highest focus score
        best_z = max(focus_scores, key=lambda x: x[1])[0]
        best_focus_z[(well, site)] = best_z
        logger.info(f"Best focus for {well}, site {site}: z={best_z}")
    
    # Dictionary to store best focus indices for all (well, site, wavelength) combinations
    best_focus = {}
    
    # Now create links for all wavelengths using the best z-index for each site
    for (well, site), best_z in best_focus_z.items():
        logger.info(f"Creating links for well {well}, site {site}, z={best_z}")
        
        # Get all files for this site
        site_files = df[(df['well'] == well) & (df['site'] == site)]
        
        # Filter to just files at the best z-index
        best_files = site_files[site_files['z_step'] == best_z]
        
        for _, row in best_files.iterrows():
            file_path = row['filepath']
            
            # Get original filename
            original_name = os.path.basename(file_path)
            
            # Remove the z-index from the filename for consistency when stitching
            if row['z_step']:
                # Remove z-index pattern (e.g., "_z001") from the filename
                new_name = re.sub(r'_z\d{3}', '', original_name)
            else:
                new_name = original_name
            
            # Create full output path
            output_path = os.path.join(output_dir, new_name)
            
            # Use symlink if possible, otherwise copy
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.symlink(file_path, output_path)
                logger.info(f"Created link: {output_path} -> {file_path}")
            except Exception as e:
                logger.warning(f"Failed to create symlink, copying instead: {e}")
                shutil.copy2(file_path, output_path)
                logger.info(f"Copied: {file_path} -> {output_path}")
            
            # Store best focus z-index
            best_focus[(well, site, row['wavelength'])] = best_z
    
    return best_focus

def stitch_images(input_dir, output_dir, stitching_wavelength='1', tile_overlap=10, max_shift=50):
    """
    Stitch images using the specified wavelength for position determination.
    
    Args:
        input_dir: Directory with best focus images
        output_dir: Directory to save stitched images
        stitching_wavelength: Wavelength to use for stitching
        tile_overlap: Percentage overlap between tiles
        max_shift: Maximum shift in pixels allowed between tiles
    """
    logger.info(f"Stitching images using wavelength {stitching_wavelength}...")
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    positions_dir = os.path.join(os.path.dirname(output_dir), "positions")
    os.makedirs(positions_dir, exist_ok=True)
    
    # Detect grid dimensions
    grid_info = find_HTD_file(os.path.dirname(input_dir))
    if grid_info[0] is None or grid_info[1] is None:
        logger.error("Could not determine grid dimensions. Using default 4x4.")
        grid_size_x, grid_size_y = 4, 4
    else:
        grid_size_x, grid_size_y = grid_info
        logger.info(f"Detected grid dimensions: {grid_size_x}x{grid_size_y}")
    
    # Get information about all files
    logger.info(f"Getting information about files in {input_dir}...")
    file_df = folder_to_df(input_dir)
    
    # Check if we have files
    if file_df.empty:
        logger.error(f"No valid files found in {input_dir}")
        return
    
    # Group by well
    well_groups = file_df.groupby('well')
    
    # Process each well
    for well, well_df in well_groups:
        logger.info(f"Processing well {well}")
        
        # Find all available wavelengths for this well
        wavelengths = well_df['wavelength'].unique()
        logger.info(f"Found wavelengths for well {well}: {wavelengths}")
        
        # Verify stitching wavelength is available
        if stitching_wavelength not in wavelengths:
            logger.warning(f"Stitching wavelength {stitching_wavelength} not found for well {well}. Using first available wavelength.")
            stitching_wavelength = wavelengths[0]
        
        # Get a sample file to extract the file name pattern
        example_file = well_df['filepath'].iloc[0]
        example_basename = os.path.basename(example_file)
        
        # Extract the file prefix (everything before _s{site})
        prefix_match = re.match(r'(.+)_s\d+', example_basename)
        if not prefix_match:
            raise ValueError(f"Could not extract filename prefix from {example_basename}. Expected format like 'prefix_s001_w1.TIF'")
        
        prefix = prefix_match.group(1)
        
        # Create patterns for each wavelength (without z-step)
        wavelength_patterns = {}
        for wavelength in wavelengths:
            # Create a pattern with site placeholder but no z-index
            # Get an example filename for this wavelength
            wave_example_file = well_df[well_df['wavelength'] == wavelength]['filepath'].iloc[0]
            wave_example_basename = os.path.basename(wave_example_file)
            
            # Extract site from the filename
            _, site, _, _, _ = parse_filename(wave_example_basename)
            
            # Create pattern by replacing site number with {iii}
            if not site:
                raise ValueError(f"Could not extract site number from {wave_example_basename}")
                
            pattern = wave_example_basename.replace(f"_s{site}", "_s{iii}")
            
            # Remove any z-index from the pattern
            pattern = re.sub(r'_z\d{3}', '', pattern)
            
            wavelength_patterns[wavelength] = pattern
            logger.info(f"Created pattern for wavelength {wavelength}: {pattern}")
        
        # Generate positions file
        positions_path = os.path.join(positions_dir, f"{well}_positions.csv")
        
        # Use pattern for stitching wavelength
        stitch_pattern = wavelength_patterns[stitching_wavelength]
        logger.info(f"Using pattern {stitch_pattern} for stitching positions")
        
        # Check if there are at least some files matching the pattern
        matching_files = []
        # First try with the original pattern
        files = path_list_from_pattern(input_dir, stitch_pattern)
        if files:
            matching_files.extend(files)
        else:
            # If no match, try with different extensions
            for ext in ['.tif', '.TIF', '.tiff', '.TIFF']:
                if '.' in stitch_pattern:
                    base_pattern = stitch_pattern.rsplit('.', 1)[0]  # Remove extension
                    test_pattern = base_pattern + ext
                else:
                    # If no extension in pattern, just add one
                    test_pattern = stitch_pattern + ext
                
                logger.debug(f"Trying pattern {test_pattern}")
                files = path_list_from_pattern(input_dir, test_pattern)
                if files:
                    matching_files.extend(files)
                    stitch_pattern = test_pattern  # Use this pattern if it works
                    break  # Use the first working pattern
        
        if not matching_files:
            logger.error(f"No files found matching pattern {stitch_pattern} in {input_dir}")
            continue
            
        logger.info(f"Found {len(matching_files)} files matching pattern {stitch_pattern}")
        
        # Use ashlar to generate positions
        try:
            logger.info(f"Generating positions for well {well} using Ashlar...")
            ashlar_stitch_v2(
                image_dir=input_dir,
                image_pattern=stitch_pattern,
                positions_path=positions_path,
                grid_size_x=grid_size_x,
                grid_size_y=grid_size_y,
                tile_overlap=tile_overlap,
                max_shift=max_shift
            )
        except Exception as e:
            logger.error(f"Error generating positions for well {well}: {e}")
            continue
        
        # Stitch each wavelength using the generated positions
        for wavelength, pattern in wavelength_patterns.items():
            # Generate stitched filename
            stitched_name = compute_stitched_name(pattern)
            output_path = os.path.join(output_dir, stitched_name)
            
            logger.info(f"Stitching {wavelength} for well {well}...")
            
            try:
                # Find files for this wavelength
                override_files = []
                # First try with the original pattern
                files = path_list_from_pattern(input_dir, pattern)
                if files:
                    override_files.extend(files)
                else:
                    # If no match, try with different extensions
                    for ext in ['.tif', '.TIF', '.tiff', '.TIFF']:
                        if '.' in pattern:
                            base_pattern = pattern.rsplit('.', 1)[0]  # Remove extension
                            test_pattern = base_pattern + ext
                        else:
                            # If no extension in pattern, just add one
                            test_pattern = pattern + ext
                            
                        logger.debug(f"Trying wavelength pattern {test_pattern}")
                        files = path_list_from_pattern(input_dir, test_pattern)
                        if files:
                            override_files.extend(files)
                            break  # Use the first working pattern
                
                if not override_files:
                    logger.warning(f"No files found matching pattern {pattern}")
                    continue
                    
                logger.info(f"Found {len(override_files)} files for wavelength {wavelength}")
                
                # Stitch images using the positions file
                assemble_image_subpixel(
                    positions_path=positions_path,
                    images_dir=input_dir,
                    output_path=output_path,
                    margin_ratio=0.1,
                    override_names=override_files
                )
                
                logger.info(f"Successfully stitched {wavelength} for well {well}: {output_path}")
            except Exception as e:
                logger.error(f"Error stitching {wavelength} for well {well}: {e}")

def process_full_workflow(input_folder, 
                          focus_wavelength='1', 
                          focus_method='combined',
                          tile_overlap=10,
                          max_shift=50):
    """
    Process the full workflow for a plate folder.
    
    Args:
        input_folder: Path to the plate folder
        focus_wavelength: Wavelength to use for focus detection (default: 1)
        focus_method: Focus detection method to use (default: combined)
        tile_overlap: Percentage overlap between tiles (default: 10)
        max_shift: Maximum shift in pixels allowed between tiles (default: 50)
    """
    start_time = time.time()
    logger.info(f"Starting full workflow for {input_folder}")
    
    # Check if this is a valid plate folder
    time_point_path = os.path.join(input_folder, "TimePoint_1")
    if not os.path.exists(time_point_path):
        logger.error(f"TimePoint_1 not found in {input_folder}")
        return
    
    # Check if this folder has Z-stacks
    zstep_pattern = re.compile(r'^ZStep_\d+$')
    has_zstack = any(zstep_pattern.match(item) for item in os.listdir(time_point_path) 
                     if os.path.isdir(os.path.join(time_point_path, item)))
    
    # Setup directory structure
    plate_name = os.path.basename(input_folder)
    base_dir = os.path.dirname(input_folder)
    
    # Define output directories
    organized_dir = os.path.join(base_dir, f"{plate_name}_organized")
    best_focus_dir = os.path.join(base_dir, f"{plate_name}_best_focus")
    stitched_dir = os.path.join(base_dir, f"{plate_name}_stitched")
    
    organized_timepoint = os.path.join(organized_dir, "TimePoint_1")
    best_focus_timepoint = os.path.join(best_focus_dir, "TimePoint_1")
    stitched_timepoint = os.path.join(stitched_dir, "TimePoint_1")
    
    # Step 1: Check for Z-stacks and organize if needed
    if has_zstack:
        logger.info(f"Z-stack detected in {input_folder}")
        
        # Create organized directory
        os.makedirs(organized_dir, exist_ok=True)
        
        # Copy HTD file if present
        htd_files = list(Path(input_folder).glob("*.HTD"))
        if htd_files:
            for htd_file in htd_files:
                shutil.copy2(htd_file, organized_dir)
        
        # Organize Z-stack folders
        logger.info("Organizing Z-stack folders...")
        organize_zstack_folders(input_folder)
        
        # Use the original folder which now has organized Z-stack images
        target_path = time_point_path
    else:
        logger.info(f"No Z-stack detected in {input_folder}")
        # No reorganization needed, we'll work with the original folder
        target_path = time_point_path
    
    # Step 2: Find best focus for each site
    logger.info(f"Finding best focus using wavelength {focus_wavelength}...")
    best_focus_map = find_best_focus_images(
        target_path, 
        best_focus_timepoint,
        focus_wavelength=focus_wavelength,
        focus_method=focus_method
    )
    
    # Copy HTD file to best focus directory
    htd_files = list(Path(input_folder).glob("*.HTD"))
    if htd_files:
        for htd_file in htd_files:
            shutil.copy2(htd_file, best_focus_dir)
    
    # Standardize filenames in the best focus directory to ensure proper site padding
    logger.info(f"Standardizing filenames in best focus directory...")
    clean_folder(best_focus_timepoint)
    
    # Step 3: Stitch images
    logger.info(f"Stitching images using wavelength {focus_wavelength}...")
    stitch_images(
        best_focus_timepoint,
        stitched_timepoint,
        stitching_wavelength=focus_wavelength,
        tile_overlap=tile_overlap,
        max_shift=max_shift
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Workflow completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results: ")
    logger.info(f"  - Best focus images: {best_focus_timepoint}")
    logger.info(f"  - Stitched images: {stitched_timepoint}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EZStitcher full workflow test")
    
    parser.add_argument("input_folder", 
                        help="Path to the plate folder containing TimePoint_1")
    parser.add_argument("--focus-wavelength", "-fw", 
                        default="1", 
                        help="Wavelength to use for focus detection (default: 1)")
    parser.add_argument("--focus-method", "-fm", 
                        default="combined", 
                        choices=["combined", "laplacian", "normalized_variance", "tenengrad"],
                        help="Focus detection method to use (default: combined)")
    parser.add_argument("--tile-overlap", "-to", 
                        type=float, 
                        default=10.0,
                        help="Percentage overlap between tiles (default: 10.0)")
    parser.add_argument("--max-shift", "-ms", 
                        type=int, 
                        default=50,
                        help="Maximum shift in pixels allowed between tiles (default: 50)")
    
    args = parser.parse_args()
    
    process_full_workflow(
        args.input_folder,
        focus_wavelength=args.focus_wavelength,
        focus_method=args.focus_method,
        tile_overlap=args.tile_overlap,
        max_shift=args.max_shift
    )