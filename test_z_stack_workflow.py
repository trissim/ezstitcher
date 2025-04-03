#!/usr/bin/env python3
"""
Test script for enhanced Z-stack processing functionality.

This script demonstrates the improved Z-stack handling capabilities:
1. Z-stack organization and detection
2. Best focus image selection across z-planes
3. 3D projections (max, mean, etc.)
4. Stitching using z-stack aware methods

Usage:
    python test_z_stack_workflow.py input_folder [options]
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add parent directory to path so we can import from ezstitcher
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import enhanced z-stack handling modules
from ezstitcher.core.z_stack_handler import (
    preprocess_plate_folder,
    select_best_focus_zstack,
    create_zstack_projections,
    stitch_across_z,
    modified_process_plate_folder
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

def process_z_stack(input_folder, 
                   focus_wavelength='1', 
                   focus_method='combined',
                   create_projections=True,
                   projection_types=None,
                   stitch_method='best_focus',
                   reference_channels=None,
                   tile_overlap=10,
                   max_shift=50):
    """
    Process a plate folder with Z-stack images using enhanced functionality.
    
    Args:
        input_folder: Path to the plate folder
        focus_wavelength: Wavelength to use for focus detection
        focus_method: Focus detection method to use
        create_projections: Whether to create Z-stack projections
        projection_types: Types of projections to create
        stitch_method: Method for stitching ('best_focus' or z-index)
        reference_channels: Wavelengths to use as reference for stitching
        tile_overlap: Percentage overlap between tiles
        max_shift: Maximum shift allowed between tiles
    """
    start_time = time.time()
    logger.info(f"Starting Z-stack processing for {input_folder}")
    
    # Default values
    if projection_types is None:
        projection_types = ['max', 'mean', 'std']
    
    if reference_channels is None:
        reference_channels = ['1']
    
    # Step 1: Preprocess and detect Z-stacks
    logger.info("Step 1: Preprocessing and detecting Z-stacks")
    has_zstack, z_info = preprocess_plate_folder(input_folder)
    
    if not has_zstack:
        logger.warning(f"No Z-stacks detected in {input_folder}. Exiting.")
        return
    
    # Step 2: Find best focused images if requested
    logger.info("Step 2: Finding best focused images")
    best_focus_success, best_focus_dir = select_best_focus_zstack(
        input_folder,
        focus_wavelength=focus_wavelength,
        focus_method=focus_method
    )
    
    if best_focus_success:
        logger.info(f"Successfully created best focus images in {best_focus_dir}")
    else:
        logger.warning("Failed to create best focus images")
    
    # Step 3: Create projections if requested
    if create_projections:
        logger.info(f"Step 3: Creating projections: {projection_types}")
        projections_success, projections_dir = create_zstack_projections(
            input_folder,
            projection_types=projection_types,
            wavelengths='all'
        )
        
        if projections_success:
            logger.info(f"Successfully created projections in {projections_dir}")
        else:
            logger.warning("Failed to create projections")
    else:
        logger.info("Step 3: Skipping projection creation")
    
    # Step 4: Stitch using Z-aware methods
    logger.info(f"Step 4: Stitching using method: {stitch_method}")
    
    # Setup common stitching parameters
    stitch_params = {
        'reference_channels': reference_channels,
        'tile_overlap': tile_overlap,
        'max_shift': max_shift
    }
    
    # Perform stitching with Z-awareness
    stitch_success = stitch_across_z(
        input_folder,
        reference_z=stitch_method,
        **stitch_params
    )
    
    if stitch_success:
        logger.info("Stitching completed successfully")
    else:
        logger.error("Stitching failed")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Z-stack processing completed in {elapsed_time:.2f} seconds")

def process_full_workflow(input_folder, 
                         focus_wavelength='1', 
                         focus_method='combined',
                         create_projections=True,
                         projection_types=None,
                         stitch_method='best_focus',
                         reference_channels=None,
                         tile_overlap=10,
                         max_shift=50):
    """
    Process the full Z-stack workflow with enhanced handling.
    
    Args:
        input_folder: Path to the plate folder
        focus_wavelength: Wavelength to use for focus detection
        focus_method: Focus detection method to use
        create_projections: Whether to create Z-stack projections
        projection_types: Types of projections to create
        stitch_method: Method for stitching ('best_focus' or z-index)
        reference_channels: Wavelengths to use as reference for stitching
        tile_overlap: Percentage overlap between tiles
        max_shift: Maximum shift allowed between tiles
    """
    start_time = time.time()
    logger.info(f"Starting full Z-stack workflow for {input_folder}")
    
    # Default values
    if projection_types is None:
        projection_types = ['max', 'mean']
    
    if reference_channels is None:
        reference_channels = ['1']
    
    # Use the enhanced modified_process_plate_folder function
    # which handles all Z-stack operations in one call
    success = modified_process_plate_folder(
        input_folder,
        focus_detect=True,
        focus_method=focus_method,
        create_projections=create_projections,
        projection_types=projection_types,
        stitch_z_reference=stitch_method,
        reference_channels=reference_channels,
        tile_overlap=tile_overlap,
        max_shift=max_shift
    )
    
    if success:
        logger.info("Full Z-stack workflow completed successfully")
    else:
        logger.error("Full Z-stack workflow encountered errors")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Full workflow completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Z-stack workflow test")
    
    parser.add_argument("input_folder", 
                       help="Path to the plate folder containing TimePoint_1")
    parser.add_argument("--focus-wavelength", "-fw", 
                       default="1", 
                       help="Wavelength to use for focus detection (default: 1, use 'all' for all wavelengths)")
    parser.add_argument("--focus-method", "-fm", 
                       default="combined", 
                       choices=["combined", "laplacian", "normalized_variance", "tenengrad"],
                       help="Focus detection method to use (default: combined)")
    parser.add_argument("--no-projections", "-np", 
                       action="store_true",
                       help="Skip creating projections")
    parser.add_argument("--projection-types", "-pt", 
                       default="max,mean",
                       help="Comma-separated list of projection types (default: max,mean)")
    parser.add_argument("--stitch-method", "-sm", 
                       default="best_focus",
                       help="Method for stitching (default: best_focus, or z-index number)")
    parser.add_argument("--reference-channels", "-rc", 
                       default="1",
                       help="Comma-separated list of reference channels for stitching (default: 1)")
    parser.add_argument("--tile-overlap", "-to", 
                       type=float, 
                       default=10.0,
                       help="Percentage overlap between tiles (default: 10.0)")
    parser.add_argument("--max-shift", "-ms", 
                       type=int, 
                       default=50,
                       help="Maximum shift in pixels allowed between tiles (default: 50)")
    parser.add_argument("--step-by-step", "-sbs", 
                       action="store_true",
                       help="Run each processing step individually for demonstration")
    
    args = parser.parse_args()
    
    # Process arguments
    projection_types = args.projection_types.split(',') if args.projection_types else ['max', 'mean']
    reference_channels = args.reference_channels.split(',')
    
    if args.step_by_step:
        # Run each step individually for demonstration
        process_z_stack(
            args.input_folder,
            focus_wavelength=args.focus_wavelength,
            focus_method=args.focus_method,
            create_projections=not args.no_projections,
            projection_types=projection_types,
            stitch_method=args.stitch_method,
            reference_channels=reference_channels,
            tile_overlap=args.tile_overlap,
            max_shift=args.max_shift
        )
    else:
        # Run the full workflow in one go
        process_full_workflow(
            args.input_folder,
            focus_wavelength=args.focus_wavelength,
            focus_method=args.focus_method,
            create_projections=not args.no_projections,
            projection_types=projection_types,
            stitch_method=args.stitch_method,
            reference_channels=reference_channels,
            tile_overlap=args.tile_overlap,
            max_shift=args.max_shift
        )