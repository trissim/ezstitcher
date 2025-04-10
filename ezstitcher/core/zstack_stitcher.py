"""
Z-stack stitching module for ezstitcher.

This module provides a class for stitching Z-stack images.
"""

import re
import os
import logging
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Callable, Tuple

from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.filename_parser import detect_parser, ImageXpressFilenameParser

logger = logging.getLogger(__name__)


class ZStackStitcher:
    """
    Handles Z-stack stitching operations:
    - Stitching across Z-planes using reference positions
    - Creating stitched images for each Z-plane
    """
    def __init__(self, config, fs_manager=None, filename_parser=None):
        """
        Initialize the ZStackStitcher.

        Args:
            config: Configuration object with Z-stack processing settings
            fs_manager: File system manager for file operations
            filename_parser: Parser for microscopy filenames
        """
        self.config = config
        self.fs_manager = fs_manager or FileSystemManager()
        self.filename_parser = filename_parser
        
    def stitch_across_z(self, plate_folder, reference_z=None, stitch_all_z_planes=True, processor=None, preprocessing_funcs=None):
        """
        Stitch all Z-planes in a plate using a reference Z-plane for positions.

        Args:
            plate_folder (str or Path): Path to the plate folder
            reference_z (str or callable, optional): Reference Z-plane to use for positions.
                If str: 'max', 'mean', or 'best_focus'
                If callable: A function that takes a Z-stack (list of images) and returns a 2D image
                If None: Uses the z_reference_function from the config
            stitch_all_z_planes (bool): Whether to stitch all Z-planes
            processor (PlateProcessor): Processor to use for stitching
            preprocessing_funcs (dict, optional): Dictionary mapping channels to preprocessing functions.
                These functions are applied to individual images before Z-stack processing.

        Returns:
            bool: True if successful, False otherwise
        """
        # Use provided preprocessing functions or the ones from initialization
        preprocessing_funcs = preprocessing_funcs or {}
        try:
            plate_path = Path(plate_folder)
            timepoint_dir = "TimePoint_1"
            timepoint_path = plate_path / timepoint_dir

            if not timepoint_path.exists():
                logger.error(f"{timepoint_dir} folder does not exist in {plate_folder}")
                return False

            # Check if folder contains Z-stack images
            has_zstack, z_indices_map = self._detect_zstack_images(timepoint_path)
            if not has_zstack:
                logger.warning(f"No Z-stack images found in {timepoint_path}")
                return False

            # Get all unique Z-indices
            all_z_indices = set()
            for base_name, indices in z_indices_map.items():
                all_z_indices.update(indices)
            z_indices = sorted(list(all_z_indices))
            logger.info(f"Found {len(z_indices)} Z-planes: {z_indices}")

            # Get reference positions
            parent_dir = plate_path.parent
            plate_name = plate_path.name

            # If reference_z is not provided, use the z_reference_function from the config
            if reference_z is None:
                # Use the reference function from the config
                reference_z = getattr(self.config, 'z_reference_function', 'max_projection')

            # For string names, convert to the appropriate function
            if isinstance(reference_z, str):
                logger.info(f"Using reference_z: {reference_z}")

                # Determine reference directory based on reference_z
                if reference_z == 'max':
                    reference_dir = parent_dir / f"{plate_name}_projections_max" / timepoint_dir
                    # Ensure the directory exists
                    self.fs_manager.ensure_directory(reference_dir)

                    # Create max projections for each Z-stack
                    logger.info(f"Creating max projections for reference")
                    self._create_projections(plate_path / timepoint_dir, reference_dir)
                elif reference_z == 'mean':
                    reference_dir = parent_dir / f"{plate_name}_projections_mean" / timepoint_dir
                    # Ensure the directory exists
                    self.fs_manager.ensure_directory(reference_dir)

                    # Create mean projections for each Z-stack
                    logger.info(f"Creating mean projections for reference")
                    self._create_projections(plate_path / timepoint_dir, reference_dir, projection_types=['mean'])
                elif reference_z == 'best_focus':
                    reference_dir = parent_dir / f"{plate_name}_best_focus" / timepoint_dir
                    # Ensure the directory exists
                    self.fs_manager.ensure_directory(reference_dir)

                    # Create best focus projections for each Z-stack
                    logger.info(f"Creating best focus projections for reference")
                    # For best focus, we need to use the find_best_focus method
                    self._find_best_focus(plate_path / timepoint_dir, reference_dir)
                else:
                    logger.error(f"Invalid reference_z string: {reference_z}. Must be 'max', 'mean', or 'best_focus'.")
                    return False
            elif callable(reference_z):
                # If reference_z is a function, we need to create a custom projection directory
                custom_proj_dir = parent_dir / f"{plate_name}_projections_custom" / timepoint_dir
                self.fs_manager.ensure_directory(custom_proj_dir)

                # Create custom projections for each Z-stack
                logger.info(f"Creating custom projections using provided function")

                # Process each Z-stack
                for base_name, z_indices in z_indices_map.items():
                    # Load all images in the Z-stack
                    image_stack = []
                    channel = '1'  # Default channel

                    for z_idx in sorted(z_indices):
                        # Check if images are in ZStep folders or have _z in the filename
                        zstep_folder = timepoint_path / f"ZStep_{z_idx}"
                        if zstep_folder.exists():
                            # Images are in ZStep folders
                            filename = f"{base_name}.tif"
                            file_path = zstep_folder / filename
                        else:
                            # Check if this is an Opera Phenix file
                            opera_match = re.match(r'(r\d{1,2}c\d{1,2}f\d+).*', base_name)
                            if opera_match:
                                # Opera Phenix format
                                opera_base = opera_match.group(1)
                                # Extract channel using the filename parser
                                metadata = self.filename_parser.parse_filename(base_name)
                                channel = str(metadata.get('channel', '1')) if metadata else '1'
                                filename = f"{opera_base}p{z_idx:02d}-ch{channel}sk1fk1fl1.tiff"
                                file_path = timepoint_path / filename
                            else:
                                # ImageXpress format
                                filename = f"{base_name}_z{z_idx:03d}.tif"
                                file_path = timepoint_path / filename
                                # Extract channel using the filename parser
                                metadata = self.filename_parser.parse_filename(base_name)
                                channel = str(metadata.get('channel', '1')) if metadata else '1'

                        if file_path.exists():
                            img = self.fs_manager.load_image(file_path)
                            if img is not None:
                                # Apply preprocessing if available for this channel
                                if channel in preprocessing_funcs:
                                    img = preprocessing_funcs[channel](img)
                                image_stack.append(img)

                    if not image_stack:
                        logger.warning(f"No images found for Z-stack {base_name}")
                        continue

                    # Apply the custom function to create the projection
                    try:
                        custom_projection = reference_z(image_stack)

                        # Save the custom projection
                        output_filename = f"{base_name}.tif"
                        output_path = custom_proj_dir / output_filename
                        self.fs_manager.save_image(output_path, custom_projection)
                        logger.info(f"Saved custom projection for {base_name} to {output_path}")
                    except Exception as e:
                        logger.error(f"Error creating custom projection for {base_name}: {e}")
                        continue

                # Use the custom projections directory as reference
                reference_dir = custom_proj_dir
            else:
                logger.error(f"Invalid reference_z type: {type(reference_z)}. Must be str or callable.")
                return False

            if not reference_dir.exists():
                logger.error(f"Reference directory does not exist: {reference_dir}")
                return False

            # Get positions from reference directory
            positions_dir = parent_dir / f"{plate_name}_positions"

            # If positions directory doesn't exist or is empty, we need to stitch the projections first
            if not positions_dir.exists() or not list(positions_dir.glob("*.csv")):
                logger.info(f"No position files found in {positions_dir}, stitching projections first")

                # Create a new processor with the same config as the one passed in
                if processor is None:
                    logger.error(f"No processor provided for stitching projections")
                    return False

                # Stitch the projections to generate position files
                success = processor.run(
                    str(reference_dir.parent)
                )

                if not success:
                    logger.error(f"Failed to stitch projections")
                    return False

                # Check if positions directory exists now
                # The positions directory might be named differently depending on the plate name
                # Try to find the positions directory by looking for directories with 'positions' in the name
                proj_positions_dirs = [d for d in parent_dir.glob(f"*positions*") if d.is_dir()]
                if not proj_positions_dirs:
                    logger.error(f"No positions directory found after stitching projections")
                    return False

                # Use the first positions directory found
                proj_positions_dir = proj_positions_dirs[0]
                logger.info(f"Found positions directory: {proj_positions_dir}")

                # Copy position files from projections positions directory to positions directory
                # Make sure the positions directory exists
                self.fs_manager.ensure_directory(positions_dir)

                # List all position files in the projections positions directory
                position_files = list(proj_positions_dir.glob("*.csv"))
                logger.info(f"Found {len(position_files)} position files in {proj_positions_dir}")

                # Copy all position files from the projections positions directory to the positions directory
                position_files_copied = 0
                for position_file in position_files:
                    # Get the well name from the position file name
                    well_name = position_file.stem.split('_')[0]
                    # Copy the position file to the positions directory
                    dest_file = positions_dir / f"{well_name}_w1.csv"
                    shutil.copy(position_file, dest_file)
                    position_files_copied += 1
                    logger.info(f"Copied position file from {position_file} to {dest_file}")

                # Check if position files were copied
                if position_files_copied == 0:
                    # Try to find position files in other directories
                    all_position_dirs = [d for d in parent_dir.glob(f"*positions*") if d.is_dir()]
                    logger.info(f"Found {len(all_position_dirs)} position directories: {all_position_dirs}")

                    # Try each directory
                    for pos_dir in all_position_dirs:
                        if pos_dir == positions_dir:
                            continue

                        position_files = list(pos_dir.glob("*.csv"))
                        logger.info(f"Found {len(position_files)} position files in {pos_dir}")

                        for position_file in position_files:
                            # Get the well name from the position file name
                            well_name = position_file.stem.split('_')[0]
                            # Copy the position file to the positions directory
                            dest_file = positions_dir / f"{well_name}_w1.csv"
                            shutil.copy(position_file, dest_file)
                            position_files_copied += 1
                            logger.info(f"Copied position file from {position_file} to {dest_file}")

                    if position_files_copied == 0:
                        logger.error(f"No position files found in any positions directory")
                        return False

                logger.info(f"Copied {position_files_copied} position files to {positions_dir}")

            # Get all position files
            position_files = list(positions_dir.glob("*.csv"))
            if not position_files:
                logger.error(f"No position files found in {positions_dir} after stitching projections")
                return False

            logger.info(f"Found {len(position_files)} position files in {positions_dir}")

            # Stitch each Z-plane using the reference positions
            stitched_dir = parent_dir / f"{plate_name}_stitched" / timepoint_dir
            self.fs_manager.ensure_directory(stitched_dir)

            # For each Z-plane, stitch all wells and wavelengths
            for z_index in z_indices:
                logger.info(f"Stitching Z-plane {z_index}")

                # For each position file (which corresponds to a well and wavelength)
                for pos_file in position_files:
                    # Extract well and wavelength from position file name
                    match = re.match(r'(.+)_w(\d+)\.csv', pos_file.name)
                    if not match:
                        logger.warning(f"Could not parse position file name: {pos_file.name}")
                        continue

                    well_pattern = match.group(1)
                    wavelength = match.group(2)

                    # Read positions from CSV
                    positions = []
                    with open(pos_file, 'r') as f:
                        for line in f:
                            # Parse the line format: "file: C01_s001_w1.tif; grid: (0, 0); position: (0.0, 0.0)"
                            if 'file:' in line and 'position:' in line:
                                # Extract the site number from the filename
                                file_part = line.split(';')[0].strip()
                                filename = file_part.split('file:')[1].strip()
                                site_match = re.search(r's(\d+)_', filename)
                                if site_match:
                                    site = int(site_match.group(1))

                                    # Extract the position coordinates
                                    pos_part = line.split('position:')[1].strip()
                                    pos_match = re.search(r'\((\d+\.\d+), (\d+\.\d+)\)', pos_part)
                                    if pos_match:
                                        x = float(pos_match.group(1))
                                        y = float(pos_match.group(2))
                                        positions.append((site, x, y))

                    if not positions:
                        logger.warning(f"No positions found in {pos_file}")
                        continue

                    # Get all tiles for this well, wavelength, and Z-plane
                    tiles = []
                    for site, x, y in positions:
                        # Construct the filename and path
                        # First, check if files are in ZStep folders
                        zstep_folder = timepoint_path / f"ZStep_{z_index}"
                        if zstep_folder.exists():
                            # Files are in ZStep folders
                            filename = f"{well_pattern}_s{site:03d}_w{wavelength}.tif"
                            file_path = zstep_folder / filename
                            logger.info(f"Looking for file in ZStep folder: {file_path}")
                        else:
                            # Use the filename parser to construct the filename
                            # Check if we need to auto-detect the parser
                            if not hasattr(self, 'filename_parser') or self.filename_parser is None:
                                # Import here to avoid circular imports
                                # Get a sample of filenames from the timepoint_path
                                sample_files = []
                                for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                                    sample_files.extend([str(f) for f in timepoint_path.glob(f"*{ext}")][:10])
                                if sample_files:
                                    self.filename_parser = detect_parser(sample_files)
                                    logger.info(f"Auto-detected parser: {self.filename_parser.__class__.__name__}")
                                else:
                                    self.filename_parser = ImageXpressFilenameParser()
                                    logger.info("No files found, defaulting to ImageXpress parser")

                            # Construct the filename using the filename parser
                            try:
                                filename = self.filename_parser.construct_filename(well_pattern, site, int(wavelength), z_index)
                                file_path = timepoint_path / filename
                                logger.info(f"Looking for file: {file_path}")

                                # If the file doesn't exist, try alternative extensions
                                if not file_path.exists():
                                    for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                                        alt_filename = self.filename_parser.construct_filename(well_pattern, site, int(wavelength), z_index, extension=ext)
                                        alt_path = timepoint_path / alt_filename
                                        if alt_path.exists():
                                            file_path = alt_path
                                            break
                            except Exception as e:
                                # Fallback to ImageXpress format if the parser fails
                                logger.warning(f"Error constructing filename: {e}. Falling back to default format.")
                                filename = f"{well_pattern}_s{site:03d}_w{wavelength}_z{z_index:03d}.tif"
                                file_path = timepoint_path / filename

                        if file_path.exists():
                            # Load the image
                            img = self.fs_manager.load_image(file_path)
                            if img is not None:
                                # Apply preprocessing if available for this channel
                                if wavelength in preprocessing_funcs:
                                    img = preprocessing_funcs[wavelength](img)
                                tiles.append((site, x, y, img))
                        else:
                            logger.warning(f"Tile not found: {file_path}")

                    if not tiles:
                        logger.warning(f"No tiles found for {well_pattern}_w{wavelength}_z{z_index}")
                        continue

                    # Stitch the tiles
                    logger.info(f"Stitching {len(tiles)} tiles for {well_pattern}_w{wavelength}_z{z_index}")

                    # Determine canvas size
                    max_x = max(x + img.shape[1] for _, x, _, img in tiles)
                    max_y = max(y + img.shape[0] for _, _, y, img in tiles)
                    canvas = np.zeros((int(max_y), int(max_x)), dtype=np.uint16)

                    # Place tiles on canvas
                    for site, x, y, img in tiles:
                        x_start, y_start = int(x), int(y)
                        x_end, y_end = x_start + img.shape[1], y_start + img.shape[0]

                        # Ensure we don't go out of bounds
                        x_end = min(x_end, canvas.shape[1])
                        y_end = min(y_end, canvas.shape[0])

                        # Place the tile
                        canvas[y_start:y_end, x_start:x_end] = img[:y_end-y_start, :x_end-x_start]

                    # Save the stitched image
                    output_filename = f"{well_pattern}_w{wavelength}_z{z_index:03d}.tif"
                    output_path = stitched_dir / output_filename
                    self.fs_manager.save_image(output_path, canvas)
                    logger.info(f"Saved stitched image to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Error in stitch_across_z: {e}", exc_info=True)
            return False
            
    def _detect_zstack_images(self, folder_path):
        """
        Detect if a folder contains Z-stack images based on filename patterns.
        
        This is a helper method that delegates to a ZStackOrganizer if available,
        or implements the detection logic directly if not.
        
        Args:
            folder_path: Path to the folder to check
            
        Returns:
            tuple: (has_zstack, z_indices_map)
        """
        # Try to import ZStackOrganizer
        try:
            from ezstitcher.core.zstack_organizer import ZStackOrganizer
            organizer = ZStackOrganizer(self.config, self.filename_parser, self.fs_manager)
            return organizer.detect_zstack_images(folder_path)
        except ImportError:
            # Implement detection logic directly
            folder_path = Path(folder_path)
            all_files = self.fs_manager.list_image_files(folder_path)
            
            # Auto-detect parser if needed
            if self.filename_parser is None:
                file_paths = [str(f) for f in all_files]
                if file_paths:
                    self.filename_parser = detect_parser(file_paths)
                else:
                    self.filename_parser = ImageXpressFilenameParser()
                    
            z_indices = {}
            
            for img_file in all_files:
                metadata = self.filename_parser.parse_filename(str(img_file))
                
                if metadata and 'z_index' in metadata and metadata['z_index'] is not None:
                    well = metadata['well']
                    site = metadata['site']
                    channel = metadata['channel']
                    
                    base_name = f"{well}_s{site:03d}_w{channel}"
                    
                    if base_name not in z_indices:
                        z_indices[base_name] = []
                        
                    z_indices[base_name].append(metadata['z_index'])
            
            # Sort z_indices
            for base_name in z_indices:
                z_indices[base_name].sort()
                
            has_zstack = len(z_indices) > 0
            
            return has_zstack, z_indices
            
    def _create_projections(self, input_dir, output_dir, projection_types=None):
        """
        Create projections from Z-stack images.
        
        This is a helper method that delegates to a ZStackProjector if available,
        or implements the projection creation directly if not.
        
        Args:
            input_dir: Directory containing Z-stack images
            output_dir: Directory to save projections
            projection_types: List of projection types to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Try to import ZStackProjector
        try:
            from ezstitcher.core.zstack_projector import ZStackProjector
            projector = ZStackProjector(self.config, None, self.fs_manager, self.filename_parser)
            return projector.create_projections(input_dir, output_dir, projection_types)[0]
        except ImportError:
            # Implement projection creation directly
            logger.error("ZStackProjector not available, cannot create projections")
            return False
            
    def _find_best_focus(self, input_dir, output_dir):
        """
        Find the best focus plane for each Z-stack.
        
        This is a helper method that delegates to a ZStackFocusManager if available,
        or implements the best focus selection directly if not.
        
        Args:
            input_dir: Directory containing Z-stack images
            output_dir: Directory to save best focus images
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Try to import ZStackFocusManager
        try:
            from ezstitcher.core.zstack_focus_manager import ZStackFocusManager
            focus_manager = ZStackFocusManager(self.config, None, self.fs_manager, self.filename_parser)
            return focus_manager.find_best_focus(input_dir, output_dir)
        except ImportError:
            # Implement best focus selection directly
            logger.error("ZStackFocusManager not available, cannot find best focus")
            return False
