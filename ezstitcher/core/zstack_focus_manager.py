"""
Z-stack focus management module for ezstitcher.

This module provides a class for finding and selecting the best focus plane in Z-stacks.
"""

import logging
import os
import re
import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Union, Any, Tuple

from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.config import FocusConfig
from ezstitcher.core.filename_parser import detect_parser, ImageXpressFilenameParser

logger = logging.getLogger(__name__)


class ZStackFocusManager:
    """
    Handles Z-stack focus operations:
    - Finding the best focus plane in a Z-stack
    - Creating best focus images from Z-stacks
    """
    def __init__(self, config, focus_analyzer=None, fs_manager=None, filename_parser=None):
        """
        Initialize the ZStackFocusManager.

        Args:
            config: Configuration object with Z-stack processing settings
            focus_analyzer: Focus analyzer for focus detection
            fs_manager: File system manager for file operations
            filename_parser: Parser for microscopy filenames
        """
        self.config = config
        self.fs_manager = fs_manager or FileSystemManager()

        # Initialize the focus analyzer
        focus_config = getattr(config, 'focus_config', None) or FocusConfig(method=getattr(config, 'focus_method', 'combined'))
        self.focus_analyzer = focus_analyzer or FocusAnalyzer(focus_config)

        self.filename_parser = filename_parser

    def create_best_focus_images(self, input_dir, output_dir=None, focus_method='combined', focus_wavelength='all'):
        """
        Select the best focused image from each Z-stack and save to output directory.

        Args:
            input_dir (str or Path): Directory with Z-stack images
            output_dir (str or Path): Directory to save best focus images. If None, creates a directory named {plate_name}_best_focus
            focus_method (str): Focus detection method
            focus_wavelength (str): Wavelength to use for focus detection

        Returns:
            tuple: (success, output_dir) where success is a boolean and output_dir is the path to the output directory
        """
        input_dir = Path(input_dir)

        # If output_dir is None, create a directory named {plate_name}_best_focus
        if output_dir is None:
            plate_path = input_dir.parent if input_dir.name == "TimePoint_1" else input_dir
            parent_dir = plate_path.parent
            plate_name = plate_path.name
            best_focus_suffix = getattr(self.config, 'best_focus_dir_suffix', "_best_focus")
            output_dir = parent_dir / f"{plate_name}{best_focus_suffix}"

        # Create TimePoint_1 directory in output_dir
        timepoint_dir = getattr(self.config, 'timepoint_dir_name', "TimePoint_1")
        timepoint_dir = output_dir / timepoint_dir
        self.fs_manager.ensure_directory(timepoint_dir)

        # Check if folder contains Z-stack images
        has_zstack, z_indices_map = self._detect_zstack_images(input_dir)
        if not has_zstack:
            logger.warning(f"No Z-stack images found in {input_dir}")
            return False, None

        # Group images by well, site, and wavelength
        images_by_coordinates = defaultdict(list)

        # Auto-detect filename parser if not provided
        if self.filename_parser is None:
            # Get a sample of filenames from the input_dir
            all_files = self.fs_manager.list_image_files(input_dir)
            file_paths = [str(f) for f in all_files]
            if file_paths:
                self.filename_parser = detect_parser(file_paths)
                logger.info(f"Auto-detected parser: {self.filename_parser.__class__.__name__}")
            else:
                self.filename_parser = ImageXpressFilenameParser()
                logger.info("No files found, defaulting to ImageXpress parser")

        # Group Z-indices by coordinates
        for base_name, z_indices in z_indices_map.items():
            # Parse the base_name to extract well, site, and channel
            # The base_name is in the format "well_sXXX_wY"
            parts = base_name.split('_')
            if len(parts) >= 3 and parts[1].startswith('s') and parts[2].startswith('w'):
                well = parts[0]
                site = int(parts[1][1:])
                wavelength = int(parts[2][1:])

                # Create coordinates key
                coordinates = (well, site, wavelength)

                # Add to dictionary
                images_by_coordinates[coordinates] = (base_name, z_indices)
            else:
                # If we get here, we couldn't parse the base_name
                logger.warning(f"Could not parse coordinates from {base_name}")

        # Process each set of coordinates
        best_focus_results = {}

        # Filter by wavelength if specified
        if focus_wavelength != 'all':
            focus_wavelength = int(focus_wavelength)
            focus_coordinates = [coords for coords in images_by_coordinates.keys() if coords[2] == focus_wavelength]
        else:
            focus_coordinates = list(images_by_coordinates.keys())

        # Process each set of focus coordinates
        for coordinates in focus_coordinates:
            well, site, wavelength = coordinates
            base_name, z_indices = images_by_coordinates[coordinates]

            # Load all Z-stack images for this coordinate
            image_stack = []
            for z_index in sorted(z_indices):
                # Parse the base_name to extract well, site, and channel
                parts = base_name.split('_')
                if len(parts) >= 3 and parts[1].startswith('s') and parts[2].startswith('w'):
                    well = parts[0]
                    site = int(parts[1][1:])
                    channel = int(parts[2][1:])

                    # Construct the filename using the filename parser
                    filename = self.filename_parser.construct_filename(well, site, channel, z_index)
                    img_path = input_dir / filename

                    # If the file doesn't exist, try alternative extensions
                    if not img_path.exists():
                        for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                            alt_filename = self.filename_parser.construct_filename(well, site, channel, z_index, extension=ext)
                            alt_path = input_dir / alt_filename
                            if alt_path.exists():
                                img_path = alt_path
                                break
                else:
                    # Fallback if we can't parse the base_name
                    img_path = input_dir / f"{base_name}_z{z_index:03d}.tif"

                img = self.fs_manager.load_image(img_path)
                if img is not None:
                    image_stack.append(img)

            if not image_stack:
                logger.warning(f"No valid images found for {base_name}")
                continue

            # Find best focus using FocusAnalyzer
            best_img, best_z, scores = self.focus_analyzer.select_best_focus(image_stack, method=focus_method)
            z_index = sorted(z_indices)[best_z]

            # Save best focus image
            output_filename = f"{well}_s{site:03d}_w{wavelength}.tif"
            output_path = timepoint_dir / output_filename
            self.fs_manager.save_image(output_path, best_img)
            logger.info(f"Saved best focus image for {base_name} (z={z_index}) to {output_path}")

            # Store best Z-index for this coordinate
            best_focus_results[coordinates] = z_index

        # If focus_wavelength is not 'all', use the same Z-index for other wavelengths
        if focus_wavelength != 'all':
            for coordinates in images_by_coordinates.keys():
                well, site, wavelength = coordinates
                if wavelength != focus_wavelength:
                    # Find the corresponding focus coordinates
                    focus_coords = (well, site, focus_wavelength)
                    if focus_coords in best_focus_results:
                        # Use the same Z-index as the focus wavelength
                        best_z = best_focus_results[focus_coords]
                        base_name, z_indices = images_by_coordinates[coordinates]

                        # Load the image at the best Z-index
                        # Parse the base_name to extract well, site, and channel
                        parts = base_name.split('_')
                        if len(parts) >= 3 and parts[1].startswith('s') and parts[2].startswith('w'):
                            well = parts[0]
                            site = int(parts[1][1:])
                            channel = int(parts[2][1:])

                            # Construct the filename using the filename parser
                            filename = self.filename_parser.construct_filename(well, site, channel, best_z)
                            img_path = input_dir / filename

                            # If the file doesn't exist, try alternative extensions
                            if not img_path.exists():
                                for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                                    alt_filename = self.filename_parser.construct_filename(well, site, channel, best_z, extension=ext)
                                    alt_path = input_dir / alt_filename
                                    if alt_path.exists():
                                        img_path = alt_path
                                        break
                        else:
                            # Fallback if we can't parse the base_name
                            img_path = input_dir / f"{base_name}_z{best_z:03d}.tif"

                        img = self.fs_manager.load_image(img_path)
                        if img is not None:
                            # Save the image
                            output_filename = f"{well}_s{site:03d}_w{wavelength}.tif"
                            output_path = timepoint_dir / output_filename
                            self.fs_manager.save_image(output_path, img)
                            logger.info(f"Saved best focus image for {base_name} (z={best_z}) to {output_path}")
                            best_focus_results[coordinates] = best_z

        return len(best_focus_results) > 0, output_dir

    def find_best_focus(self, timepoint_dir, output_dir):
        """
        Find the best focus plane for each Z-stack and save it to the output directory.

        Args:
            timepoint_dir (Path): Path to the TimePoint_1 directory
            output_dir (Path): Path to the output directory

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure the output directory exists
            self.fs_manager.ensure_directory(output_dir)

            # Get all Z-stack info
            z_info = self._get_zstack_info(timepoint_dir)
            if not z_info:
                logger.error(f"No Z-stack info found in {timepoint_dir}")
                return False

            # Process each unique image stack
            for stack_id, stack_info in z_info.items():
                # Skip stacks with only one Z-plane
                if len(stack_info['z_indices']) <= 1:
                    # Just copy the single image to the output directory
                    for wavelength in stack_info['wavelengths']:
                        src_path = stack_info['files'][wavelength][stack_info['z_indices'][0]]
                        dst_path = output_dir / f"{stack_id}_w{wavelength}.tif"
                        shutil.copy(src_path, dst_path)
                        logger.info(f"Copied single Z-plane image to {dst_path}")
                    continue

                # Process each wavelength separately
                for wavelength in stack_info['wavelengths']:
                    # Load all images for this wavelength
                    images = []
                    for z_idx in stack_info['z_indices']:
                        img_path = stack_info['files'][wavelength][z_idx]
                        img = self.fs_manager.load_image(img_path)
                        if img is None:
                            logger.error(f"Failed to read image: {img_path}")
                            continue
                        images.append(img)

                    if not images:
                        logger.error(f"No images loaded for {stack_id}_w{wavelength}")
                        continue

                    # Find the best focus plane
                    best_idx, _ = self.focus_analyzer.find_best_focus(images, method=getattr(self.config, 'focus_method', 'combined'))
                    best_z = stack_info['z_indices'][best_idx]

                    # Save the best focus image
                    best_img = images[best_idx]
                    output_path = output_dir / f"{stack_id}_w{wavelength}.tif"
                    self.fs_manager.save_image(output_path, best_img)
                    logger.info(f"Saved best focus for {stack_id}_w{wavelength} to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Error in find_best_focus: {e}", exc_info=True)
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

    def _get_zstack_info(self, folder_path):
        """
        Get detailed information about Z-stacks in a folder.

        This is a helper method that delegates to a ZStackOrganizer if available,
        or implements the logic directly if not.

        Args:
            folder_path: Path to the folder to check

        Returns:
            dict: Dictionary mapping stack IDs to information about each stack
        """
        # Try to import ZStackOrganizer
        try:
            from ezstitcher.core.zstack_organizer import ZStackOrganizer
            organizer = ZStackOrganizer(self.config, self.filename_parser, self.fs_manager)
            return organizer.get_zstack_info(folder_path)
        except ImportError:
            # Implement the logic directly
            folder_path = Path(folder_path)

            # Detect Z-stack images
            has_zstack, z_indices_map = self._detect_zstack_images(folder_path)

            if not has_zstack:
                return {}

            # Get all image files
            all_files = self.fs_manager.list_image_files(folder_path)

            # Group files by stack ID, wavelength, and Z-index
            z_info = {}

            for img_file in all_files:
                metadata = self.filename_parser.parse_filename(str(img_file))

                if not metadata or 'z_index' not in metadata or metadata['z_index'] is None:
                    continue

                well = metadata['well']
                site = metadata['site']
                channel = metadata['channel']
                z_index = metadata['z_index']

                # Create a consistent base name for grouping
                stack_id = f"{well}_s{site:03d}"

                # Initialize the stack info if it doesn't exist
                if stack_id not in z_info:
                    z_info[stack_id] = {
                        'z_indices': sorted(z_indices_map.get(f"{stack_id}_w{channel}", [])),
                        'wavelengths': set(),
                        'files': {}
                    }

                # Add the wavelength
                z_info[stack_id]['wavelengths'].add(channel)

                # Initialize the wavelength dict if it doesn't exist
                if channel not in z_info[stack_id]['files']:
                    z_info[stack_id]['files'][channel] = {}

                # Add the file
                z_info[stack_id]['files'][channel][z_index] = img_file

            # Convert wavelengths sets to sorted lists
            for stack_id in z_info:
                z_info[stack_id]['wavelengths'] = sorted(z_info[stack_id]['wavelengths'])

            return z_info
