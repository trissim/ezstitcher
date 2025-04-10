"""
Z-stack organization module for ezstitcher.

This module provides a class for detecting and organizing Z-stack folders and images.
"""

import re
import os
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any, Union

from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.filename_parser import FilenameParser, ImageXpressFilenameParser, detect_parser

logger = logging.getLogger(__name__)


class ZStackOrganizer:
    """
    Handles Z-stack detection and organization operations:
    - Detection of Z-stack folders
    - Detection of Z-stack images
    - Organization of Z-stack folders into a consistent structure
    """
    def __init__(self, config, filename_parser=None, fs_manager=None):
        """
        Initialize the ZStackOrganizer.

        Args:
            config: Configuration object with Z-stack processing settings
            filename_parser: Parser for microscopy filenames
            fs_manager: File system manager for file operations
        """
        self.config = config
        self.fs_manager = fs_manager or FileSystemManager()
        self.filename_parser = filename_parser
        self._z_indices = []

    def detect_zstack_folders(self, plate_folder: str) -> Tuple[bool, List[Tuple[int, Path]]]:
        """
        Detect Z-stack folders in a plate folder.

        Args:
            plate_folder (str): Path to the plate folder

        Returns:
            tuple: (has_zstack, z_folders) where z_folders is a list of (z_index, folder_path) tuples
        """
        logger.debug(f"Called detect_zstack_folders with plate_folder={plate_folder}")

        try:
            # Initialize directory structure manager
            dir_structure = self.fs_manager.initialize_dir_structure(plate_folder)
            logger.info(f"Detected directory structure: {dir_structure.structure_type}")

            # Get timepoint directory from directory structure manager
            timepoint_path = dir_structure.get_timepoint_dir()
            if not timepoint_path:
                logger.error(f"Timepoint directory not found in {plate_folder}")
                logger.debug("Returning (False, []) due to missing timepoint directory")
                return False, []

            # Get Z-stack directories from directory structure manager
            z_folders = dir_structure.get_z_stack_dirs()

            z_folders.sort(key=lambda x: x[0])

            has_zstack = len(z_folders) > 0
            if has_zstack:
                logger.info(f"Found {len(z_folders)} Z-stack folders in {plate_folder}")
                for z_index, folder in z_folders[:3]:  # Log first 3 for brevity
                    logger.info(f"Z-stack folder: {folder.name}, Z-index: {z_index}")

                # Store the Z-indices for later use
                self._z_indices = [z[0] for z in z_folders]
            else:
                logger.info(f"No Z-stack folders found in {plate_folder}")

            logger.debug(f"Returning ({has_zstack}, z_folders with {len(z_folders)} items)")
            return has_zstack, z_folders

        except Exception as e:
            logger.error(f"Exception in detect_zstack_folders: {e}", exc_info=True)
            raise

    def detect_zstack_images(self, folder_path: Union[str, Path]) -> Tuple[bool, Dict[str, List[int]]]:
        """
        Detect if a folder contains Z-stack images based on filename patterns.

        Args:
            folder_path (str or Path): Path to the folder

        Returns:
            tuple: (has_zstack, z_indices_map) where z_indices_map is a dict mapping base filenames to Z-indices
        """
        folder_path = Path(folder_path)

        # Use FileSystemManager to list image files
        all_files = self.fs_manager.list_image_files(folder_path)

        # Check if we need to auto-detect the parser
        if not hasattr(self, 'filename_parser') or self.filename_parser is None:
            # Import here to avoid circular imports
            file_paths = [str(f) for f in all_files]
            if file_paths:
                self.filename_parser = detect_parser(file_paths)
                logger.info(f"Auto-detected parser: {self.filename_parser.__class__.__name__}")
            else:
                self.filename_parser = ImageXpressFilenameParser()
                logger.info("No files found, defaulting to ImageXpress parser")

        z_indices = defaultdict(list)

        # Group files by their base components (well, site, channel)
        for img_file in all_files:
            # Parse the filename using the appropriate parser
            metadata = self.filename_parser.parse_filename(str(img_file))

            if metadata and 'z_index' in metadata and metadata['z_index'] is not None:
                # Create a base name without the Z-index
                well = metadata['well']
                site = metadata['site']
                channel = metadata['channel']

                # Create a consistent base name for grouping
                base_name = f"{well}_s{site:03d}_w{channel}"

                # Add the Z-index to the list for this base name
                z_indices[base_name].append(metadata['z_index'])
                logger.debug(f"Matched z-index: {img_file.name} -> base:{base_name}, z:{metadata['z_index']}")
            else:
                logger.debug(f"No z-index found for file: {img_file.name}")

        has_zstack = len(z_indices) > 0
        if has_zstack:
            for base_name in z_indices:
                z_indices[base_name].sort()

            logger.info(f"Found Z-stack images in {folder_path}")
            logger.info(f"Detected {len(z_indices)} unique image stacks")

            for i, (base_name, indices) in enumerate(list(z_indices.items())[:3]):
                logger.info(f"Example {i+1}: {base_name} has {len(indices)} z-planes: {indices}")
        else:
            logger.info(f"No Z-stack images detected in {folder_path}")

        return has_zstack, dict(z_indices)

    def organize_zstack_folders(self, plate_folder: Union[str, Path]) -> bool:
        """
        Organize Z-stack folders by moving files to TimePoint_1 with proper naming.

        Args:
            plate_folder (str or Path): Path to the plate folder

        Returns:
            bool: True if Z-stack was organized, False otherwise
        """
        has_zstack, z_folders = self.detect_zstack_folders(plate_folder)

        if not has_zstack:
            return False

        # Initialize directory structure manager
        dir_structure = self.fs_manager.initialize_dir_structure(plate_folder)
        logger.info(f"Detected directory structure: {dir_structure.structure_type}")

        # Get timepoint directory from directory structure manager
        timepoint_path = dir_structure.get_timepoint_dir()
        if not timepoint_path:
            # Create timepoint directory if it doesn't exist
            timepoint_dir = getattr(self.config, 'timepoint_dir_name', "TimePoint_1")
            plate_path = Path(plate_folder)
            timepoint_path = plate_path / timepoint_dir
            self.fs_manager.ensure_directory(timepoint_path)

        for z_index, z_folder in z_folders:
            logger.info(f"Processing Z-stack folder: {z_folder.name}")

            # Use FileSystemManager to list image files
            image_files = self.fs_manager.list_image_files(z_folder)

            for img_file in image_files:
                # Use the instance's filename parser to parse the filename
                if not hasattr(self, 'filename_parser') or self.filename_parser is None:
                    # Import here to avoid circular imports
                    file_paths = [str(img_file)]
                    self.filename_parser = detect_parser(file_paths)
                    logger.info(f"Auto-detected parser: {self.filename_parser.__class__.__name__}")

                metadata = self.filename_parser.parse_filename(str(img_file))

                if metadata:
                    well = metadata['well']
                    site = str(metadata['site']).zfill(3)
                    wavelength = str(metadata['channel'])
                    extension = os.path.splitext(img_file.name)[1]

                    # Use the filename parser's construct_filename method for all microscope types
                    # This is the proper OOP approach - let the parser handle the format-specific details
                    new_name = self.filename_parser.construct_filename(well, int(site), int(wavelength), z_index, extension=extension)

                    new_path = timepoint_path / new_name

                    self.fs_manager.copy_file(img_file, new_path)
                    logger.info(f"Copied {img_file.name} to {new_path.name}")
                else:
                    logger.warning(f"Could not parse filename: {img_file.name}")

        for z_index, z_folder in z_folders:
            if self.fs_manager.remove_directory(z_folder):
                logger.info(f"Removed Z-stack folder: {z_folder}")
            else:
                logger.warning(f"Failed to remove Z-stack folder {z_folder}")

        return True

    def get_zstack_info(self, folder_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about Z-stacks in a folder.

        Args:
            folder_path (str or Path): Path to the folder containing Z-stack images

        Returns:
            dict: Dictionary mapping stack IDs to information about each stack
        """
        folder_path = Path(folder_path)

        # Detect Z-stack images
        has_zstack, z_indices_map = self.detect_zstack_images(folder_path)

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

    def get_z_indices(self) -> List[int]:
        """
        Return the list of detected Z indices.

        Returns:
            list: List of Z indices
        """
        return self._z_indices
