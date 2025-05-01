"""
File system manager for ezstitcher.

This module provides a class for managing file system operations.
"""

import os
import re
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Pattern
import tifffile
import numpy as np
import shutil
#import imagecodecs
#import imagecodecs  # Import imagecodecs for OperaPhenix TIFF reading

logger = logging.getLogger(__name__)


class FileSystemManager:
    """
    Manages file system operations for ezstitcher.
    Abstracts away direct file system interactions for improved testability.
    """

    default_extensions = ['.tif', '.TIF', '.tiff', '.TIFF',
                          '.jpg', '.JPG', '.jpeg', '.JPEG',
                          '.png', '.PNG']

    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            directory (str or Path): Directory path to ensure exists

        Returns:
            Path: Path object for the directory
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def list_image_files(directory: Union[str, Path],
                         extensions: Optional[List[str]] = None,
                         recursive: bool = True,
                         ) -> List[Path]:
        """
        List all image files in a directory with specified extensions.

        Args:
            directory (str or Path): Directory to search
            extensions (list): List of file extensions to include
            recursive (bool): Whether to search recursively

        Returns:
            list: List of Path objects for image files
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []

        if extensions is None:
            extensions = FileSystemManager.default_extensions

        # Regular directory search
        image_files = []
        for ext in extensions:
            if recursive:
                # Use ** for recursive search
                found_files = list(directory.glob(f"**/*{ext}"))
            else:
                # Use * for non-recursive search
                found_files = list(directory.glob(f"*{ext}"))

            image_files.extend(found_files)

        return sorted(image_files)

    # Removed path_list_from_pattern - use pattern_matcher.path_list_from_pattern directly

    @staticmethod
    def load_image(file_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load an image. Only 2D images are supported.

        Args:
            file_path (str or Path): Path to the image file

        Returns:
            numpy.ndarray: 2D image or None if loading fails
        """
        try:
            img = tifffile.imread(str(file_path))

            # Check if image is 3D and raise an error
            if img.ndim == 3:
                raise ValueError("3D images are not supported. Only 2D images can be loaded.")

            return img
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None

    @staticmethod
    def save_image(image: np.ndarray, file_path: Union[str, Path],
                  compression: Optional[str] = None) -> bool:
        """
        Save an image to disk.

        Args:
            file_path (str or Path): Path to save the image
            image (numpy.ndarray): Image to save
            compression (str or None): Compression method

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            directory = Path(file_path).parent
            directory.mkdir(parents=True, exist_ok=True)

            # Save image
            tifffile.imwrite(str(file_path), image, compression=compression)
            return True
        except Exception as e:
            logger.error(f"Error saving image {file_path}: {e}")
            return False

    @staticmethod
    def rename_files_with_consistent_padding(directory, parser, width=3, force_suffixes=False):
        """
        Rename files in a directory to have consistent site number and Z-index padding.
        Optionally force the addition of missing optional suffixes (site, channel, z-index).

        Args:
            directory (str or Path): Directory containing files to rename
            parser (FilenameParser): Parser to use for filename parsing and padding (required)
            width (int, optional): Width to pad site numbers to
            force_suffixes (bool, optional): If True, add missing optional suffixes with default values

        Returns:
            dict: Dictionary mapping original filenames to new filenames

        Raises:
            ValueError: If parser is None
        """
        directory = Path(directory)

        # Ensure parser is provided
        if parser is None:
            raise ValueError("A FilenameParser instance must be provided")

        # Find all image files
        image_files = FileSystemManager.list_image_files(directory, recursive=False)

        # Map original filenames to reconstructed filenames
        rename_map = {}
        for file_path in image_files:
            original_name = file_path.name

            # Parse the filename components
            metadata = parser.parse_filename(original_name)
            if not metadata:
                raise ValueError(f"Could not parse filename: {original_name}")

            # Reconstruct the filename with proper padding
            # If force_suffixes is True, add default values for missing components
            if force_suffixes:
                # Default values for missing components
                site = metadata['site'] or 1
                channel = metadata['channel'] or 1
                z_index = metadata['z_index'] or 1
            else:
                # Use existing values or None
                site = metadata.get('site')
                channel = metadata.get('channel')
                z_index = metadata.get('z_index')

            # Reconstruct the filename with proper padding
            new_name = parser.construct_filename(
                well=metadata['well'],
                site=site,
                channel=channel,
                z_index=z_index,
                extension=metadata['extension'],
                site_padding=width,
                z_padding=width
            )

            # Add to rename map if different
            if original_name != new_name:
                rename_map[original_name] = new_name

        # Perform the renaming
        for original_name, new_name in rename_map.items():
            original_path = directory / original_name
            new_path = directory / new_name

            try:
                original_path.rename(new_path)
                logger.debug(f"Renamed {original_path} to {new_path}")
            except Exception as e:
                logger.error(f"Error renaming {original_path} to {new_path}: {e}")

        return rename_map

    @staticmethod
    def find_z_stack_dirs(root_dir: Union[str, Path],
                         pattern: str = r"ZStep_\d+",
                         recursive: bool = True) -> List[Tuple[int, Path]]:
        """
        Find directories matching a pattern (default: ZStep_#) recursively.

        Args:
            root_dir (str or Path): Root directory to start the search
            pattern (str): Regex pattern to match directory names (default: ZStep_ followed by digits)
            recursive (bool): Whether to search recursively in subdirectories

        Returns:
            List of (z_index, directory) tuples where z_index is extracted from the pattern
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            logger.warning(f"Directory does not exist: {root_dir}")
            return []

        z_stack_dirs = []
        z_pattern = re.compile(pattern)

        # Walk through directory structure
        for dirpath, dirnames, _ in os.walk(root_dir):
            # Process each directory at this level
            for dirname in dirnames:
                if z_pattern.search(dirname):
                    dir_path = Path(dirpath) / dirname
                    # Extract z-index from directory name (default to 0 if not found)
                    try:
                        digits_match = re.search(r'\d+', dirname)
                        z_index = int(digits_match.group(0)) if digits_match else 0
                    except (ValueError, IndexError):
                        z_index = 0

                    z_stack_dirs.append((z_index, dir_path))

            # Stop recursion if not requested
            if not recursive:
                break

        # Sort by Z-index
        z_stack_dirs.sort(key=lambda x: x[0])

        logger.debug(f"Found {len(z_stack_dirs)} directories matching pattern '{pattern}'")
        return z_stack_dirs

    @staticmethod
    def find_image_directory(plate_folder: Union[str, Path], extensions: Optional[List[str]] = None) -> Path:
        """
        Find the directory where images are actually located.

        Handles both cases:
        1. Images directly in a folder (returns that folder)
        2. Images split across ZStep folders (returns parent of ZStep folders)

        Args:
            plate_folder (str or Path): Base directory to search
            extensions (list): List of file extensions to include. If None, uses default_extensions.

        Returns:
            Path: Path to the directory containing images
        """
        plate_folder = Path(plate_folder)
        if not plate_folder.exists():
            return plate_folder

        # First check if we have ZStep folders
        z_stack_dirs = FileSystemManager.find_z_stack_dirs(plate_folder)
        if z_stack_dirs:
            # Check if there are images in the ZStep folders
            for _, z_dir in z_stack_dirs:
                if FileSystemManager.list_image_files(z_dir, extensions, recursive=False):
                    # Return the parent directory of the first ZStep folder with images
                    return z_dir.parent

        # If no ZStep folders with images, find all images recursively
        images = FileSystemManager.list_image_files(plate_folder, extensions, recursive=True)

        # If no images found, return original folder
        if not images:
            return plate_folder

        # Count images by parent directory
        dir_counts = {}
        for img in images:
            parent = img.parent
            dir_counts[parent] = dir_counts.get(parent, 0) + 1

        # Return directory with most images
        return max(dir_counts.items(), key=lambda x: x[1])[0]

    @staticmethod
    def detect_zstack_folders(plate_folder, pattern=None):
        """
        Detect Z-stack folders in a plate folder.

        Args:
            plate_folder (str or Path): Path to the plate folder
            pattern (str or Pattern, optional): Regex pattern to match Z-stack folders

        Returns:
            tuple: (has_zstack, z_folders) where z_folders is a list of (z_index, folder_path) tuples
        """

        plate_path = FileSystemManager.find_image_directory(Path(plate_folder))

        # Use find_z_stack_dirs to find Z-stack directories
        z_folders = FileSystemManager.find_z_stack_dirs(
            plate_path,
            pattern=pattern or r'ZStep_\d+',
            recursive=False  # Only look in the immediate directory
        )

        return bool(z_folders), z_folders