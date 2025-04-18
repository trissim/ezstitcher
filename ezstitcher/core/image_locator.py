"""
Image locator module for ezstitcher.

This module provides a class for locating images in various directory structures.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Pattern

logger = logging.getLogger(__name__)


class ImageLocator:
    """
    Locates images in various directory structures.
    """

    DEFAULT_EXTENSIONS = ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

    @staticmethod
    def find_images_in_directory(directory: Union[str, Path],
                                extensions: Optional[List[str]] = None,
                                recursive: bool = True,
                                filename_parser: Optional[Any] = None # filename_parser is unused here now
                                ) -> List[Path]:
        """
        Find all images in a directory.

        Args:
            directory: Directory to search
            extensions: List of file extensions to include. If None, uses DEFAULT_EXTENSIONS.
            recursive: Whether to search recursively in subdirectories
            filename_parser: Optional filename parser (currently unused in this method)

        Returns:
            List of Path objects for image files
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []

        if extensions is None:
            extensions = ImageLocator.DEFAULT_EXTENSIONS

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


    @staticmethod
    def find_images_by_pattern(directory: Union[str, Path],
                              pattern: Union[str, Pattern],
                              extensions: Optional[List[str]] = None) -> List[Path]:
        """
        Find images matching a pattern in a directory.

        Args:
            directory: Directory to search
            pattern: Regex pattern to match
            extensions: List of file extensions to include. If None, uses DEFAULT_EXTENSIONS.

        Returns:
            List of Path objects for matching image files
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []

        if extensions is None:
            extensions = ImageLocator.DEFAULT_EXTENSIONS

        # Compile pattern if it's a string
        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        image_files = []
        for ext in extensions:
            for file_path in directory.glob(f"*{ext}"):
                if pattern.search(file_path.name):
                    image_files.append(file_path)

        return sorted(image_files)


    @staticmethod
    def find_z_stack_dirs(root_dir: Union[str, Path],
                         pattern: str = r"ZStep_\d+",
                         recursive: bool = True) -> List[Tuple[int, Path]]:
        """
        Find directories matching a pattern (default: ZStep_#) recursively.

        Args:
            root_dir: Root directory to start the search
            pattern: Regex pattern to match directory names (default: ZStep_ followed by digits)
            recursive: Whether to search recursively in subdirectories

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
    def find_image_locations(plate_folder: Union[str, Path],
                            extensions: Optional[List[str]] = None) -> Dict[str, List[Path]]:
        """
        Find all image files recursively within plate_folder.

        Args:
            plate_folder: Path to the plate folder
            extensions: List of file extensions to include. If None, uses DEFAULT_EXTENSIONS.

        Returns:
            Dictionary with all images found in the plate folder
        """
        plate_folder = Path(plate_folder)
        if extensions is None:
            extensions = ImageLocator.DEFAULT_EXTENSIONS

        # Find all image files recursively
        all_images = ImageLocator.find_images_in_directory(plate_folder, extensions, recursive=True)

        # Simple dictionary with all images
        return {'all': all_images}

    @staticmethod
    def find_image_directory(plate_folder: Union[str, Path], extensions: Optional[List[str]] = None) -> Path:
        """
        Find the directory where images are actually located.

        Handles both cases:
        1. Images directly in a folder (returns that folder)
        2. Images split across ZStep folders (returns parent of ZStep folders)

        Args:
            plate_folder: Base directory to search
            extensions: List of file extensions to include. If None, uses DEFAULT_EXTENSIONS.

        Returns:
            Path to the directory containing images
        """
        plate_folder = Path(plate_folder)
        if not plate_folder.exists():
            return plate_folder

        # First check if we have ZStep folders
        z_stack_dirs = ImageLocator.find_z_stack_dirs(plate_folder)
        if z_stack_dirs:
            # Check if there are images in the ZStep folders
            for _, z_dir in z_stack_dirs:
                if ImageLocator.find_images_in_directory(z_dir, extensions, recursive=False):
                    # Return the parent directory of the first ZStep folder with images
                    return z_dir.parent

        # If no ZStep folders with images, find all images recursively
        images = ImageLocator.find_images_in_directory(plate_folder, extensions, recursive=True)

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