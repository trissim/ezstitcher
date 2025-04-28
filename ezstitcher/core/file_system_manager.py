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
    def save_image(file_path: Union[str, Path], image: np.ndarray,
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
    def copy_file(source_path: Union[str, Path], dest_path: Union[str, Path]) -> bool:
        """
        Copy a file from source to destination, preserving metadata.

        This method abstracts the file copying operation, ensuring that the destination
        directory exists and handling any errors that might occur. It preserves file
        metadata such as timestamps and permissions.

        Args:
            source_path (str or Path): Source file path
            dest_path (str or Path): Destination file path

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure destination directory exists
            directory = Path(dest_path).parent
            directory.mkdir(parents=True, exist_ok=True)

            # Copy file with metadata
            shutil.copy2(source_path, dest_path)
            return True
        except Exception as e:
            logger.error(f"Error copying file from {source_path} to {dest_path}: {e}")
            return False

    @staticmethod
    def remove_directory(directory_path: Union[str, Path], recursive: bool = True) -> bool:
        """
        Remove a directory and optionally all its contents.

        This method abstracts directory removal operations, handling both recursive
        and non-recursive removal. It provides error handling and logging for
        directory removal operations.

        Args:
            directory_path (str or Path): Path to the directory to remove
            recursive (bool): Whether to remove the directory recursively

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import shutil
            directory_path = Path(directory_path)

            if recursive:
                shutil.rmtree(directory_path)
            else:
                directory_path.rmdir()

            return True
        except Exception as e:
            logger.error(f"Error removing directory {directory_path}: {e}")
            return False

    @staticmethod
    def empty_directory(directory_path: Union[str, Path]) -> bool:
        """
        Empty a directory by recursively deleting all its contents.

        This method removes all files and subdirectories within the specified directory
        but preserves the directory itself. It provides error handling and logging for
        directory emptying operations.

        Args:
            directory_path (str or Path): Path to the directory to empty

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            directory_path = Path(directory_path)

            if not directory_path.exists() or not directory_path.is_dir():
                logger.error(f"Cannot empty {directory_path}: Not a valid directory")
                return False

            # Iterate through all entries in the directory
            for item in directory_path.iterdir():
                if item.is_file() or item.is_symlink():
                    # Remove files and symlinks
                    item.unlink()
                elif item.is_dir():
                    # Recursively remove subdirectories
                    import shutil
                    shutil.rmtree(item)

            return True
        except Exception as e:
            logger.error(f"Error emptying directory {directory_path}: {e}")
            return False


    @staticmethod
    def find_file_recursive(directory: Union[str, Path], filename: str) -> Optional[Path]:
        """
        Recursively search for a file by name in a directory and its subdirectories.
        Returns the first instance found.

        Args:
            directory (str or Path): Directory to search in
            filename (str): Name of the file to find

        Returns:
            Path or None: Path to the first instance of the file, or None if not found
        """
        try:
            directory = Path(directory)

            # Check if the file exists in the current directory
            file_path = directory / filename
            if file_path.exists() and file_path.is_file():
                logger.debug(f"Found file {filename} in {directory}")
                return file_path

            # Recursively search in subdirectories
            for item in directory.iterdir():
                if item.is_dir():
                    result = FileSystemManager.find_file_recursive(item, filename)
                    if result is not None:
                        return result

            # File not found in this directory or its subdirectories
            return None
        except Exception as e:
            logger.error(f"Error searching for file {filename} in {directory}: {e}")
            return None

    @staticmethod
    def find_directory_substring_recursive(start_path: Union[str, Path], substring: str) -> Optional[Path]:
        """
        Recursively search for a directory containing a substring in its name.
        Returns the path to the first directory found, or None if not found.

        Args:
            start_path (str or Path): The directory path to start the search from.
            substring (str): The substring to search for in directory names.

        Returns:
            Path or None: Path to the first matching directory, or None if not found.
        """
        try:
            start_path = Path(start_path)

            for root, dirs, files in os.walk(start_path):
                for dir_name in dirs:
                    if substring in dir_name:
                        found_dir_path = Path(root) / dir_name
                        logger.debug(f"Found directory with substring '{substring}': {found_dir_path}")
                        return found_dir_path

            # Directory not found
            logger.debug(f"No directory found containing substring '{substring}' starting from {start_path}")
            return None
        except Exception as e:
            logger.error(f"Error searching for directory with substring '{substring}' in {start_path}: {e}")
            return None




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

    @staticmethod
    def organize_zstack_folders(plate_folder, filename_parser):
        """
        Organize Z-stack folders by moving files to the plate folder with proper naming.

        Args:
            plate_folder (str or Path): Path to the plate folder
            filename_parser (FilenameParser): Parser for microscopy filenames (required)

        Returns:
            bool: True if Z-stack was organized, False otherwise

        Raises:
            ValueError: If filename_parser is None
        """
        # Ensure parser is provided
        if filename_parser is None:
            raise ValueError("A FilenameParser instance must be provided")

        has_zstack_folders, z_folders = FileSystemManager.detect_zstack_folders(plate_folder)
        if not has_zstack_folders:
            return False

        plate_path = FileSystemManager.find_image_directory(plate_folder)

        # Process each Z-stack folder
        for z_index, z_folder in z_folders:
            # Get all image files in this folder
            image_files = FileSystemManager.list_image_files(z_folder)

            for img_file in image_files:
                # Parse the filename
                metadata = filename_parser.parse_filename(str(img_file))
                if not metadata:
                    continue

                # Construct new filename with Z-index
                new_name = filename_parser.construct_filename(
                    well=metadata['well'],
                    site=metadata['site'],
                    channel=metadata['channel'],
                    z_index=z_index,
                    extension=metadata['extension']
                )

                # Copy file to plate folder
                new_path = plate_path / new_name
                FileSystemManager.copy_file(img_file, new_path)

        # Remove Z-stack folders
        for _, z_folder in z_folders:
            FileSystemManager.remove_directory(z_folder)

        return True

    @staticmethod
    def delete_file(file_path: Union[str, Path]) -> bool:
        """
        Delete a file from the file system.

        This method abstracts the file deletion operation, handling any errors that might occur.
        It provides proper error handling and logging for file deletion operations.

        Args:
            file_path (str or Path): Path to the file to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)

            # Check if the file exists
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return False

            # Check if it's a file (not a directory)
            if not file_path.is_file():
                logger.error(f"Not a file: {file_path}")
                return False

            # Delete the file
            file_path.unlink()
            logger.debug(f"Deleted file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False

    #### SMELLY ####
    #### becoming god class ####
    @staticmethod
    def mirror_directory_with_symlinks(source_dir: Union[str, Path],
                                      target_dir: Union[str, Path],
                                      recursive: bool = True,
                                      overwrite: bool = True) -> int:
        """
        Mirror a directory structure from source to target and create symlinks to all files.
        If the target directory exists and overwrite is True, it will be deleted and recreated.

        Args:
            source_dir (str or Path): Path to the source directory to mirror
            target_dir (str or Path): Path to the target directory where the mirrored structure will be created
            recursive (bool, optional): Whether to recursively mirror subdirectories. Defaults to True.
            overwrite (bool, optional): Whether to overwrite the target directory if it exists. Defaults to True.

        Returns:
            int: Number of symlinks created
        """
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)

        # Ensure source directory exists
        if not source_dir.is_dir():
            logger.error(f"Source directory not found: {source_dir}")
            return 0

        # If target directory exists and overwrite is True, delete it
        if target_dir.exists() and overwrite:
            logger.info(f"Removing existing target directory: {target_dir}")
            try:
                shutil.rmtree(target_dir)
            except Exception as e:
                logger.error(f"Error removing target directory {target_dir}: {e}")
                logger.info("Continuing without removing the directory...")

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Counter for created symlinks
        symlinks_created = 0

        # Get all items in the source directory
        try:
            items = list(source_dir.iterdir())
            total_items = len(items)
            print(f"Found {total_items} items in {source_dir}")
            sys.stdout.flush()

            # Process all items
            for i, item in enumerate(items):
                # Log progress every 100 items
                if i > 0 and i % 100 == 0:
                    print(f"Processed {i}/{total_items} items ({(i/total_items)*100:.1f}%)")
                    sys.stdout.flush()

                # Handle subdirectories
                if item.is_dir() and recursive:
                    symlinks_created += FileSystemManager.mirror_directory_with_symlinks(
                        item, target_dir / item.name, recursive, False  # Don't overwrite subdirectories
                    )
                    continue

                # Skip non-files
                if not item.is_file():
                    continue

                # Create symlink
                target_path = target_dir / item.name

                try:
                    # Remove existing symlink if it exists
                    if target_path.exists():
                        target_path.unlink()

                    # Create new symlink
                    os.symlink(item.resolve(), target_path)
                    symlinks_created += 1
                except Exception as e:
                    logger.error(f"Error creating symlink from {item} to {target_path}: {e}")

            print(f"Completed processing all {total_items} items in {source_dir}")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error processing directory {source_dir}: {e}")
            print(f"Error processing directory {source_dir}: {e}")
            sys.stdout.flush()

        return symlinks_created