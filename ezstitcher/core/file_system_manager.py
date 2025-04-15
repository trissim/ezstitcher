"""
File system manager for ezstitcher.

This module provides a class for managing file system operations.
"""

import os
import re
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Pattern
import tifffile
import numpy as np

from ezstitcher.core.filename_parser import FilenameParser, ImageXpressFilenameParser
from ezstitcher.core.csv_handler import CSVHandler
from ezstitcher.core.pattern_matcher import PatternMatcher
from ezstitcher.core.image_locator import ImageLocator
from ezstitcher.core.opera_phenix_xml_parser import OperaPhenixXmlParser

logger = logging.getLogger(__name__)


class FileSystemManager:
    """
    Manages file system operations for ezstitcher.
    Abstracts away direct file system interactions for improved testability.
    """

    def __init__(self, config=None, filename_parser=None):
        """
        Initialize the FileSystemManager.

        Args:
            config (dict, optional): Configuration dictionary
            filename_parser (FilenameParser, optional): Parser for microscopy filenames
        """
        self.config = config or {}
        self.default_extensions = ['.tif', '.TIF', '.tiff', '.TIFF',
                                  '.jpg', '.JPG', '.jpeg', '.JPEG',
                                  '.png', '.PNG']
        self.filename_parser = filename_parser or ImageXpressFilenameParser()
        self.csv_handler = CSVHandler()
        self.pattern_matcher = PatternMatcher(self.filename_parser)

    def ensure_directory(self, directory: Union[str, Path]) -> Path:
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

    def list_image_files(self, directory: Union[str, Path],
                         extensions: Optional[List[str]] = None,
                         recursive: bool = False, # Add recursive argument
                         flatten: bool = False    # Add flatten argument
                         ) -> List[Path]:
        """
        List all image files in a directory with specified extensions.

        Args:
            directory (str or Path): Directory to search
            extensions (list): List of file extensions to include
            recursive (bool): Whether to search recursively
            flatten (bool): Whether to flatten Z-stack directories (implies recursive)

        Returns:
            list: List of Path objects for image files
        """
        if extensions is None:
            extensions = self.default_extensions

        # Use ImageLocator to find images, passing through arguments
        # Pass recursive and flatten arguments here
        return ImageLocator.find_images_in_directory(directory, extensions, recursive=recursive)

    # Removed path_list_from_pattern - use pattern_matcher.path_list_from_pattern directly

    def load_image(self, file_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load an image and ensure it's 2D grayscale.

        Args:
            file_path (str or Path): Path to the image file

        Returns:
            numpy.ndarray: 2D grayscale image or None if loading fails
        """
        try:
            img = tifffile.imread(str(file_path))

            # Convert to 2D grayscale if needed
            if img.ndim == 3:
                # Check if it's a channel-first format (C, H, W)
                if img.shape[0] <= 4:  # Assuming max 4 channels (RGBA)
                    # Convert channel-first to 2D by taking mean across channels
                    img = np.mean(img, axis=0).astype(img.dtype)
                # Check if it's a channel-last format (H, W, C)
                elif img.shape[2] <= 4:  # Assuming max 4 channels (RGBA)
                    # Convert channel-last to 2D by taking mean across channels
                    img = np.mean(img, axis=2).astype(img.dtype)
                else:
                    # If it's a 3D image with a different structure, use the first slice
                    img = img[0].astype(img.dtype)

            return img
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None

    def save_image(self, file_path: Union[str, Path], image: np.ndarray,
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

    def create_symlinks_from_pattern(self, source_dir: Union[str, Path], pattern: str, target_dir: Union[str, Path]) -> int:
        """
        Create symlinks in the target directory for all files matching a pattern in the source directory.

        This ensures that reference images are always available in the reference directory,
        even when no preprocessing or composition is performed.

        Args:
            source_dir (str or Path): Directory containing source files
            pattern (str): Pattern to match with {iii} placeholder for site index
            target_dir (str or Path): Directory to create symlinks in

        Returns:
            int: Number of symlinks created
        """
        try:
            source_dir = Path(source_dir)
            target_dir = Path(target_dir)
            self.ensure_directory(target_dir)

            # Get matching files
            matching_files = self.pattern_matcher.path_list_from_pattern(source_dir, pattern)
            if not matching_files:
                logger.warning(f"No files found matching pattern {pattern} in {source_dir}")
                return 0

            # Create symlinks
            count = 0
            for filename in matching_files:
                source_path = source_dir / filename
                target_path = target_dir / filename

                # Skip if target already exists
                if target_path.exists():
                    if target_path.is_symlink() and target_path.resolve() == source_path.resolve():
                        logger.debug(f"Symlink already exists: {target_path} -> {source_path}")
                        count += 1
                        continue
                    else:
                        logger.warning(f"Target file already exists and is not a symlink to source: {target_path}")
                        continue

                # Create symlink
                try:
                    target_path.symlink_to(source_path)
                    logger.debug(f"Created symlink: {target_path} -> {source_path}")
                    count += 1
                except Exception as e:
                    logger.error(f"Error creating symlink {target_path} -> {source_path}: {e}")

            logger.info(f"Created {count} symlinks in {target_dir}")
            return count

        except Exception as e:
            logger.error(f"Error creating symlinks from {source_dir} to {target_dir}: {e}")
            return 0

    def copy_file(self, source_path: Union[str, Path], dest_path: Union[str, Path]) -> bool:
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
            import shutil
            # Ensure destination directory exists
            directory = Path(dest_path).parent
            directory.mkdir(parents=True, exist_ok=True)

            # Copy file with metadata
            shutil.copy2(source_path, dest_path)
            return True
        except Exception as e:
            logger.error(f"Error copying file from {source_path} to {dest_path}: {e}")
            return False

    def remove_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> bool:
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

    def find_htd_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """
        Find the HTD file for a plate or Index.xml for Opera Phenix.

        Args:
            plate_path (str or Path): Path to the plate folder

        Returns:
            Path or None: Path to the HTD file, or None if not found
        """
        plate_path = Path(plate_path)

        # Look for Opera Phenix Index.xml file
        if hasattr(self.filename_parser, '__class__') and self.filename_parser.__class__.__name__ == 'OperaPhenixFilenameParser':
            # Check for Index.xml in the plate directory
            index_xml = plate_path / "Index.xml"
            if index_xml.exists():
                return index_xml

            # Check for Index.xml in parent directory
            parent_index_xml = plate_path.parent / "Index.xml"
            if parent_index_xml.exists():
                return parent_index_xml

        # Look for ImageXpress HTD file in plate directory
        htd_files = list(plate_path.glob("*.HTD"))
        if htd_files:
            for htd_file in htd_files:
                if 'plate' in htd_file.name.lower():
                    return htd_file
            return htd_files[0]

    def get_pixel_size(self, htd_path: Union[str, Path]) -> Optional[float]:
        """
        Get the pixel size from an HTD file or Index.xml file.

        Args:
            htd_path (str or Path): Path to the HTD file or Index.xml file

        Returns:
            float: Pixel size in micrometers, or None if not found
        """
        try:
            htd_path = Path(htd_path)

            # Check if this is an Opera Phenix Index.xml file
            if htd_path.name == "Index.xml":
                logger.info(f"Getting pixel size from Opera Phenix Index.xml file: {htd_path}")

                try:
                    # Use the OperaPhenixXmlParser to get the pixel size
                    xml_parser = OperaPhenixXmlParser(htd_path)
                    pixel_size = xml_parser.get_pixel_size()

                    if pixel_size > 0:
                        logger.info(f"Determined pixel size from Opera Phenix Index.xml: {pixel_size} µm")
                        return pixel_size

                    # If we couldn't determine the pixel size, use a default
                    logger.warning(f"Could not determine pixel size from Opera Phenix Index.xml, using default 0.65 µm")
                    return 0.65  # Default value in micrometers
                except Exception as e:
                    logger.error(f"Error getting pixel size from Opera Phenix Index.xml: {e}")
                    return 0.65  # Default value in micrometers

            # For ImageXpress HTD files, we don't have pixel size information
            # We would need to extract it from the image metadata
            logger.warning(f"Pixel size not available in HTD file: {htd_path}")
            return None
        except Exception as e:
            logger.error(f"Error getting pixel size: {e}")
            return None

    def parse_htd_file(self, htd_path: Union[str, Path]) -> Optional[Tuple[int, int]]:
        """
        Parse an HTD file or Index.xml file to extract grid dimensions.

        Args:
            htd_path (str or Path): Path to the HTD file or Index.xml file

        Returns:
            tuple: (grid_size_x, grid_size_y) or None if parsing fails
        """
        try:
            htd_path = Path(htd_path)

            # Check if this is an Opera Phenix Index.xml file
            if htd_path.name == "Index.xml":
                logger.info(f"Found Opera Phenix Index.xml file: {htd_path}")

                try:
                    # Use the OperaPhenixXmlParser to get the grid size
                    xml_parser = OperaPhenixXmlParser(htd_path)
                    grid_size = xml_parser.get_grid_size()

                    if grid_size[0] > 0 and grid_size[1] > 0:
                        logger.info(f"Determined grid size from Opera Phenix Index.xml: {grid_size[0]}x{grid_size[1]}")
                        return grid_size

                    # If we couldn't determine the grid size, use a default
                    logger.warning(f"Could not determine grid size from Opera Phenix Index.xml, using default 2x2")
                    return 2, 2
                except Exception as e:
                    logger.error(f"Error parsing Opera Phenix Index.xml: {e}")
                    # Use a default grid size for Opera Phenix
                    return 2, 2

            # For ImageXpress HTD files
            with open(htd_path, 'r') as f:
                htd_content = f.read()

            # Extract grid dimensions - try multiple formats
            # First try the new format with "XSites" and "YSites"
            cols_match = re.search(r'"XSites", (\d+)', htd_content)
            rows_match = re.search(r'"YSites", (\d+)', htd_content)

            # If not found, try the old format with SiteColumns and SiteRows
            if not (cols_match and rows_match):
                cols_match = re.search(r'SiteColumns=(\d+)', htd_content)
                rows_match = re.search(r'SiteRows=(\d+)', htd_content)

            if cols_match and rows_match:
                grid_size_x = int(cols_match.group(1))
                grid_size_y = int(rows_match.group(1))
                return grid_size_x, grid_size_y
            else:
                logger.warning(f"Could not parse grid dimensions from HTD file: {htd_path}")
                return None
        except Exception as e:
            logger.error(f"Error parsing HTD file {htd_path}: {e}")
            return None




    def clean_temp_folders(self, parent_dir: Union[str, Path], base_name: str, keep_suffixes=None) -> None:
        """
        Clean up temporary folders created during processing.

        Args:
            parent_dir (str or Path): Parent directory
            base_name (str): Base name of the plate folder
            keep_suffixes (list, optional): List of suffixes to keep
        """
        parent_dir = Path(parent_dir)
        if keep_suffixes is None:
            keep_suffixes = ['_stitched', '_positions']

        # Find all folders with the base name and a suffix
        for item in parent_dir.iterdir():
            if item.is_dir() and item.name.startswith(base_name) and item.name != base_name:
                # Check if the suffix should be kept
                suffix = item.name[len(base_name):]
                if suffix not in keep_suffixes:
                    logger.info(f"Removing temporary folder: {item}")
                    import shutil
                    shutil.rmtree(item)

    def create_output_directories(self, plate_path, suffixes):
        """
        Create output directories for a plate.

        Args:
            plate_path (str): Path to plate folder
            suffixes (dict): Dictionary mapping directory types to suffixes

        Returns:
            dict: Dictionary mapping directory types to Path objects
        """
        parent_dir = Path(plate_path).parent
        plate_name = Path(plate_path).name
        #parent_dir = Path(parent_dir)
        dirs = {}

        # Create directories for each suffix
        for dir_type, suffix in suffixes.items():
            dir_path = parent_dir / f"{plate_name}{suffix}"
            self.ensure_directory(dir_path)
            dirs[dir_type] = dir_path

        return dirs

    def get_or_detect_parser(self, directory: Union[str, Path]) -> Optional[FilenameParser]:
        """
        Get the configured parser or detect one from files in the directory.

        Args:
            directory (str or Path): Directory containing files to analyze

        Returns:
            FilenameParser or None: The parser to use, or None if detection fails
        """
        from ezstitcher.core.filename_parser import FilenameParser, create_parser
        from ezstitcher.core.image_locator import ImageLocator

        # Use the configured parser if available
        if self.filename_parser is not None:
            return self.filename_parser

        # Otherwise, try to detect format from files in directory
        directory = Path(directory)

        # Use ImageLocator to find all image files
        image_files = ImageLocator.find_images_in_directory(directory, recursive=False)

        if not image_files:
            logger.warning(f"No image files found in {directory}")
            return None

        # Get filenames only
        filenames = [f.name for f in image_files]

        # Detect format
        format_type = FilenameParser.detect_format(filenames)
        if format_type is None:
            logger.warning(f"Could not detect format for files in {directory}")
            return None

        # Create and return parser
        logger.info(f"Detected format {format_type} for files in {directory}")
        return create_parser(format_type)

    def rename_files_with_consistent_padding(self, directory, parser=None, width=3):
        """
        Rename files in a directory to have consistent site number and Z-index padding.

        Args:
            directory (str or Path): Directory containing files to rename
            parser (FilenameParser, optional): Parser to use for filename parsing and padding
            width (int, optional): Width to pad site numbers to
            z_width (int, optional): Width to pad Z-index numbers to

        Returns:
            dict: Dictionary mapping original filenames to new filenames
        """
        from ezstitcher.core.image_locator import ImageLocator

        directory = Path(directory)

        # Use provided parser or detect one
        if parser is None:
            parser = self.filename_parser or self.get_or_detect_parser(directory)
            if parser is None:
                return {}  # No parser available

        # Use ImageLocator to find all image files
        image_files = ImageLocator.find_images_in_directory(directory, recursive=False)

        # Map original filenames to reconstructed filenames
        rename_map = {}
        for file_path in image_files:
            original_name = file_path.name

            # Parse the filename components
            metadata = parser.parse_filename(original_name)
            if not metadata:
                continue  # Skip files that can't be parsed

            # Reconstruct the filename with proper padding
            new_name = parser.construct_filename(
                well=metadata['well'],
                site=metadata['site'],
                channel=metadata['channel'],
                z_index=metadata.get('z_index'),
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


        def convert_opera_phenix_to_imagexpress(self, input_dir, output_dir=None):
            """
            Convert Opera Phenix files to ImageXpress format.

            This function copies Opera Phenix files to a new directory with ImageXpress-style filenames.
            If output_dir is None, files are renamed in place.

            Args:
                input_dir (str or Path): Directory containing Opera Phenix files
                output_dir (str or Path, optional): Directory to save converted files

            Returns:
                bool: True if successful, False otherwise
            """
            try:
                import shutil
                from ezstitcher.core.filename_parser import OperaPhenixFilenameParser

                input_dir = Path(input_dir)
                if output_dir is None:
                    output_dir = input_dir
                else:
                    output_dir = Path(output_dir)
                    self.ensure_directory(output_dir)

                    # Create TimePoint_1 directory in the output directory
                    # This is needed because ImageXpress format expects images in a TimePoint_1 subdirectory
                    timepoint_dir = output_dir / "TimePoint_1"
                    self.ensure_directory(timepoint_dir)

                    # Update output_dir to point to the TimePoint_1 directory
                    output_dir = timepoint_dir

                # Create an Opera Phenix filename parser
                opera_parser = OperaPhenixFilenameParser()

                # Get all image files in the input directory
                image_files = self.list_image_files(input_dir)

                # Process each file
                for img_file in image_files:
                    # Parse the filename
                    metadata = opera_parser.parse_filename(str(img_file))

                    if metadata:
                        well = metadata['well']
                        site = metadata['site']
                        channel = metadata['channel']
                        z_index = metadata.get('z_index')
                        extension = img_file.suffix

                        # Construct new filename in ImageXpress format
                        # Convert Opera Phenix well format (R01C01) to ImageXpress well format (A01)
                        # Extract row and column from Opera Phenix well format
                        match = re.match(r"R(\d{2})C(\d{2})", well, re.I)
                        if match:
                            row = int(match.group(1))
                            col = int(match.group(2))
                            # Convert row number to letter (1 -> A, 2 -> B, etc.)
                            row_letter = chr(64 + row)  # ASCII: 'A' = 65
                            # Create ImageXpress well format
                            imx_well = f"{row_letter}{col:02d}"
                        else:
                            # If well is not in Opera Phenix format, use it as is
                            imx_well = well

                        # Construct new filename in ImageXpress format
                        if z_index is not None:
                            new_name = f"{imx_well}_s{site:03d}_w{channel}_z{z_index:03d}{extension}"
                        else:
                            new_name = f"{imx_well}_s{site:03d}_w{channel}{extension}"

                        # Create output path
                        output_path = output_dir / new_name

                        # Copy or rename the file
                        if output_dir == input_dir:
                            # Rename in place
                            img_file.rename(output_path)
                            logger.info(f"Renamed {img_file.name} to {new_name}")
                        else:
                            # Copy to new directory
                            shutil.copy2(img_file, output_path)
                            logger.info(f"Copied {img_file.name} to {new_name}")
                    else:
                        logger.warning(f"Could not parse filename: {img_file.name}")

                return True
            except Exception as e:
                logger.error(f"Error converting Opera Phenix files to ImageXpress format: {e}")
                return False

        def mirror_directory_structure(self, source_dir: Union[str, Path], base_output_dir: Union[str, Path]) -> Path:
            """
            Mirror the directory structure from source_dir to base_output_dir and copy non-image files.

            This method recursively creates a directory structure in base_output_dir that mirrors the structure
            in source_dir. It also copies all non-image files (HTD, XML, etc.) to the mirrored directory.

            Args:
                source_dir (str or Path): Source directory to mirror
                base_output_dir (str or Path): Base output directory where the mirrored structure will be created

            Returns:
                Path: Path to the mirrored directory
            """
            source_dir = Path(source_dir)
            base_output_dir = Path(base_output_dir)

            # Ensure base output directory exists
            self.ensure_directory(base_output_dir)

            # Walk through the source directory structure
            for root, dirs, files in os.walk(source_dir):
                # Get relative path from source_dir
                rel_path = Path(root).relative_to(source_dir)

                # Create directories
                for dir_name in dirs:
                    # Skip directories that might contain image files but don't need structure mirroring
                    if dir_name.lower() in ['thumbnails', 'thumb', 'thumbnail']:
                        continue
                    self.ensure_directory(base_output_dir / rel_path / dir_name)

                # Copy non-image files
                for file_name in files:
                    if not any(file_name.lower().endswith(ext) for ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']):
                        source_file = Path(root) / file_name
                        target_file = base_output_dir / rel_path / file_name
                        logger.info(f"Copying non-image file: {file_name}")
                        self.copy_file(source_file, target_file)

            # Return the base output directory
            return base_output_dir

        # _copy_non_image_files method has been merged into mirror_directory_structure

        def move_file(self, source_path: Union[str, Path], dest_path: Union[str, Path]) -> bool:
            """
            Move a file from source to destination.

            Ensures the destination directory exists and handles errors.

            Args:
                source_path (str or Path): Source file path.
                dest_path (str or Path): Destination file path.

            Returns:
                bool: True if successful, False otherwise.
            """
            try:
                import shutil
                dest_dir = Path(dest_path).parent
                self.ensure_directory(dest_dir) # Ensure destination directory exists

                shutil.move(str(source_path), str(dest_path))
                logger.debug(f"Moved file from {source_path} to {dest_path}")
                return True
            except Exception as e:
                logger.error(f"Error moving file from {source_path} to {dest_path}: {e}")
                return False
            return all_success

    def cleanup_processed_files(self, processed_files, output_files):
        """
        Clean up processed files after they've been used to create output files.

        Args:
            processed_files (set or list): Set or list of file paths to clean up
            output_files (list): List of output file paths to preserve

        Returns:
            int: Number of files successfully removed
        """
        removed_count = 0

        # Convert to sets for efficient operations
        processed_set = set(processed_files)
        output_set = set(output_files)

        # Only remove files that are in processed_files but not in output_files
        files_to_remove = processed_set - output_set

        for file_path in files_to_remove:
            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove processed file {file_path}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} processed files")

        return removed_count