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
from ezstitcher.core.directory_structure_manager import DirectoryStructureManager
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
        self.dir_structure_manager = None  # Will be initialized when needed

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
                         extensions: Optional[List[str]] = None) -> List[Path]:
        """
        List all image files in a directory with specified extensions.

        Args:
            directory (str or Path): Directory to search
            extensions (list): List of file extensions to include

        Returns:
            list: List of Path objects for image files
        """
        if extensions is None:
            extensions = self.default_extensions

        # Use ImageLocator to find images
        from ezstitcher.core.image_locator import ImageLocator
        return ImageLocator.find_images_in_directory(directory, extensions)

    def path_list_from_pattern(self, directory: Union[str, Path], pattern: str) -> List[str]:
        """
        Get a list of filenames matching a pattern in a directory.

        Args:
            directory (str or Path): Directory to search
            pattern (str): Pattern to match with {iii} placeholder for site index

        Returns:
            list: List of matching filenames
        """
        # Use the PatternMatcher with the current filename parser
        return self.pattern_matcher.path_list_from_pattern(directory, pattern)

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

    def find_files_by_pattern(self, directory: Union[str, Path],
                             pattern: Union[str, Pattern]) -> List[Path]:
        """
        Find files matching a regex pattern.

        Args:
            directory (str or Path): Directory to search
            pattern (str or Pattern): Regex pattern to match

        Returns:
            list: List of matching Path objects
        """
        directory = Path(directory)

        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        matching_files = []
        for file_path in directory.glob('*'):
            if pattern.match(file_path.name):
                matching_files.append(file_path)

        return sorted(matching_files)

    def find_files_by_parser(self, directory: Union[str, Path],
                           parser: Optional[FilenameParser] = None,
                           well: Optional[str] = None,
                           site: Optional[int] = None,
                           channel: Optional[int] = None,
                           z_plane: Optional[int] = None) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        Find files matching criteria using a filename parser.

        Args:
            directory (str or Path): Directory to search
            parser (FilenameParser, optional): Filename parser instance (uses instance parser if None)
            well (str, optional): Well identifier to match
            site (int, optional): Site number to match
            channel (int, optional): Channel number to match
            z_plane (int, optional): Z-plane to match

        Returns:
            list: List of tuples (Path, metadata) for matching files
        """
        directory = Path(directory)
        parser = parser or self.filename_parser

        matching_files = []

        # Check if TimePoint_1 directory exists and use it if it does
        if (directory / "TimePoint_1").exists() and (directory / "TimePoint_1").is_dir():
            search_dir = directory / "TimePoint_1"
        else:
            search_dir = directory

        logger.debug(f"Searching for files in {search_dir} using {parser.__class__.__name__}")

        for file_path in search_dir.glob('*'):
            if not file_path.is_file():
                continue

            try:
                metadata = parser.parse_filename(file_path.name)

                # Skip if metadata is None
                if metadata is None:
                    continue

                # Check if the file matches the criteria
                # Handle both 'z_index' and 'z_plane' keys for Z-plane information
                file_z_plane = metadata.get('z_index', metadata.get('z_plane'))

                if (well is None or metadata.get('well') == well) and \
                   (site is None or metadata.get('site') == site) and \
                   (channel is None or metadata.get('channel') == channel) and \
                   (z_plane is None or file_z_plane == z_plane):
                    matching_files.append((file_path, metadata))
            except (ValueError, KeyError, AttributeError) as e:
                # Not a valid filename for this parser
                logger.debug(f"Could not parse {file_path.name}: {e}")
                continue

        # Sort by well, site, channel, z_plane if available
        def sort_key(item):
            meta = item[1]
            if meta is None:
                return ('', 0, 0, 0)  # Default values for sorting

            # Ensure site, channel, and z_plane are treated as integers for sorting
            return (
                meta.get('well', ''),
                int(meta.get('site', 0)),  # Convert to int to ensure proper numerical sorting
                int(meta.get('channel', 0)),  # Convert to int to ensure proper numerical sorting
                int(meta.get('z_plane', 0))  # Convert to int to ensure proper numerical sorting
            )

        sorted_files = sorted(matching_files, key=sort_key)
        logger.debug(f"Found {len(sorted_files)} matching files")
        return sorted_files

    def parse_positions_csv(self, csv_path: Union[str, Path]) -> List[Tuple[str, float, float]]:
        """
        Parse a CSV file with lines of the form:
          file: <filename>; grid: (col, row); position: (x, y)

        Args:
            csv_path (str or Path): Path to the CSV file

        Returns:
            list: List of tuples (filename, x_float, y_float)
        """
        return self.csv_handler.parse_positions_csv(csv_path)

    def generate_positions_df(self, image_files, positions, grid_positions):
        """
        Generate a DataFrame with position information.

        Args:
            image_files (list): List of image filenames
            positions (list): List of (x, y) position tuples
            grid_positions (list): List of (row, col) grid position tuples

        Returns:
            pandas.DataFrame: DataFrame with position information
        """
        return self.csv_handler.generate_positions_df(image_files, positions, grid_positions)

    def save_positions_df(self, df, positions_path):
        """
        Save a positions DataFrame to CSV.

        Args:
            df (pandas.DataFrame): DataFrame to save
            positions_path (str or Path): Path to save the CSV file

        Returns:
            bool: True if successful, False otherwise
        """
        return self.csv_handler.save_positions_df(df, positions_path)

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

        # Look in parent directory
        parent_dir = plate_path.parent
        htd_files = list(parent_dir.glob("*.HTD"))
        if htd_files:
            for htd_file in htd_files:
                if 'plate' in htd_file.name.lower():
                    return htd_file
            return htd_files[0]

        # Look for Index.xml as a fallback for any microscope type
        index_xml = plate_path / "Index.xml"
        if index_xml.exists():
            return index_xml

        # Check for Index.xml in parent directory
        parent_index_xml = plate_path.parent / "Index.xml"
        if parent_index_xml.exists():
            return parent_index_xml

        return None

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

    def find_wells(self, timepoint_dir: Union[str, Path]) -> List[str]:
        """
        Find all wells in the timepoint directory.

        Deprecated: Use initialize_dir_structure(plate_folder).get_wells() instead.

        Args:
            timepoint_dir (str or Path): Path to the TimePoint_1 directory

        Returns:
            list: List of well names (e.g., ['A01', 'A02', ...])
        """
        warnings.warn(
            "FileSystemManager.find_wells() is deprecated. Use initialize_dir_structure(plate_folder).get_wells() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Initialize directory structure manager for the parent directory
        dir_structure = self.initialize_dir_structure(Path(timepoint_dir).parent)

        # Return wells
        return dir_structure.get_wells()

    def initialize_dir_structure(self, plate_folder: Union[str, Path]) -> DirectoryStructureManager:
        """
        Initialize the directory structure manager for a plate folder.

        Args:
            plate_folder: Path to the plate folder

        Returns:
            DirectoryStructureManager instance
        """
        timepoint_dir_name = getattr(self.config, 'timepoint_dir_name', "TimePoint_1")
        self.dir_structure_manager = DirectoryStructureManager(
            plate_folder,
            self.filename_parser,
            timepoint_dir_name
        )
        return self.dir_structure_manager

    def get_image_path(self, plate_folder: Union[str, Path], well: str, site: int,
                      channel: int, z_index: Optional[int] = None) -> Optional[Path]:
        """
        Get the path to an image based on its metadata.

        Args:
            plate_folder: Path to the plate folder
            well: Well ID (e.g., 'A01' or 'R01C01')
            site: Site number
            channel: Channel number
            z_index: Z-index (optional)

        Returns:
            Path to the image if found, None otherwise
        """
        if self.dir_structure_manager is None or Path(plate_folder) != Path(self.dir_structure_manager.plate_folder):
            self.initialize_dir_structure(plate_folder)
        return self.dir_structure_manager.get_image_path(well, site, channel, z_index)

    def list_images_by_metadata(self, plate_folder: Union[str, Path], well: Optional[str] = None,
                              site: Optional[int] = None, channel: Optional[int] = None,
                              z_index: Optional[int] = None) -> List[Path]:
        """
        List images matching the specified metadata criteria.

        Args:
            plate_folder: Path to the plate folder
            well: Well ID (optional)
            site: Site number (optional)
            channel: Channel number (optional)
            z_index: Z-index (optional)

        Returns:
            List of Path objects for matching images
        """
        if self.dir_structure_manager is None or Path(plate_folder) != Path(self.dir_structure_manager.plate_folder):
            self.initialize_dir_structure(plate_folder)
        return self.dir_structure_manager.list_images(well, site, channel, z_index)

    def get_timepoint_dir(self, plate_folder: Union[str, Path]) -> Optional[Path]:
        """
        Get the path to the TimePoint directory if it exists.

        Args:
            plate_folder: Path to the plate folder

        Returns:
            Path to the TimePoint directory if found, None otherwise
        """
        if self.dir_structure_manager is None or Path(plate_folder) != Path(self.dir_structure_manager.plate_folder):
            self.initialize_dir_structure(plate_folder)
        return self.dir_structure_manager.get_timepoint_dir()

    def get_z_stack_dirs(self, plate_folder: Union[str, Path]) -> List[Tuple[int, Path]]:
        """
        Get the paths to Z-stack directories if they exist.

        Args:
            plate_folder: Path to the plate folder

        Returns:
            List of (z_index, directory) tuples
        """
        if self.dir_structure_manager is None or Path(plate_folder) != Path(self.dir_structure_manager.plate_folder):
            self.initialize_dir_structure(plate_folder)
        return self.dir_structure_manager.get_z_stack_dirs()

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

    def create_output_directories(self, parent_dir, plate_name, suffixes):
        """
        Create output directories for a plate.

        Args:
            parent_dir (str or Path): Parent directory
            plate_name (str): Name of the plate
            suffixes (dict): Dictionary mapping directory types to suffixes

        Returns:
            dict: Dictionary mapping directory types to Path objects
        """
        parent_dir = Path(parent_dir)
        dirs = {}

        # Create directories for each suffix
        for dir_type, suffix in suffixes.items():
            dir_path = parent_dir / f"{plate_name}{suffix}"
            self.ensure_directory(dir_path)
            dirs[dir_type] = dir_path

        return dirs

    def parse_filename(self, filename):
        """
        Parse a microscopy image filename.

        Args:
            filename (str): Filename to parse

        Returns:
            dict: Dictionary with extracted components
        """
        return self.filename_parser.parse_filename(filename)

    def pad_site_number(self, filename, width=3):
        """
        Ensure site number is padded to the specified width.

        Args:
            filename (str): Filename to pad
            width (int, optional): Width to pad site numbers to. Defaults to 3.

        Returns:
            str: Filename with padded site number
        """
        return self.filename_parser.pad_site_number(filename, width=width)

    def rename_files_with_consistent_padding(self, directory, parser=None, width=3, dry_run=False):
        """
        Rename files in a directory to have consistent site number padding.

        Args:
            directory (str or Path): Directory containing files to rename
            parser (FilenameParser, optional): Parser to use for filename parsing and padding
            width (int, optional): Width to pad site numbers to
            dry_run (bool, optional): If True, only print what would be done without actually renaming

        Returns:
            dict: Dictionary mapping original filenames to new filenames
        """
        from ezstitcher.core.filename_parser import FilenameParser, create_parser

        directory = Path(directory)

        # Use default parser if none provided
        if parser is None:
            parser = self.filename_parser

            # If the default parser is not set, try to detect format from files in directory
            if parser is None:
                files = list(directory.glob('*.tif')) + list(directory.glob('*.tiff'))
                if not files:
                    logger.warning(f"No image files found in {directory}")
                    return {}

                # Get filenames only
                filenames = [f.name for f in files]

                # Detect format
                format_type = FilenameParser.detect_format(filenames)
                if format_type is None:
                    logger.warning(f"Could not detect format for files in {directory}")
                    return {}

                # Create parser
                parser = create_parser(format_type)

        # Find all image files
        files = list(directory.glob('*.tif')) + list(directory.glob('*.tiff'))

        # Map original filenames to padded filenames
        rename_map = {}
        for file_path in files:
            original_name = file_path.name
            padded_name = parser.pad_site_number(original_name, width=width)

            # Only include files that need renaming
            if original_name != padded_name:
                rename_map[original_name] = padded_name

        # Check for conflicts (e.g., both s1_w1.tif and s001_w1.tif exist)
        # In this case, we'll skip renaming to avoid overwriting files
        new_names = set(rename_map.values())
        existing_names = set(f.name for f in files)
        conflicts = new_names.intersection(existing_names)

        if conflicts:
            logger.warning(f"Found {len(conflicts)} filename conflicts. These files will not be renamed.")
            for conflict in conflicts:
                # Find all original names that would map to this conflict
                conflicting_originals = [orig for orig, new in rename_map.items() if new == conflict]
                logger.warning(f"Conflict: {conflicting_originals} -> {conflict}")

                # Remove these entries from the rename map
                for orig in conflicting_originals:
                    if orig in rename_map:
                        del rename_map[orig]

        # Perform the renaming
        if not dry_run:
            for original_name, padded_name in rename_map.items():
                original_path = directory / original_name
                padded_path = directory / padded_name

                try:
                    original_path.rename(padded_path)
                    logger.info(f"Renamed {original_name} -> {padded_name}")
                except Exception as e:
                    logger.error(f"Failed to rename {original_name} -> {padded_name}: {e}")
        else:
            for original_name, padded_name in rename_map.items():
                logger.info(f"Would rename {original_name} -> {padded_name}")

        return rename_map

    def construct_filename(self, well, site, wavelength, z_index=None, extension='.tif'):
        """
        Construct a filename from components.

        Args:
            well (str): Well ID (e.g., 'A01')
            site (int): Site number
            wavelength (int): Wavelength number
            z_index (int, optional): Z-index
            extension (str, optional): File extension

        Returns:
            str: Constructed filename
        """
        return self.filename_parser.construct_filename(well, site, wavelength, z_index, extension)

    def auto_detect_patterns(self, folder_path, well_filter=None):
        """
        Automatically detect image patterns in a folder.

        Args:
            folder_path (str or Path): Path to the folder
            well_filter (list): Optional list of wells to include

        Returns:
            dict: Dictionary mapping wells to wavelength patterns
        """
        return self.pattern_matcher.auto_detect_patterns(folder_path, well_filter)

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
