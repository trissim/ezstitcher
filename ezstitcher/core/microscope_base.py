from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path
import re
import logging
from collections import defaultdict
from ezstitcher.io.filemanager import FileManager 

logger = logging.getLogger(__name__)

class FilenameParser(ABC):
    """
    Abstract base class for parsing microscopy image filenames.
    """

    # Constants
    FILENAME_COMPONENTS = ['well', 'site', 'channel', 'z_index', 'extension']
    PLACEHOLDER_PATTERN = '{iii}'

    @classmethod
    @abstractmethod
    def can_parse(cls, filename: str) -> bool:
        """
        Check if this parser can parse the given filename.

        Args:
            filename (str): Filename to check

        Returns:
            bool: True if this parser can parse the filename, False otherwise
        """
        pass

    @abstractmethod
    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse a microscopy image filename to extract all components.

        Args:
            filename (str): Filename to parse

        Returns:
            dict or None: Dictionary with extracted components or None if parsing fails
        """
        pass

    @abstractmethod
    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None,
                          channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None,
                          extension: str = '.tif',
                          site_padding: int = 3, z_padding: int = 3) -> str:
        """
        Construct a filename from components.

        Args:
            well (str): Well ID (e.g., 'A01')
            site (int or str, optional): Site number or placeholder string (e.g., '{iii}')
            channel (int, optional): Channel/wavelength number
            z_index (int or str, optional): Z-index or placeholder string (e.g., '{zzz}')
            extension (str, optional): File extension
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 3)

        Returns:
            str: Constructed filename
        """
        pass

    def path_list_from_pattern(self, directory, pattern, fm = FileManager(backend='disk')):
        """
        Get a list of filenames matching a pattern in a directory.

        Args:
            directory (str or Path): Directory to search
            pattern (str): Pattern to match with {iii} placeholder for site index

        Returns:
            list: List of matching filenames
        """
        directory = Path(directory)

        # Handle substitution of {series} if present (from Ashlar)
        if "{series}" in pattern:
            pattern = pattern.replace("{series}", "{iii}")

        # Parse the pattern to extract expected components
        pattern_metadata = self.parse_filename(pattern)
        if not pattern_metadata:
            logger.warning(f"Could not parse pattern: {pattern}")
            return []

        # Find all files in the directory
        matching_files = []
        all_images = fm.list_image_files(directory)
        for file_path in all_images:
            if not file_path.is_file():
                continue

            # Parse the filename to extract its components
            file_metadata = self.parse_filename(file_path.name)
            if not file_metadata:
                continue

            # Check if all non-None components in the pattern match the file
            is_match = True
            for key, value in pattern_metadata.items():
                # Skip components that are None in the pattern (placeholders)
                if value is None:
                    continue

                # Check if the component exists in the file metadata and matches
                if key not in file_metadata or file_metadata[key] != value:
                    is_match = False
                    break

            if is_match:
                matching_files.append(file_path.name)

        # Log debug information
        logger.debug("Pattern: %s, Directory: %s, Files found: %d", pattern, directory, len(matching_files))
        if matching_files and logger.isEnabledFor(logging.DEBUG):
            logger.debug("First file: %s", matching_files[0])

        # Use natural sorting instead of lexicographical sorting
        return self._natural_sort(matching_files)

    def _natural_sort(self, file_list):
        """
        Sort filenames naturally, so that site numbers are sorted numerically.
        E.g., ["s1", "s10", "s2"] -> ["s1", "s2", "s10"]

        Args:
            file_list (list): List of filenames to sort

        Returns:
            list: Naturally sorted list of filenames
        """
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        return sorted(file_list, key=natural_sort_key)

    def group_patterns_by_component(self, patterns, component='channel', default_value='1'):
        """
        Group patterns by a specific component (channel, z_index, site, well, etc.)

        Args:
            patterns (list): List of patterns to group
            component (str): Component to group by (e.g., 'channel', 'z_index', 'site', 'well')
            default_value (str): Default value to use if component is not found

        Returns:
            dict: Dictionary mapping component values to patterns
        """
        grouped_patterns = {}

        for pattern in patterns:
            # Replace placeholder with dummy value for parsing
            pattern_with_dummy = pattern.replace(self.PLACEHOLDER_PATTERN, '001')
            metadata = self.parse_filename(pattern_with_dummy)

            if metadata and component in metadata and metadata[component] is not None:
                # Extract component value and convert to string
                value = str(metadata[component])
                if value not in grouped_patterns:
                    grouped_patterns[value] = []
                grouped_patterns[value].append(pattern)
            else:
                # Use default value if component not found
                if default_value not in grouped_patterns:
                    grouped_patterns[default_value] = []
                grouped_patterns[default_value].append(pattern)
                if component != 'z_index':  # z_index commonly defaults to 1, so don't log warning
                    logger.warning("Could not extract %s from pattern: %s", component, pattern)

        return grouped_patterns

    def auto_detect_patterns(self, folder_path, well_filter=None, extensions=None,
                           group_by=None, variable_components=None, flat=False):
        """
        Automatically detect image patterns in a folder.

        Args:
            folder_path (str or Path): Path to the folder
            well_filter (list): Optional list of wells to include
            extensions (list): Optional list of file extensions to include
            group_by (str, optional): Component to group patterns by (e.g., 'channel', 'z_index', 'well')
                                      If None, returns a flat list of patterns per well
            variable_components (list): List of components to make variable (e.g., ['site', 'z_index'])
            flat (bool): Deprecated. Use group_by=None instead.

        Returns:
            dict: Dictionary mapping wells to patterns (either grouped by component or flat list)
        """
        # Set default variable components if not provided
        if variable_components is None:
            variable_components = ['site']

        # Find all image files and group by well
        files_by_well = self._find_and_filter_images(folder_path, well_filter, extensions)

        if not files_by_well:
            return {}

        # Generate patterns for each well
        result = {}
        for well, files in files_by_well.items():
            # Generate patterns for this well
            patterns = self._generate_patterns_for_files(files, variable_components)

            # Return patterns based on requested format
            if flat or group_by is None:
                result[well] = patterns
            else:
                result[well] = self.group_patterns_by_component(patterns, component=group_by)

        return result

    def _find_and_filter_images(self, folder_path, well_filter=None, extensions=None, fm = FileManager(backend='disk')):
        """
        Find all image files in a directory and filter by well.

        Args:
            folder_path (str or Path): Path to the folder
            well_filter (list): Optional list of wells to include
            extensions (list): Optional list of file extensions to include

        Returns:
            dict: Dictionary mapping wells to lists of image files
        """
        import time  # Import here for timing

        start_time = time.time()
        logger.info("Finding and filtering images in %s", folder_path)

        # Find all image files
        folder_path = Path(folder_path)
        extensions = extensions or ['.tif', '.TIF', '.tiff', '.TIFF']
        image_dir = folder_path 
        logger.info("Using image directory: %s", image_dir)

        image_paths = fm.list_image_files(image_dir, extensions, recursive=True)

        if not image_paths:
            logger.warning("No image files found in %s", folder_path)
            return {}

        logger.info("Found %d image files in %.2f seconds. Grouping by well...",
                   len(image_paths), time.time() - start_time)
        group_start = time.time()

        # Group files by well
        files_by_well = defaultdict(list)
        for img_path in image_paths:
            metadata = self.parse_filename(img_path.name)
            if not metadata:
                continue

            well = metadata['well']
            # Case-insensitive well filtering
            if not well_filter or any(well.lower() == w.lower() for w in well_filter):
                files_by_well[well].append(img_path)

        logger.info("Grouped %d files into %d wells in %.2f seconds",
                   len(image_paths), len(files_by_well), time.time() - group_start)
        return files_by_well

    def _generate_patterns_for_files(self, files, variable_components):
        """
        Generate patterns for a list of files with specified variable components.

        Args:
            files (list): List of file paths
            variable_components (list): List of components to make variable

        Returns:
            list: List of patterns
        """
        # Use the imported defaultdict from the top of the file

        # Get unique combinations of non-variable components
        component_combinations = defaultdict(list)

        for file_path in files:
            metadata = self.parse_filename(file_path.name)
            if not metadata:
                continue

            # Create a key based on non-variable components
            key_parts = []
            for comp in self.FILENAME_COMPONENTS:
                if comp in metadata and comp not in variable_components and metadata[comp] is not None:
                    key_parts.append(f"{comp}={metadata[comp]}")

            key = ",".join(key_parts)
            component_combinations[key].append((file_path, metadata))

        # Generate patterns for each combination
        patterns = []
        for _, files_metadata in component_combinations.items():
            if not files_metadata:
                continue

            # Use the first file's metadata as a template
            _, template_metadata = files_metadata[0]

            # Create pattern by replacing variable components with placeholders
            pattern_args = {}
            for comp in self.FILENAME_COMPONENTS:
                if comp in template_metadata:  # Only include components that exist in the metadata
                    if comp in variable_components:
                        pattern_args[comp] = self.PLACEHOLDER_PATTERN
                    else:
                        pattern_args[comp] = template_metadata[comp]

            # Construct the pattern
            pattern = self.construct_filename(
                well=pattern_args['well'],
                site=pattern_args.get('site'),
                channel=pattern_args.get('channel'),
                z_index=pattern_args.get('z_index'),
                extension=pattern_args.get('extension', '.tif')
            )

            patterns.append(pattern)

        return patterns


class MetadataHandler(ABC):
    """
    Abstract base class for handling microscope metadata.
    """

    @abstractmethod
    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """
        Find the metadata file for a plate.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Path to the metadata file, or None if not found
        """
        pass

    @abstractmethod
    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """
        Get grid dimensions for stitching from metadata.

        Args:
            plate_path: Path to the plate folder

        Returns:
            (grid_size_x, grid_size_y)
        """
        pass

    @abstractmethod
    def get_pixel_size(self, plate_path: Union[str, Path]) -> float:
        """
        Get the pixel size from metadata.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Pixel size in micrometers

        Raises:
            ValueError: If pixel size cannot be determined from metadata
        """
        pass