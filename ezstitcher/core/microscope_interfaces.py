"""
Microscope interfaces for ezstitcher.

This module provides abstract base classes for handling microscope-specific
functionality, including filename parsing and metadata handling.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import re

logger = logging.getLogger(__name__)


class FilenameParser(ABC):
    """
    Abstract base class for parsing microscopy image filenames.
    """

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

    def path_list_from_pattern(self, directory, pattern):
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

        # Convert pattern to regex
        # Replace {iii} with (\d+) to match any number of digits (padded or not)
        regex_pattern = pattern.replace('{iii}', '(\\d+)')

        # Handle _z001 suffix if not already in the pattern
        if '_z' not in regex_pattern:
            regex_pattern = regex_pattern.replace('_w1', '_w1(?:_z\\d+)?')

        # Handle .tif vs .tiff extension
        if regex_pattern.endswith('.tif'):
            regex_pattern = regex_pattern[:-4] + '(?:\\.tif|\\.tiff)'

        logger.debug(f"Regex pattern: {regex_pattern}")
        regex = re.compile(regex_pattern)

        # Find all matching files
        matching_files = []
        for file_path in directory.glob('*'):
            if file_path.is_file() and regex.match(file_path.name):
                matching_files.append(file_path.name)

        # Log debug information
        logger.debug(f"Pattern: {pattern}, Directory: {directory}, Files found: {len(matching_files)}")
        if matching_files and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"First file: {matching_files[0]}")

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

    def group_patterns_by_channel(self, patterns):
        """
        Group patterns by channel/wavelength using the filename parser.

        Args:
            patterns (list): List of patterns to group

        Returns:
            dict: Dictionary mapping channel numbers to patterns
        """
        patterns_by_channel = {}

        for pattern in patterns:
            # Use the filename parser to extract the channel
            # Replace {iii} with a dummy site number for parsing
            pattern_with_site = pattern.replace('{iii}', '001')
            # Also replace {zzz} with a dummy z-index if present
            if '{zzz}' in pattern_with_site:
                pattern_with_site = pattern_with_site.replace('{zzz}', '001')

            metadata = self.parse_filename(pattern_with_site)

            if metadata and 'channel' in metadata:
                channel = str(metadata['channel'])
                if channel not in patterns_by_channel:
                    patterns_by_channel[channel] = []
                patterns_by_channel[channel].append(pattern)
            else:
                logger.warning(f"Could not extract channel from pattern: {pattern}")
                continue

        return patterns_by_channel

    def group_patterns_by_z_index(self, patterns):
        """
        Group patterns by z-index using the filename parser.

        Args:
            patterns (list): List of patterns to group

        Returns:
            dict: Dictionary mapping z-indices to patterns
        """
        patterns_by_z = {}

        for pattern in patterns:
            # Check if this is a z-variable pattern (contains {zzz})
            if '{zzz}' in pattern:
                # This is a pattern with variable z-index
                # Add it to a special 'variable' category
                if 'variable' not in patterns_by_z:
                    patterns_by_z['variable'] = []
                patterns_by_z['variable'].append(pattern)
                continue

            # For fixed z-index patterns, extract the z-index
            # Replace {iii} with a dummy site number for parsing
            pattern_with_site = pattern.replace('{iii}', '001')
            metadata = self.parse_filename(pattern_with_site)

            if metadata and 'z_index' in metadata and metadata['z_index'] is not None:
                z_index = str(metadata['z_index'])
                if z_index not in patterns_by_z:
                    patterns_by_z[z_index] = []
                patterns_by_z[z_index].append(pattern)
            else:
                # If no z-index is found, put it in the '1' category (default z-index)
                if '1' not in patterns_by_z:
                    patterns_by_z['1'] = []
                patterns_by_z['1'].append(pattern)

        return patterns_by_z

    def auto_detect_patterns(self, folder_path, well_filter=None, extensions=None,
                           group_by='channel', variable_site=True, variable_z=False):
        """
        Automatically detect image patterns in a folder.

        Args:
            folder_path (str or Path): Path to the folder
            well_filter (list): Optional list of wells to include
            extensions (list): Optional list of file extensions to include
            group_by (str): How to group patterns ('channel' or 'z_index')
            variable_site (bool): Whether to generate patterns with variable site (default: True)
            variable_z (bool): Whether to generate patterns with variable z-index (default: False)

        Returns:
            dict: Dictionary mapping wells to patterns grouped by channel or z-index
        """
        from collections import defaultdict
        from ezstitcher.core.image_locator import ImageLocator

        folder_path = Path(folder_path)
        extensions = extensions or ['.tif', '.TIF', '.tiff', '.TIFF']

        # Find all image files
        image_dir = ImageLocator.find_image_directory(folder_path)
        logger.info(f"Using image directory: {image_dir}")
        image_paths = ImageLocator.find_images_in_directory(image_dir, extensions, recursive=True)

        if not image_paths:
            logger.warning(f"No image files found in {folder_path}")
            return {}

        # Process all images
        patterns_by_well = defaultdict(list)
        for img_path in image_paths:
            metadata = self.parse_filename(img_path.name)
            if not metadata or (well_filter and metadata['well'] not in well_filter):
                continue

            well = metadata['well']
            channel = metadata['channel']
            has_z = 'z_index' in metadata and metadata['z_index'] is not None

            # Generate patterns based on flags
            if "_s" in img_path.name and "_w" in img_path.name:  # ImageXpress format
                if variable_site and not variable_z:
                    # Only variable site
                    pattern = self.construct_filename(
                        well=well, site="{iii}", channel=channel,
                        z_index=metadata.get('z_index'),
                        extension=metadata['extension']
                    )
                    patterns_by_well[well].append(pattern)
                elif variable_z and not variable_site:
                    # Only variable z-index
                    if has_z:
                        pattern = self.construct_filename(
                            well=well, site=metadata['site'], channel=channel,
                            z_index="{iii}", extension=metadata['extension']
                        )
                        pattern = re.sub(r'_z\d+', '_z{iii}', pattern)
                        patterns_by_well[well].append(pattern)
                elif variable_site and variable_z:
                    # Both variable
                    if has_z:
                        pattern = self.construct_filename(
                            well=well, site="{iii}", channel=channel,
                            z_index="{iii}", extension=metadata['extension']
                        )
                        pattern = re.sub(r'_z\d+', '_z{iii}', pattern)
                        patterns_by_well[well].append(pattern)
                    else:
                        pattern = self.construct_filename(
                            well=well, site="{iii}", channel=channel,
                            extension=metadata['extension']
                        )
                        patterns_by_well[well].append(pattern)
            else:  # Opera Phenix format
                if variable_site and not variable_z:
                    # Only variable site
                    pattern = re.sub(r'f\d+', 'f{iii}', img_path.name)
                    patterns_by_well[well].append(pattern)
                elif variable_z and not variable_site:
                    # Only variable z-index
                    if has_z:
                        pattern = re.sub(r'p\d+', 'p{iii}', img_path.name)
                        patterns_by_well[well].append(pattern)
                elif variable_site and variable_z:
                    # Both variable
                    pattern = re.sub(r'f\d+', 'f{iii}', img_path.name)
                    if has_z:
                        pattern = re.sub(r'p\d+', 'p{iii}', pattern)
                    patterns_by_well[well].append(pattern)

        # Group patterns and remove duplicates
        result = {}
        for well, patterns in patterns_by_well.items():
            # Remove duplicates while preserving order
            unique_patterns = []
            for pattern in patterns:
                if pattern not in unique_patterns:
                    unique_patterns.append(pattern)

            # Group by channel or z-index
            if group_by == 'z_index':
                result[well] = self.group_patterns_by_z_index(unique_patterns)
            else:  # Default to channel grouping
                result[well] = self.group_patterns_by_channel(unique_patterns)

        return result


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
    def get_pixel_size(self, plate_path: Union[str, Path]) -> Optional[float]:
        """
        Get the pixel size from metadata.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Pixel size in micrometers, or None if not available
        """
        pass


class MicroscopeHandler:
    """Composed class for handling microscope-specific functionality."""

    DEFAULT_MICROSCOPE = 'ImageXpress'
    _handlers_cache = None

    @classmethod
    def _discover_handlers(cls):
        """Discover all microscope handlers from the microscopes subpackage."""
        if cls._handlers_cache:
            return cls._handlers_cache

        import importlib, inspect, pkgutil
        from ezstitcher.microscopes import __path__ as microscopes_path

        handlers = {}

        # Find all modules in the microscopes package
        for _, module_name, _ in pkgutil.iter_modules(microscopes_path, 'ezstitcher.microscopes.'):
            try:
                module = importlib.import_module(module_name)

                # Find FilenameParser implementations in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ != module.__name__ or not issubclass(obj, FilenameParser) or obj == FilenameParser:
                        continue

                    # Extract microscope type from class name
                    microscope_type = name.replace('FilenameParser', '')

                    # Look for matching MetadataHandler
                    handler_name = f"{microscope_type}MetadataHandler"
                    handler_class = getattr(module, handler_name, None)

                    if handler_class and issubclass(handler_class, MetadataHandler):
                        handlers[microscope_type] = (obj, handler_class)
            except Exception as e:
                logger.debug(f"Error inspecting module {module_name}: {e}")

        cls._handlers_cache = handlers
        return handlers

    def __init__(self, plate_folder=None, parser=None, metadata_handler=None, microscope_type='auto'):
        """Initialize with plate folder and optional components."""
        self.plate_folder = Path(plate_folder) if plate_folder else None

        if parser is None or metadata_handler is None:
            detected_type = self._detect_microscope_type(microscope_type)
            self.parser, self.metadata_handler = self._create_handlers(detected_type, parser, metadata_handler)
        else:
            self.parser, self.metadata_handler = parser, metadata_handler

    def _detect_microscope_type(self, microscope_type):
        """Detect microscope type from files or use specified type."""
        if microscope_type.lower() != 'auto' or not self.plate_folder:
            return microscope_type if microscope_type.lower() != 'auto' else self.DEFAULT_MICROSCOPE

        try:
            # Get sample files and test each parser
            from ezstitcher.core.image_locator import ImageLocator
            sample_files = ImageLocator.find_images_in_directory(self.plate_folder)[:10]

            if not sample_files:
                return self.DEFAULT_MICROSCOPE

            matches =  {}
            for name, (parser_class, _) in self._discover_handlers().items():
                matches[name] = 0
                for f in sample_files:
                    if parser_class.can_parse(f.name):
                        matches[name] += 1


            best_match = max(matches.items(), key=lambda x: x[1]) if matches else (self.DEFAULT_MICROSCOPE, 0)
            if best_match[1] > 0:
                logger.info(f"Auto-detected {best_match[0]} format ({best_match[1]}/{len(sample_files)} files matched)")
                return best_match[0]

            return self.DEFAULT_MICROSCOPE
        except Exception as e:
            logger.error(f"Error during auto-detection: {e}")
            return self.DEFAULT_MICROSCOPE

    def _create_handlers(self, microscope_type, parser=None, metadata_handler=None):
        """Create parser and metadata handler for the specified microscope type."""
        handlers = self._discover_handlers()
        parser_class, handler_class = handlers.get(microscope_type, handlers.get(self.DEFAULT_MICROSCOPE, (None, None)))

        return (parser or (parser_class() if parser_class else None),
                metadata_handler or (handler_class() if handler_class else None))

    # Delegate filename parsing methods to parser

    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Delegate to parser."""
        return self.parser.parse_filename(filename)

    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None,
                          channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None,
                          extension: str = '.tif',
                          site_padding: int = 3, z_padding: int = 3) -> str:
        """Delegate to parser."""
        return self.parser.construct_filename(
            well, site, channel, z_index, extension, site_padding, z_padding
        )

    def auto_detect_patterns(self, folder_path, **kwargs):
        """Delegate to parser."""
        return self.parser.auto_detect_patterns(folder_path, **kwargs)

    def path_list_from_pattern(self, directory, pattern):
        """Delegate to parser."""
        return self.parser.path_list_from_pattern(directory, pattern)

    # Delegate metadata handling methods to metadata_handler

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """Delegate to metadata handler."""
        return self.metadata_handler.find_metadata_file(plate_path)

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """Delegate to metadata handler."""
        return self.metadata_handler.get_grid_dimensions(plate_path)

    def get_pixel_size(self, plate_path: Union[str, Path]) -> Optional[float]:
        """Delegate to metadata handler."""
        return self.metadata_handler.get_pixel_size(plate_path)


def create_microscope_handler(microscope_type: str = 'auto', **kwargs) -> MicroscopeHandler:
    """Create the appropriate microscope handler."""
    return MicroscopeHandler(microscope_type=microscope_type, **kwargs)
