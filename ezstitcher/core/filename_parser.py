"""
Filename parser for ezstitcher.

This module provides abstract and concrete classes for parsing microscopy image filenames
from different microscope platforms.
"""

import re
import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple, Union

logger = logging.getLogger(__name__)


class FilenameParser(ABC):
    """Abstract base class for parsing microscopy image filenames and matching patterns.

    This class handles both filename parsing and pattern matching functionality.
    """

    @staticmethod
    def replace_placeholders(pattern: str, replacements: Union[str, Dict[str, str]] = '001') -> str:
        """
        Replace all placeholders in curly braces with specified values.

        Args:
            pattern (str): The string containing placeholders like {iii}, {zzz}, etc.
            replacements (str or dict): Either a single replacement string for all placeholders,
                                        or a dictionary mapping placeholder patterns to replacement values.

        Returns:
            str: The pattern with all placeholders replaced.

        Examples:
            >>> FilenameParser.replace_placeholders("A01_s{iii}_w2.tif")
            'A01_s001_w2.tif'
            >>> FilenameParser.replace_placeholders("A01_s{iii}_w{www}.tif", {'iii': '001', 'www': '002'})
            'A01_s001_w002.tif'
        """
        if isinstance(replacements, str):
            # Simple case: replace all placeholders with the same value
            return re.sub(r'\{[^}]*\}', replacements, pattern)
        else:
            # Advanced case: use different replacements for different patterns
            result = pattern
            for placeholder, replacement in replacements.items():
                # Escape curly braces for regex
                placeholder_pattern = f'\\{{{placeholder}\\}}'
                result = re.sub(placeholder_pattern, replacement, result)
            # Replace any remaining unmatched placeholders with a default value
            result = re.sub(r'\{[^}]*\}', '001', result)
            return result


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
            dict or None: Dictionary with extracted components (including 'extension') or None if parsing fails
        """
        pass

    @abstractmethod
    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None, channel: Optional[int] = None,
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

    def metadata_from_pattern(self, pattern):
        """
        Extract metadata from a filename pattern.

        Args:
            pattern (str): Filename pattern

        Returns:
            dict: Dictionary with extracted metadata
        """
        return self.parse_filename(pattern.replace('{iii}', '001'))

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

    def auto_detect_patterns(self, folder_path, well_filter=None, extensions=None, group_by='channel', variable_site=True, variable_z=False):
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
            has_z = 'z_index' in metadata

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
            result[well] = unique_patterns

            # Group by channel or z-index
            if group_by == 'z_index':
                result[well] = self.group_patterns_by_z_index(unique_patterns)
            else:  # Default to channel grouping
                result[well] = self.group_patterns_by_channel(unique_patterns)

        return result

    @classmethod
    def detect_format(cls, filenames: List[str]) -> Optional[str]:
        """
        Detect the microscope format from a list of filenames.

        Args:
            filenames (list): List of filenames to analyze

        Returns:
            str or None: Detected format ('ImageXpress', 'OperaPhenix', etc.) or None if unknown
        """
        if not filenames:
            return None

        # Import here to avoid circular imports
        from ezstitcher.core.filename_parser import ImageXpressFilenameParser, OperaPhenixFilenameParser

        # Check at least a few filenames
        sample_size = min(len(filenames), 10)
        samples = [os.path.basename(f) for f in filenames[:sample_size]]

        # Count matches for each format using the can_parse method
        imagexpress_matches = sum(1 for f in samples if ImageXpressFilenameParser.can_parse(f))
        opera_matches = sum(1 for f in samples if OperaPhenixFilenameParser.can_parse(f))

        # Determine the most likely format
        if imagexpress_matches > opera_matches and imagexpress_matches > 0:
            return 'ImageXpress'
        elif opera_matches > imagexpress_matches and opera_matches > 0:
            return 'OperaPhenix'
        elif imagexpress_matches > 0:
            return 'ImageXpress'  # Default to ImageXpress if tied
        elif opera_matches > 0:
            return 'OperaPhenix'  # Only Opera matches found

        return None


class ImageXpressFilenameParser(FilenameParser):
    """Parser for ImageXpress microscope filenames.

    Handles standard ImageXpress format filenames like:
    - A01_s001_w1.tif
    - A01_s1_w1_z1.tif

    Does NOT handle Opera Phenix format filenames, even if they use ImageXpress-style naming.
    For those, use the OperaPhenixFilenameParser.
    """

    _pattern = re.compile(r'(?:.*?_)?([A-Z]\d+)(?:_s(\d+))?(?:_w(\d+))?(?:_z(\d+))?(\.\w+)?$')

    @classmethod
    def can_parse(cls, filename: str) -> bool:
        """
        Check if this parser can parse the given filename.
        Args:
            filename (str): Filename to check
        Returns:
            bool: True if this parser can parse the filename, False otherwise
        """
        # Extract just the basename
        basename = os.path.basename(filename)
        # Check if the filename matches the ImageXpress pattern
        # but is not an Opera Phenix well in ImageXpress format
        return bool(cls._pattern.match(basename))

    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse an ImageXpress filename to extract all components, including extension.
        Overrides the base class implementation for efficiency.
        """
        basename = os.path.basename(filename)
        match = self._pattern.match(basename)

        if match:
            well, site_str, channel_str, z_str, ext = match.groups()

            # Handle optional components - return None if missing

            result = {'well' : well,
                      'site' : int(site_str) if site_str else None,
                      'channel' : int(channel_str) if channel_str else None,
                      'z_index' : int(z_str) if z_str else None,
                      'extension' : ext if ext else '.tif'} # Default if somehow empty

            return result
        else:
            logger.debug(f"Could not parse ImageXpress filename: {filename}")
            return None


    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None, channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None, extension: str = '.tif',
                          site_padding: int = 3, z_padding: int = 3) -> str:
        """Construct an ImageXpress filename from components, only including parts if provided.

        Args:
            well (str): Well ID (e.g., 'A01')
            site (int or str, optional): Site number or placeholder string (e.g., '{iii}')
            channel (int, optional): Channel number
            z_index (int or str, optional): Z-index or placeholder string (e.g., '{zzz}')
            extension (str, optional): File extension
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 3)

        Returns:
            str: Constructed filename
        """
        if not well:
            raise ValueError("Well ID cannot be empty or None.")

        parts = [well]
        if site is not None:
            if isinstance(site, str):
                # If site is a string (e.g., '{iii}'), use it directly
                parts.append(f"_s{site}")
            else:
                # Otherwise, format it as a padded integer
                parts.append(f"_s{site:0{site_padding}d}")

        if channel is not None:
            parts.append(f"_w{channel}")

        if z_index is not None:
            if isinstance(z_index, str):
                # If z_index is a string (e.g., '{zzz}'), use it directly
                parts.append(f"_z{z_index}")
            else:
                # Otherwise, format it as a padded integer
                parts.append(f"_z{z_index:0{z_padding}d}")

        base_name = "".join(parts)
        return f"{base_name}{extension}"


class OperaPhenixFilenameParser(FilenameParser):
    """Parser for Opera Phenix microscope filenames.

    Handles Opera Phenix format filenames like:
    - r01c01f001p01-ch1sk1fk1fl1.tiff
    - r01c01f001p01-ch1.tiff
    """

    # Native Opera Phenix format pattern
    # Groups: 1=row, 2=col, 3=field(site), 4=plane(z), 5=channel, 6=extension
    _pattern = re.compile(r"r(\d{1,2})c(\d{1,2})f(\d+)p(\d+)-ch(\d+)(?:sk\d+)?(?:fk\d+)?(?:fl\d+)?(\.\w+)$", re.I)

    # Pattern for extracting row and column from Opera Phenix well format
    _well_pattern = re.compile(r"R(\d{2})C(\d{2})", re.I)

    @classmethod
    def can_parse(cls, filename: str) -> bool:
        """
        Check if this parser can parse the given filename.

        Args:
            filename (str): Filename to check

        Returns:
            bool: True if this parser can parse the filename, False otherwise
        """
        # Extract just the basename
        basename = os.path.basename(filename)

        # Check if the filename matches the Opera Phenix pattern
        return bool(cls._pattern.match(basename))



    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse an Opera Phenix filename to extract all components.
        Overrides the base class implementation for efficiency.

        Args:
            filename (str): Filename to parse

        Returns:
            dict or None: Dictionary with extracted components or None if parsing fails.
        """
        basename = os.path.basename(filename)

        # Try parsing using the Opera Phenix pattern
        match = self._pattern.match(basename)
        if match:
            row, col, site, z_index, channel, ext = match.groups()
            well = f"R{int(row):02d}C{int(col):02d}"
            extension = ext if ext else '.tif'
            result = {
                'well': well,
                'site': int(site),
                'channel': int(channel),
                'wavelength': int(channel),  # For backward compatibility
                'z_index': int(z_index),
                'extension': extension
            }
            return result

        return None

    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None, channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None, extension: str = '.tiff',
                          site_padding: int = 3, z_padding: int = 2) -> str:
        """
        Construct an Opera Phenix filename from components.

        Args:
            well (str): Well ID (e.g., 'R03C04' or 'A01')
            site: Site/field number (int) or placeholder string
            channel (int): Channel number
            z_index: Z-index/plane (int) or placeholder string
            extension (str, optional): File extension
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 2)

        Returns:
            str: Constructed filename
        """
        # Extract row and column from well name
        # Check if well is in Opera Phenix format (e.g., 'R01C03')
        match = self._well_pattern.match(well)
        if match:
            # Extract row and column from Opera Phenix format
            row = int(match.group(1))
            col = int(match.group(2))
        elif re.match(r"[A-Z]\d{2}", well):
            # Convert ImageXpress format (e.g., 'A01') to row and column
            row_letter = well[0]
            col = int(well[1:3])
            # Convert row letter to number (A -> 1, B -> 2, etc.)
            row = ord(row_letter) - 64  # ASCII: 'A' = 65
        else:
            raise ValueError(f"Invalid well format: {well}. Expected format: 'R01C03' or 'A01'")

        # Default Z-index to 1 if not provided
        z_index = 1 if z_index is None else z_index

        # Construct filename in Opera Phenix format
        if isinstance(site, str):
            # If site is a string (e.g., '{iii}'), use it directly
            site_part = f"f{site}"
        else:
            # Otherwise, format it as a padded integer
            site_part = f"f{site:0{site_padding}d}"

        if isinstance(z_index, str):
            # If z_index is a string (e.g., '{zzz}'), use it directly
            z_part = f"p{z_index}"
        else:
            # Otherwise, format it as a padded integer
            z_part = f"p{z_index:0{z_padding}d}"

        return f"r{row:02d}c{col:02d}{site_part}{z_part}-ch{channel}sk1fk1fl1{extension}"



def create_parser(microscope_type: str, sample_files: Optional[List[str]] = None, plate_folder: Optional[Union[str, Path]] = None) -> FilenameParser:
    """
    Factory function to create the appropriate parser for a microscope type.

    Args:
        microscope_type (str): Type of microscope ('ImageXpress', 'OperaPhenix', 'auto', etc.)
        sample_files (list, optional): List of sample filenames for auto-detection
        plate_folder (str or Path, optional): Path to plate folder for auto-detection

    Returns:
        FilenameParser: Instance of the appropriate parser

    Raises:
        ValueError: If microscope_type is not supported or auto-detection fails
    """
    microscope_type = microscope_type.lower()

    if microscope_type == 'imagexpress':
        return ImageXpressFilenameParser()
    elif microscope_type == 'operaphenix':
        return OperaPhenixFilenameParser()
    elif microscope_type == 'auto':
        # Perform actual auto-detection
        if sample_files:
            # Detect based on sample filenames
            detected_type = FilenameParser.detect_format(sample_files)
            if detected_type:
                logger.info(f"Auto-detected microscope type from filenames: {detected_type}")
                return create_parser(detected_type)  # Recursive call with detected type

        if plate_folder:
            # Detect based on folder structure
            from ezstitcher.core.image_locator import ImageLocator

            # Find all image locations
            image_locations = ImageLocator.find_image_locations(plate_folder)

            # Collect sample files from all locations
            all_samples = []
            # With the simplified implementation, all images are in the 'all' key
            all_samples.extend([Path(f).name for f in image_locations['all'][:10]])

            if all_samples:
                detected_type = FilenameParser.detect_format(all_samples)
                if detected_type:
                    logger.info(f"Auto-detected microscope type from folder structure: {detected_type}")
                    return create_parser(detected_type)  # Recursive call with detected type

        # If we couldn't detect, default to ImageXpress but log a warning
        logger.warning("Could not auto-detect microscope type, defaulting to ImageXpress")
        return ImageXpressFilenameParser()
    else:
        raise ValueError(f"Unsupported microscope type: {microscope_type}")


def detect_parser(filenames: List[str]) -> FilenameParser:
    """
    Automatically detect the appropriate parser for a list of filenames.

    Args:
        filenames (list): List of filenames to analyze

    Returns:
        FilenameParser: Instance of the detected parser

    Raises:
        ValueError: If the format cannot be detected
    """
    format_type = FilenameParser.detect_format(filenames)
    if format_type:
        return create_parser(format_type)
    else:
        # Default to ImageXpress if detection fails
        logger.warning("Could not detect microscope format, defaulting to ImageXpress")
        return ImageXpressFilenameParser()
