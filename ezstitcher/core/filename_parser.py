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
    """Abstract base class for parsing microscopy image filenames."""

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

    @abstractmethod
    def parse_well(self, filename: str) -> Optional[str]:
        """
        Parse well ID from a filename.

        Args:
            filename (str): Filename to parse

        Returns:
            str or None: Well ID (e.g., 'A01') or None if parsing fails
        """
        pass

    @abstractmethod
    def parse_site(self, filename: str) -> Optional[int]:
        """
        Parse site number from a filename.

        Args:
            filename (str): Filename to parse

        Returns:
            int or None: Site number or None if parsing fails
        """
        pass

    @abstractmethod
    def parse_z_index(self, filename: str) -> Optional[int]:
        """
        Parse Z-index from a filename.

        Args:
            filename (str): Filename to parse

        Returns:
            int or None: Z-index or None if parsing fails
        """
        pass

    @abstractmethod
    def parse_channel(self, filename: str) -> Optional[int]:
        """
        Parse channel/wavelength from a filename.

        Args:
            filename (str): Filename to parse

        Returns:
            int or None: Channel/wavelength number or None if parsing fails
        """
        pass

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

    def pad_site_number(self, filename: str, width: int = 3) -> str:
        """
        Ensure site number is padded to a consistent width.

        Args:
            filename (str): Filename to pad
            width (int): Width to pad to

        Returns:
            str: Filename with padded site number
        """
        # Extract just the filename without the path
        basename = os.path.basename(filename)

        # Get site number
        site = self.parse_site(basename)
        if site is None:
            return filename  # Return original if site can't be parsed

        # Pad site number and replace in filename
        site_str = str(site)
        padded_site = site_str.zfill(width)

        # Use a generic approach that works for all formats
        # For ImageXpress format: A01_s1_w1.tif -> A01_s001_w1.tif
        result = re.sub(r'_s' + site_str + r'_', f'_s{padded_site}_', basename)

        # If the filename didn't change, it might be in a different format
        # Try Opera Phenix format: r01c01f1p01-ch1.tiff -> r01c01f001p01-ch1.tiff
        if result == basename:
            result = re.sub(r'f' + site_str + r'p', f'f{padded_site}p', basename)

        # If the path was provided, maintain it
        if filename != basename:
            return os.path.join(os.path.dirname(filename), result)

        return result

    def add_z_suffix(self, filename: str, z_index: int) -> str:
        """
        Add or update Z-plane suffix in a filename.

        Args:
            filename (str): Filename to update
            z_index (int): Z-index to add

        Returns:
            str: Filename with Z-plane suffix
        """
        # Extract just the filename without the path
        basename = os.path.basename(filename)

        # Check if filename already has a z-suffix
        existing_z = self.parse_z_index(basename)

        if existing_z is not None:
            # Replace existing z-suffix
            return re.sub(r'_z\d+(\.\w+)$', f'_z{z_index:03d}\1', basename)
        else:
            # Add new z-suffix before the extension
            name, ext = os.path.splitext(basename)
            return f"{name}_z{z_index:03d}{ext}"

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

    #_pattern = re.compile(r'(?:.*?_)?([A-Z]d+)(?:_s(d+))?(?:_w(d+))?(?:_z(d+))?(.w+)?$') # All components except well are optional
    _pattern = re.compile(r'(?:.*?_)?([A-Z]\d+)(?:_s(\d+))?(?:_w(\d+))?(?:_z(\d+))?(\.\w+)?$')
    #.*?_([A-Z]\d+)_s(\d+)_w(\d+)(?:_z(\d+))?(\.\w+)$
    # Regex to capture components including optional z and extension
    # Groups: 1=Well, 2=Site, 3=Channel, 4=Z-index (optional), 5=Extension

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

#
#            # Only add components if they exist
#            if site is not None:
#                result['site'] = site
#
#            if channel is not None:
#                result['channel'] = channel
#                result['wavelength'] = channel  # For backward compatibility
#
#            if z_index is not None:
#                result['z_index'] = z_index
#
            return result
        else:
            logger.debug(f"Could not parse ImageXpress filename: {filename}")
            return None

    def parse_well(self, filename: str) -> Optional[str]:
        """Parse well ID from an ImageXpress filename."""
        # Extract just the filename without the path
        basename = os.path.basename(filename)

        # Use the class-level pattern
        match = self._pattern.match(basename)
        return match.group(1) if match else None

    def parse_site(self, filename: str) -> Optional[int]:
        """Parse site number from an ImageXpress filename."""
        # Extract just the filename without the path
        basename = os.path.basename(filename)

        # Use the class-level pattern
        match = self._pattern.match(basename)
        return int(match.group(2)) if match and match.group(2) else None

    def parse_z_index(self, filename: str) -> Optional[int]:
        """Parse Z-index from an ImageXpress filename."""

        # Extract just the filename without the path
        basename = os.path.basename(filename)

        # Use the class-level pattern
        match = self._pattern.match(basename)
        if match and match.group(4):
            return int(match.group(4))

        # Try to extract Z-index from the folder name (e.g., ZStep_1)
        dirname = os.path.dirname(filename)
        if dirname:
            folder_name = os.path.basename(dirname)
            if folder_name.startswith('ZStep_'):
                try:
                    return int(folder_name.split('_')[1])
                except (IndexError, ValueError):
                    pass

        return None

    def parse_channel(self, filename: str) -> Optional[int]:
        """Parse channel/wavelength from an ImageXpress filename."""

        # Extract just the filename without the path
        basename = os.path.basename(filename)

        # Use the class-level pattern
        match = self._pattern.match(basename)
        return int(match.group(3)) if match and match.group(3) else None

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

    def parse_well(self, filename: str) -> Optional[str]:
        """
        Parse well ID from an Opera Phenix filename.

        Example: r03c04f144p05-ch3sk1fk1fl1.tiff -> 'R03C04'
        """
        # Extract just the filename without the path
        basename = os.path.basename(filename)

        # Try to match the Opera Phenix pattern
        match = self._pattern.match(basename)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            return f"R{row:02d}C{col:02d}"

        return None

    def parse_site(self, filename: str) -> Optional[int]:
        """
        Parse site (field) number from an Opera Phenix filename.

        Example: r03c04f144p05-ch3sk1fk1fl1.tiff -> 144
        """
        # Extract just the filename without the path
        basename = os.path.basename(filename)

        # Try to extract site from the Opera Phenix filename
        match = self._pattern.match(basename)
        if match:
            return int(match.group(3))

        return None

    def parse_z_index(self, filename: str) -> Optional[int]:
        """
        Parse Z-index (plane) from an Opera Phenix filename.

        Example: r03c04f144p05-ch3sk1fk1fl1.tiff -> 5
        """
        # Extract just the filename without the path
        basename = os.path.basename(filename)

        # Try to extract Z-index from the Opera Phenix filename
        match = self._pattern.match(basename)
        if match:
            return int(match.group(4))

        # Try to extract Z-index from the folder name (e.g., ZStep_1)
        dirname = os.path.dirname(filename)
        if dirname:
            folder_name = os.path.basename(dirname)
            if folder_name.startswith('ZStep_'):
                try:
                    return int(folder_name.split('_')[1])
                except (IndexError, ValueError):
                    pass

        return None

    def parse_channel(self, filename: str) -> Optional[int]:
        """
        Parse channel from an Opera Phenix filename.

        Example: r03c04f144p05-ch3sk1fk1fl1.tiff -> 3
        """
        # Extract just the filename without the path
        basename = os.path.basename(filename)

        # Try to extract channel from the Opera Phenix filename
        match = self._pattern.match(basename)
        if match:
            return int(match.group(5))

        return None

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
