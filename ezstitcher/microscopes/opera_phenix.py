"""
Opera Phenix microscope implementations for ezstitcher.

This module provides concrete implementations of FilenameParser and MetadataHandler
for Opera Phenix microscopes.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from ezstitcher.core.microscope_interfaces import FilenameParser, MetadataHandler
from ezstitcher.core.opera_phenix_xml_parser import OperaPhenixXmlParser
from ezstitcher.core.file_system_manager import FileSystemManager

logger = logging.getLogger(__name__)


class OperaPhenixFilenameParser(FilenameParser):
    """Parser for Opera Phenix microscope filenames.

    Handles Opera Phenix format filenames like:
    - r01c01f001p01-ch1sk1fk1fl1.tiff
    - r01c01f001p01-ch1.tiff
    """

    # Regular expression pattern for Opera Phenix filenames
    _pattern = re.compile(r"r(\d{1,2})c(\d{1,2})f(\d+|\{[^\}]*\})p(\d+|\{[^\}]*\})-ch(\d+|\{[^\}]*\})(?:sk\d+)?(?:fk\d+)?(?:fl\d+)?(\.\w+)$", re.I)

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
        Supports placeholders like {iii} which will return None for that field.

        Args:
            filename (str): Filename to parse

        Returns:
            dict or None: Dictionary with extracted components or None if parsing fails.
        """
        basename = os.path.basename(filename)

        # Try parsing using the Opera Phenix pattern
        match = self._pattern.match(basename)
        if match:
            row, col, site_str, z_str, channel_str, ext = match.groups()

            # Helper function to parse component strings
            parse_comp = lambda s: None if not s or '{' in s else int(s)

            # Create well ID from row and column
            well = f"R{int(row):02d}C{int(col):02d}"

            # Parse components
            site = parse_comp(site_str)
            channel = parse_comp(channel_str)
            z_index = parse_comp(z_str)

            result = {
                'well': well,
                'site': site,
                'channel': channel,
                'wavelength': channel,  # For backward compatibility
                'z_index': z_index,
                'extension': ext if ext else '.tif'
            }
            return result

        return None

    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None, channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None, extension: str = '.tiff',
                          site_padding: int = 3, z_padding: int = 3) -> str:
        """
        Construct an Opera Phenix filename from components.

        Args:
            well (str): Well ID (e.g., 'R03C04' or 'A01')
            site: Site/field number (int) or placeholder string
            channel (int): Channel number
            z_index: Z-index/plane (int) or placeholder string
            extension (str, optional): File extension
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 3)

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
        else:
            raise ValueError(f"Invalid well format: {well}. Expected format: 'R01C03'")

        # Default Z-index to 1 if not provided
        z_index = 1 if z_index is None else z_index
        channel = 1 if channel is None else channel

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

    def remap_field_in_filename(self, filename: str, xml_parser: Optional[OperaPhenixXmlParser] = None) -> str:
        """
        Remap the field ID in a filename to follow a top-left to bottom-right pattern.

        Args:
            filename: Original filename
            xml_parser: Parser with XML data

        Returns:
            str: New filename with remapped field ID
        """
        if xml_parser is None:
            return filename

        # Parse the filename
        metadata = self.parse_filename(filename)
        if not metadata or 'site' not in metadata or metadata['site'] is None:
            return filename

        # Get the mapping and remap the field ID
        mapping = xml_parser.get_field_id_mapping()
        new_field_id = xml_parser.remap_field_id(metadata['site'], mapping)

        # Always create a new filename with the remapped field ID and consistent padding
        # This ensures all filenames have the same format, even if the field ID didn't change
        return self.construct_filename(
            well=metadata['well'],
            site=new_field_id,
            channel=metadata['channel'],
            z_index=metadata['z_index'],
            extension=metadata['extension'],
            site_padding=3,
            z_padding=3
        )


class OperaPhenixMetadataHandler(MetadataHandler):
    """
    Metadata handler for Opera Phenix microscopes.

    Handles finding and parsing Index.xml files for Opera Phenix microscopes.
    """

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """
        Find the Index.xml file for an Opera Phenix plate.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Path to the Index.xml file, or None if not found
        """
        return FileSystemManager.find_file_recursive(plate_path, "Index.xml")

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """
        Get grid dimensions for stitching from Index.xml file.

        Args:
            plate_path: Path to the plate folder

        Returns:
            (grid_size_x, grid_size_y)
        """
        index_xml = self.find_metadata_file(plate_path)

        if index_xml:
            try:
                # Use the OperaPhenixXmlParser to get the grid size
                xml_parser = self.create_xml_parser(index_xml)
                grid_size = xml_parser.get_grid_size()

                if grid_size[0] > 0 and grid_size[1] > 0:
                    logger.info("Determined grid size from Opera Phenix Index.xml: %dx%d", grid_size[0], grid_size[1])
                    return grid_size
            except Exception as e:
                logger.error("Error parsing Opera Phenix Index.xml: %s", e)

        # Default grid dimensions
        logger.warning("Using default grid dimensions: 2x2")
        return 2, 2

    def get_pixel_size(self, plate_path: Union[str, Path]) -> Optional[float]:
        """
        Get the pixel size from Index.xml file.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Pixel size in micrometers, or None if not available
        """
        index_xml = self.find_metadata_file(plate_path)

        if index_xml:
            try:
                # Use the OperaPhenixXmlParser to get the pixel size
                xml_parser = self.create_xml_parser(index_xml)
                pixel_size = xml_parser.get_pixel_size()

                if pixel_size > 0:
                    logger.info("Determined pixel size from Opera Phenix Index.xml: %.4f μm", pixel_size)
                    return pixel_size
            except Exception as e:
                logger.error("Error getting pixel size from Opera Phenix Index.xml: %s", e)

        # Default value
        logger.warning("Using default pixel size: 0.65 μm")
        return 0.65  # Default value in micrometers

    def create_xml_parser(self, xml_path: Union[str, Path]):
        """
        Create an OperaPhenixXmlParser for the given XML file.

        Args:
            xml_path: Path to the XML file

        Returns:
            OperaPhenixXmlParser: Parser for the XML file
        """
        return OperaPhenixXmlParser(xml_path)
