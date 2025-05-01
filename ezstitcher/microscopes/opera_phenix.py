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

#from ezstitcher.core.microscope_interfaces import FilenameParser, MetadataHandler
from ezstitcher.core.microscope_base import FilenameParser, MetadataHandler
from ezstitcher.core.microscope_interfaces import MicroscopeHandler

from ezstitcher.core.opera_phenix_xml_parser import OperaPhenixXmlParser
# Removed: from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.io.filemanager import FileManager # Added

logger = logging.getLogger(__name__)


class OperaPhenixHandler(MicroscopeHandler):
    """
    MicroscopeHandler implementation for Opera Phenix systems.

    This handler combines the OperaPhenix filename parser with its
    corresponding metadata handler. It guarantees aligned behavior
    for plate structure parsing, metadata extraction, and any optional
    post-processing steps required after workspace setup.
    """

    def __init__(self):
        super().__init__(
            parser=OperaPhenixFilenameParser(),
            metadata_handler=OperaPhenixMetadataHandler()
        )

    @property
    def common_dirs(self) -> str:
        """Subdirectory names commonly used by Opera Phenix."""
        return 'Image'

    def _normalize_workspace(self, workspace_path: Path, fm=None) -> Path:
        """
        Renames Opera Phenix images to follow a consistent field order
        based on spatial layout extracted from Index.xml. Uses remapped
        filenames and replaces the directory in-place.

        Args:
            workspace_path: Path to the symlinked workspace
            fm: Optional FileManager instance. If None, uses self.file_manager

        Returns:
            Path to the normalized image directory.
        """
        # Use the provided file manager or fall back to the instance's file manager
        fm = fm or self.file_manager
        image_dir = fm.find_image_directory(workspace_path)

        # Locate Index.xml and load mapping
        index_xml = self.file_manager.find_file_recursive(workspace_path, "Index.xml")
        if not index_xml:
            raise ValueError(f"Index.xml not found in workspace: {workspace_path}")

        xml_parser = OperaPhenixXmlParser(index_xml)
        field_mapping = xml_parser.get_field_id_mapping()

        temp_dir = image_dir / "__renamed"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for file in image_dir.glob("*.tif*"):
            if not file.is_file():
                continue

            # Parse file metadata
            metadata = self.parser.parse_filename(file.name)
            if not metadata or 'site' not in metadata or metadata['site'] is None:
                continue

            # Remap the field ID using the spatial layout
            original_field_id = metadata['site']
            new_field_id = field_mapping.get(original_field_id, original_field_id)

            new_name = self.parser.construct_filename(
                well=metadata['well'],
                site=new_field_id,
                channel=metadata['channel'],
                z_index=metadata['z_index'],
                extension=metadata['extension'],
                site_padding=3,
                z_padding=3
            )

            new_path = temp_dir / new_name
            fm.copy_file(file, new_path)

        # Clean up and replace old dir
        for file in image_dir.glob("*.tif*"):
            fm.delete_file(file)

        for file in temp_dir.iterdir():
            # Use copy and delete since FileManager doesn't have a move_file method
            fm.copy_file(file, image_dir / file.name)
            fm.delete_file(file)

        # Remove the temp directory (use Path.rmdir() since FileManager doesn't have a remove_directory method)
        temp_dir.rmdir()

        return image_dir


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
        logger.debug(f"OperaPhenixFilenameParser attempting to parse basename: '{basename}'") # Add logging

        # Try parsing using the Opera Phenix pattern
        match = self._pattern.match(basename)
        if match:
            logger.debug(f"Regex match successful for '{basename}'") # Add logging
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

        logger.warning(f"Regex match failed for basename: '{basename}'") # Add logging
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

    def __init__(self, file_manager: Optional[FileManager] = None):
        """
        Initialize the metadata handler.

        Args:
            file_manager: FileManager instance. If None, a disk-based FileManager is created.
        """
        if file_manager is None:
            file_manager = FileManager(backend='disk')
            logger.debug("Created default disk-based FileManager for OperaPhenixMetadataHandler")

        self.file_manager = file_manager

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """Finds the Index.xml file using file_manager.find_file_recursive."""
        # TRANSITIONAL: Assumes Index.xml exists on a file system accessible
        # by the backend used by file_manager (likely disk).
        try:
            # Use injected file_manager
            return self.file_manager.find_file_recursive(plate_path, "Index.xml")
        except Exception as e:
            logger.error(f"Error finding Index.xml in {plate_path}: {e}")
            return None

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """
        Get grid dimensions for stitching from Index.xml file.

        Args:
            plate_path: Path to the plate folder

        Returns:
            (grid_size_x, grid_size_y)

        Raises:
            ValueError: If grid dimensions cannot be determined from metadata
        """
        index_xml = self.find_metadata_file(plate_path)
        if not index_xml:
            raise ValueError(f"Cannot find Index.xml in {plate_path}. Grid dimensions cannot be determined.")

        try:
            # Use the OperaPhenixXmlParser to get the grid size
            xml_parser = self.create_xml_parser(index_xml)
            grid_size = xml_parser.get_grid_size()

            if grid_size[0] > 0 and grid_size[1] > 0:
                logger.info("Determined grid size from Opera Phenix Index.xml: %dx%d", grid_size[0], grid_size[1])
                return grid_size
            else:
                raise ValueError(f"Invalid grid dimensions (0x0) found in Index.xml")
        except Exception as e:
            logger.error("Error parsing Opera Phenix Index.xml: %s", e)
            raise ValueError(f"Failed to extract grid dimensions from Index.xml: {e}") from e

    def get_pixel_size(self, plate_path: Union[str, Path]) -> float:
        """
        Get the pixel size from Index.xml file.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Pixel size in micrometers

        Raises:
            ValueError: If pixel size cannot be determined from metadata
        """
        index_xml = self.find_metadata_file(plate_path)
        if not index_xml:
            raise ValueError(f"Cannot find Index.xml in {plate_path}. Pixel size cannot be determined.")

        try:
            # Use the OperaPhenixXmlParser to get the pixel size
            xml_parser = self.create_xml_parser(index_xml)
            pixel_size = xml_parser.get_pixel_size()

            if pixel_size > 0:
                logger.info("Determined pixel size from Opera Phenix Index.xml: %.4f μm", pixel_size)
                return pixel_size
            else:
                raise ValueError(f"Invalid pixel size (0 or negative) found in Index.xml")
        except Exception as e:
            logger.error("Error getting pixel size from Opera Phenix Index.xml: %s", e)
            raise ValueError(f"Failed to extract pixel size from Index.xml: {e}") from e

    def create_xml_parser(self, xml_path: Union[str, Path]):
        """
        Create an OperaPhenixXmlParser for the given XML file.

        Args:
            xml_path: Path to the XML file

        Returns:
            OperaPhenixXmlParser: Parser for the XML file
        """
        return OperaPhenixXmlParser(xml_path)
