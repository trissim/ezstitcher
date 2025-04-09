"""
Opera Phenix XML parser for ezstitcher.

This module provides a class for parsing Opera Phenix Index.xml files.
"""

import os
import re
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set

logger = logging.getLogger(__name__)


class OperaPhenixXmlParser:
    """Parser for Opera Phenix Index.xml files."""

    def __init__(self, xml_path: Union[str, Path]):
        """
        Initialize the parser with the path to the Index.xml file.

        Args:
            xml_path: Path to the Index.xml file
        """
        self.xml_path = Path(xml_path)
        self.tree = None
        self.root = None
        self.namespace = ""
        self._parse_xml()

    def _parse_xml(self):
        """Parse the XML file and extract the namespace."""
        try:
            self.tree = ET.parse(self.xml_path)
            self.root = self.tree.getroot()

            # Extract namespace from the root tag
            match = re.match(r'{.*}', self.root.tag)
            self.namespace = match.group(0) if match else ""

            logger.info(f"Parsed Opera Phenix XML file: {self.xml_path}")
            logger.debug(f"XML namespace: {self.namespace}")
        except Exception as e:
            logger.error(f"Error parsing Opera Phenix XML file {self.xml_path}: {e}")
            raise

    def get_plate_info(self) -> Dict[str, Any]:
        """
        Extract plate information from the XML.

        Returns:
            Dict containing plate information
        """
        if self.root is None:
            return {}

        plate_elem = self.root.find(f".//{self.namespace}Plate")
        if plate_elem is None:
            logger.warning("No Plate element found in XML")
            return {}

        plate_info = {
            'plate_id': self._get_element_text(plate_elem, 'PlateID'),
            'measurement_id': self._get_element_text(plate_elem, 'MeasurementID'),
            'plate_type': self._get_element_text(plate_elem, 'PlateTypeName'),
            'rows': int(self._get_element_text(plate_elem, 'PlateRows') or 0),
            'columns': int(self._get_element_text(plate_elem, 'PlateColumns') or 0),
        }

        # Get well IDs
        well_elems = plate_elem.findall(f"{self.namespace}Well")
        plate_info['wells'] = [well.get('id') for well in well_elems if well.get('id')]

        logger.debug(f"Plate info: {plate_info}")
        return plate_info

    def get_grid_size(self) -> Tuple[int, int]:
        """
        Determine the grid size (number of fields per well).

        Returns:
            Tuple of (grid_size_x, grid_size_y)
        """
        if self.root is None:
            return (0, 0)

        # Get all wells
        wells = self.root.findall(f".//{self.namespace}Well")
        if not wells:
            logger.warning("No Well elements found in XML")
            return (0, 0)

        # Find the first well with images
        well_with_images = None
        for well in wells:
            well_id = self._get_element_text(well, 'id')
            if well_id:
                # Find all images for this well
                images = well.findall(f"{self.namespace}Image")
                if images:
                    well_with_images = well
                    break

        if well_with_images is None:
            logger.warning("No wells with images found in XML")
            return (0, 0)

        # Extract field IDs from image IDs
        field_ids = set()
        for image in well_with_images.findall(f"{self.namespace}Image"):
            image_id = image.get('id', '')
            # Parse field ID from image ID (format: WWWWK1FXXXPYYRZ)
            # Where WWWW is well ID, XXX is field ID, YY is plane ID, Z is channel ID
            match = re.search(r'K\d+F(\d+)P\d+R\d+', image_id)
            if match:
                field_ids.add(int(match.group(1)))

        # Count unique field IDs
        num_fields = len(field_ids)
        if num_fields == 0:
            logger.warning("No field IDs found in images")
            return (0, 0)

        # Assume a square grid for now (can be refined later)
        # Most microscopes use square grids (e.g., 2x2, 3x3, etc.)
        grid_size = int(num_fields ** 0.5)

        # If not a perfect square, try to find a reasonable grid size
        if grid_size ** 2 != num_fields:
            # Find factors of num_fields
            factors = []
            for i in range(1, int(num_fields ** 0.5) + 1):
                if num_fields % i == 0:
                    factors.append((i, num_fields // i))

            # Choose the factor pair with the smallest difference
            if factors:
                factors.sort(key=lambda x: abs(x[0] - x[1]))
                grid_size_x, grid_size_y = factors[0]
                logger.info(f"Non-square grid detected: {grid_size_x}x{grid_size_y} ({num_fields} fields)")
                return (grid_size_x, grid_size_y)

        logger.info(f"Detected grid size: {grid_size}x{grid_size} ({num_fields} fields)")
        return (grid_size, grid_size)

    def get_pixel_size(self) -> float:
        """
        Extract pixel size information from the XML.

        Returns:
            Pixel size in micrometers
        """
        if self.root is None:
            return 0.0

        # First, look for ImageResolutionX in the XML (most common in Opera Phenix)
        resolution_elem = self.root.find(f".//{self.namespace}ImageResolutionX")
        if resolution_elem is not None and resolution_elem.text:
            try:
                # Get the unit attribute
                unit = resolution_elem.get('Unit', '')
                value = float(resolution_elem.text)

                # Convert to micrometers based on unit
                if unit.lower() == 'm':
                    # Convert from meters to micrometers
                    pixel_size = value * 1e6
                    logger.info(f"Found pixel size from ImageResolutionX: {pixel_size:.4f} µm")
                    return pixel_size
                else:
                    logger.warning(f"Unknown resolution unit: {unit}, assuming meters")
                    pixel_size = value * 1e6
                    logger.info(f"Found pixel size from ImageResolutionX: {pixel_size:.4f} µm (assuming meters)")
                    return pixel_size
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing ImageResolutionX: {e}")

        # If not found in ImageResolutionX, look for PixelSize in the PixelSizeCalibration section
        pixel_size_elem = self.root.find(f".//{self.namespace}PixelSize")
        if pixel_size_elem is not None and pixel_size_elem.text:
            try:
                # Get the unit attribute to ensure we're returning micrometers
                unit = pixel_size_elem.get('Unit', '')
                value = float(pixel_size_elem.text)

                # Convert to micrometers if necessary
                if unit.lower() in ['µm', 'um', 'micrometer', 'micrometers']:
                    logger.info(f"Found pixel size from PixelSize: {value:.4f} µm")
                    return value
                elif unit.lower() in ['m', 'meter', 'meters']:
                    pixel_size = value * 1e6
                    logger.info(f"Found pixel size from PixelSize: {pixel_size:.4f} µm")
                    return pixel_size
                elif unit.lower() in ['mm', 'millimeter', 'millimeters']:
                    pixel_size = value * 1e3
                    logger.info(f"Found pixel size from PixelSize: {pixel_size:.4f} µm")
                    return pixel_size
                elif unit.lower() in ['nm', 'nanometer', 'nanometers']:
                    pixel_size = value * 1e-3
                    logger.info(f"Found pixel size from PixelSize: {pixel_size:.4f} µm")
                    return pixel_size
                else:
                    logger.warning(f"Unknown pixel size unit: {unit}, assuming micrometers")
                    logger.info(f"Found pixel size from PixelSize: {value:.4f} µm (assuming micrometers)")
                    return value
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing PixelSize: {e}")

        # If not found, use the default value
        logger.warning("Pixel size not found in XML, using default value")
        return 0.65  # Default value in micrometers

    def get_channel_info(self) -> List[Dict[str, Any]]:
        """
        Extract channel information from the XML.

        Returns:
            List of dictionaries containing channel information
        """
        if self.root is None:
            return []

        # Look for channel entries in the Map section
        channel_entries = self.root.findall(f".//{self.namespace}Map/{self.namespace}Entry")
        if not channel_entries:
            logger.warning("No channel entries found in XML")
            return []

        channel_info = []
        for entry in channel_entries:
            channel_id = entry.get('ChannelID')
            if channel_id:
                channel_data = {
                    'channel_id': int(channel_id),
                    'magnification': self._get_element_text(entry, 'ObjectiveMagnification'),
                    'na': self._get_element_text(entry, 'ObjectiveNA'),
                    'exposure_time': self._get_element_text(entry, 'ExposureTime'),
                    'exposure_time_unit': self._get_element_attribute(entry, 'ExposureTime', 'Unit'),
                    'channel_name': self._get_element_text(entry, 'ChannelName'),
                }
                channel_info.append(channel_data)

        # Sort by channel ID
        channel_info.sort(key=lambda x: x['channel_id'])

        logger.debug(f"Channel info: {channel_info}")
        return channel_info

    def get_image_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract image information from the XML.

        Returns:
            Dictionary mapping image IDs to dictionaries containing image information
        """
        if self.root is None:
            return {}

        # Look for Image elements
        image_elems = self.root.findall(f".//{self.namespace}Image[@Version]")
        if not image_elems:
            logger.warning("No Image elements with Version attribute found in XML")
            return {}

        image_info = {}
        for image in image_elems:
            image_id = self._get_element_text(image, 'id')
            if image_id:
                image_data = {
                    'url': self._get_element_text(image, 'URL'),
                    'row': int(self._get_element_text(image, 'Row') or 0),
                    'col': int(self._get_element_text(image, 'Col') or 0),
                    'field_id': int(self._get_element_text(image, 'FieldID') or 0),
                    'plane_id': int(self._get_element_text(image, 'PlaneID') or 0),
                    'channel_id': int(self._get_element_text(image, 'ChannelID') or 0),
                    'position_x': self._get_element_text(image, 'PositionX'),
                    'position_y': self._get_element_text(image, 'PositionY'),
                    'position_z': self._get_element_text(image, 'PositionZ'),
                }
                image_info[image_id] = image_data

        logger.debug(f"Found {len(image_info)} images in XML")
        return image_info

    def get_z_step_size(self) -> float:
        """
        Calculate the Z-step size from image positions.

        Returns:
            Z-step size in micrometers
        """
        if self.root is None:
            return 0.0

        # Extract all PositionZ elements
        position_z_elems = self.root.findall(f".//{self.namespace}PositionZ")
        if not position_z_elems:
            logger.warning("No PositionZ elements found in XML")
            return 0.0

        # Extract unique Z positions
        z_positions = set()
        for elem in position_z_elems:
            if elem.text and elem.get('Unit', '').lower() == 'm':
                z_positions.add(float(elem.text))

        # Sort Z positions
        z_positions = sorted(z_positions)
        logger.debug(f"Found {len(z_positions)} unique Z positions: {z_positions}")

        if len(z_positions) <= 1:
            logger.warning("Not enough Z positions to calculate step size")
            return 0.0

        # Calculate differences between consecutive Z positions
        diffs = [abs(z_positions[i+1] - z_positions[i]) for i in range(len(z_positions)-1)]

        # Use the median difference as the Z-step size
        if diffs:
            # Find the most common difference (mode)
            from collections import Counter
            diff_counts = Counter(diffs)
            most_common_diff = diff_counts.most_common(1)[0][0]

            # Convert from meters to micrometers
            z_step_size = most_common_diff * 1e6
            logger.info(f"Calculated Z-step size: {z_step_size:.2f} µm")
            return z_step_size

        logger.warning("Could not calculate Z-step size from XML")
        return 0.0

    def get_well_positions(self) -> Dict[str, Tuple[int, int]]:
        """
        Extract well positions from the XML.

        Returns:
            Dictionary mapping well IDs to (row, column) tuples
        """
        if self.root is None:
            return {}

        # Look for Well elements
        well_elems = self.root.findall(f".//{self.namespace}Wells/{self.namespace}Well")
        if not well_elems:
            logger.warning("No Well elements found in XML")
            return {}

        well_positions = {}
        for well in well_elems:
            well_id = self._get_element_text(well, 'id')
            row = self._get_element_text(well, 'Row')
            col = self._get_element_text(well, 'Col')

            if well_id and row and col:
                well_positions[well_id] = (int(row), int(col))

        logger.debug(f"Well positions: {well_positions}")
        return well_positions

    def _get_element_text(self, parent_elem, tag_name: str) -> Optional[str]:
        """Helper method to get element text with namespace."""
        elem = parent_elem.find(f"{self.namespace}{tag_name}")
        return elem.text if elem is not None else None

    def _get_element_attribute(self, parent_elem, tag_name: str, attr_name: str) -> Optional[str]:
        """Helper method to get element attribute with namespace."""
        elem = parent_elem.find(f"{self.namespace}{tag_name}")
        return elem.get(attr_name) if elem is not None else None
