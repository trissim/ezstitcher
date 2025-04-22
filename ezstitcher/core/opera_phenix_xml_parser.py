"""
Opera Phenix XML parser for ezstitcher.

This module provides a class for parsing Opera Phenix Index.xml files.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple
import re
import numpy as np

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

            logger.info("Parsed Opera Phenix XML file: %s", self.xml_path)
            logger.debug("XML namespace: %s", self.namespace)
        except Exception as e:
            logger.error("Error parsing Opera Phenix XML file %s: %s", self.xml_path, e)
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

        logger.debug("Plate info: %s", plate_info)
        return plate_info

    def get_grid_size(self) -> Tuple[int, int]:
        """
        Determine the grid size (number of fields per well) by analyzing image positions.

        This method analyzes the positions of images for a single well, channel, and plane
        to determine the grid dimensions.

        Returns:
            Tuple of (grid_size_x, grid_size_y)
        """
        if self.root is None:
            logger.error("XML not parsed, cannot determine grid size")
            return (2, 2)  # Default grid size

        # Get all image elements
        image_elements = self.root.findall(f".//{self.namespace}Image")

        if not image_elements:
            logger.warning("No Image elements found in XML")
            return (2, 2)  # Default grid size

        # Group images by well (Row+Col), channel, and plane
        # We'll use the first group with multiple fields to determine grid size
        image_groups = {}

        for image in image_elements:
            # Extract well, channel, and plane information
            row_elem = image.find(f"{self.namespace}Row")
            col_elem = image.find(f"{self.namespace}Col")
            channel_elem = image.find(f"{self.namespace}ChannelID")
            plane_elem = image.find(f"{self.namespace}PlaneID")

            if (row_elem is not None and row_elem.text and
                col_elem is not None and col_elem.text and
                channel_elem is not None and channel_elem.text and
                plane_elem is not None and plane_elem.text):

                # Create a key for grouping
                group_key = f"R{row_elem.text}C{col_elem.text}_CH{channel_elem.text}_P{plane_elem.text}"

                # Extract position information
                pos_x_elem = image.find(f"{self.namespace}PositionX")
                pos_y_elem = image.find(f"{self.namespace}PositionY")
                field_elem = image.find(f"{self.namespace}FieldID")

                if (pos_x_elem is not None and pos_x_elem.text and
                    pos_y_elem is not None and pos_y_elem.text and
                    field_elem is not None and field_elem.text):

                    try:
                        # Parse position values
                        x_value = float(pos_x_elem.text)
                        y_value = float(pos_y_elem.text)
                        field_id = int(field_elem.text)

                        # Add to group
                        if group_key not in image_groups:
                            image_groups[group_key] = []

                        image_groups[group_key].append({
                            'field_id': field_id,
                            'pos_x': x_value,
                            'pos_y': y_value,
                            'pos_x_unit': pos_x_elem.get('Unit', ''),
                            'pos_y_unit': pos_y_elem.get('Unit', '')
                        })
                    except (ValueError, TypeError):
                        logger.warning("Could not parse position values for image in group %s", group_key)

        # Find the first group with multiple fields
        for group_key, images in image_groups.items():
            if len(images) > 1:
                logger.debug("Using image group %s with %d fields to determine grid size", group_key, len(images))

                # Extract unique X and Y positions
                # Use a small epsilon for floating point comparison
                epsilon = 1e-10
                x_positions = [img['pos_x'] for img in images]
                y_positions = [img['pos_y'] for img in images]

                # Use numpy to find unique positions
                unique_x = np.unique(np.round(np.array(x_positions) / epsilon) * epsilon)
                unique_y = np.unique(np.round(np.array(y_positions) / epsilon) * epsilon)

                # Count unique positions
                num_x_positions = len(unique_x)
                num_y_positions = len(unique_y)

                # If we have a reasonable number of positions, use them as grid dimensions
                if num_x_positions > 0 and num_y_positions > 0:
                    logger.info("Determined grid size from positions: %dx%d", num_x_positions, num_y_positions)
                    return (num_x_positions, num_y_positions)

                # Alternative approach: try to infer grid size from field IDs
                if len(images) > 1:
                    # Sort images by field ID
                    sorted_images = sorted(images, key=lambda x: x['field_id'])
                    max_field_id = sorted_images[-1]['field_id']

                    # Try to determine if it's a square grid
                    grid_size = int(np.sqrt(max_field_id) + 0.5)  # Round to nearest integer

                    if grid_size ** 2 == max_field_id:
                        logger.info("Determined square grid size from field IDs: %dx%d", grid_size, grid_size)
                        return (grid_size, grid_size)

                    # If not a perfect square, try to find factors
                    for i in range(1, int(np.sqrt(max_field_id)) + 1):
                        if max_field_id % i == 0:
                            j = max_field_id // i
                            logger.info("Determined grid size from field IDs: %dx%d", i, j)
                            return (i, j)

        # If we couldn't determine grid size, use a default
        logger.warning("Could not determine grid size from XML, using default 2x2")
        return (2, 2)  # Default grid size

    def get_pixel_size(self) -> float:
        """
        Extract pixel size from the XML.

        The pixel size is stored in ImageResolutionX/Y elements with Unit="m".

        Returns:
            Pixel size in micrometers (μm)
        """
        if self.root is None:
            logger.warning("XML not parsed, using default pixel size")
            return 0.65  # Default value in micrometers

        # Try to find ImageResolutionX element
        resolution_x = self.root.find(f".//{self.namespace}ImageResolutionX")
        if resolution_x is not None and resolution_x.text:
            try:
                # Convert from meters to micrometers
                pixel_size = float(resolution_x.text) * 1e6
                logger.info("Found pixel size from ImageResolutionX: %.4f μm", pixel_size)
                return pixel_size
            except (ValueError, TypeError):
                logger.warning("Could not parse pixel size from ImageResolutionX")

        # If not found in ImageResolutionX, try ImageResolutionY
        resolution_y = self.root.find(f".//{self.namespace}ImageResolutionY")
        if resolution_y is not None and resolution_y.text:
            try:
                # Convert from meters to micrometers
                pixel_size = float(resolution_y.text) * 1e6
                logger.info("Found pixel size from ImageResolutionY: %.4f μm", pixel_size)
                return pixel_size
            except (ValueError, TypeError):
                logger.warning("Could not parse pixel size from ImageResolutionY")

        # If not found, use default value
        logger.warning("Pixel size not found in XML, using default value of 0.65 μm")
        return 0.65  # Default value in micrometers



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

        logger.debug("Found %d images in XML", len(image_info))
        return image_info



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

        logger.debug("Well positions: %s", well_positions)
        return well_positions

    def _get_element_text(self, parent_elem, tag_name: str) -> Optional[str]:
        """Helper method to get element text with namespace."""
        elem = parent_elem.find(f"{self.namespace}{tag_name}")
        return elem.text if elem is not None else None

    def _get_element_attribute(self, parent_elem, tag_name: str, attr_name: str) -> Optional[str]:
        """Helper method to get element attribute with namespace."""
        elem = parent_elem.find(f"{self.namespace}{tag_name}")
        return elem.get(attr_name) if elem is not None else None

    def get_field_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Extract field IDs and their X,Y positions from the Index.xml file.

        Returns:
            dict: Mapping of field IDs to (x, y) position tuples
        """
        field_positions = {}

        # Find all Image elements
        image_elems = self.root.findall(f".//{self.namespace}Image")

        for image in image_elems:
            # Check if this element has FieldID, PositionX, and PositionY children
            field_id_elem = image.find(f"{self.namespace}FieldID")
            pos_x_elem = image.find(f"{self.namespace}PositionX")
            pos_y_elem = image.find(f"{self.namespace}PositionY")

            if field_id_elem is not None and pos_x_elem is not None and pos_y_elem is not None:
                try:
                    field_id = int(field_id_elem.text)
                    pos_x = float(pos_x_elem.text)
                    pos_y = float(pos_y_elem.text)

                    # Only add if we don't already have this field ID
                    if field_id not in field_positions:
                        field_positions[field_id] = (pos_x, pos_y)
                except (ValueError, TypeError):
                    # Skip entries with invalid data
                    continue

        return field_positions

    def sort_fields_by_position(self, positions: Dict[int, Tuple[float, float]]) -> list:
        """
        Sort fields based on their positions in a raster pattern starting from the top.
        All rows go left-to-right in a consistent raster scan pattern.

        Args:
            positions: Dictionary mapping field IDs to (x, y) position tuples

        Returns:
            list: Field IDs sorted in raster pattern order starting from the top
        """
        if not positions:
            return []

        # Get all unique x and y coordinates
        x_coords = sorted(set(pos[0] for pos in positions.values()))
        y_coords = sorted(set(pos[1] for pos in positions.values()), reverse=True)  # Reverse to get top row first

        # Create a grid of field IDs
        grid = {}
        for field_id, (x, y) in positions.items():
            # Find the closest x and y coordinates in our sorted lists
            x_idx = x_coords.index(x)
            y_idx = y_coords.index(y)  # This will now map top row to index 0
            grid[(x_idx, y_idx)] = field_id

        # Debug output to help diagnose field mapping issues
        logger.info("Field position grid:")
        for y_idx in range(len(y_coords)):
            row_str = ""
            for x_idx in range(len(x_coords)):
                field_id = grid.get((x_idx, y_idx), 0)
                row_str += f"{field_id:3d} "
            logger.info(row_str)

        # Sort field IDs by row (y) then column (x)
        # Use raster pattern: all rows go left-to-right in a consistent pattern
        sorted_field_ids = []
        for y_idx in range(len(y_coords)):
            row_fields = []
            # All rows go left to right in a raster pattern
            x_range = range(len(x_coords))

            for x_idx in x_range:
                if (x_idx, y_idx) in grid:
                    row_fields.append(grid[(x_idx, y_idx)])
            sorted_field_ids.extend(row_fields)

        return sorted_field_ids

    def get_field_id_mapping(self) -> Dict[int, int]:
        """
        Generate a mapping from original field IDs to new field IDs based on position data.

        Returns:
            dict: Mapping of original field IDs to new field IDs
        """
        # Get field positions
        field_positions = self.get_field_positions()

        # Sort fields by position
        sorted_field_ids = self.sort_fields_by_position(field_positions)

        # Create mapping from original to new field IDs
        return {field_id: i + 1 for i, field_id in enumerate(sorted_field_ids)}

    def remap_field_id(self, field_id: int, mapping: Optional[Dict[int, int]] = None) -> int:
        """
        Remap a field ID using the position-based mapping.

        Args:
            field_id: Original field ID
            mapping: Mapping to use. If None, generates a new mapping.

        Returns:
            int: New field ID, or original if not in mapping
        """
        if mapping is None:
            mapping = self.get_field_id_mapping()

        return mapping.get(field_id, field_id)
