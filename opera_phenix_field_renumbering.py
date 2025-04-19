"""
Standalone implementation of Opera Phenix field renumbering functionality.

This module provides functionality to remap Opera Phenix field IDs to follow
a logical top-left to bottom-right raster pattern based on position data in the Index.xml file.
"""

import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

class OperaPhenixFieldRemapper:
    """
    Utility class for remapping Opera Phenix field IDs based on position data.
    """
    
    def __init__(self, xml_path: Union[str, Path]):
        """
        Initialize with path to Index.xml file.
        
        Args:
            xml_path: Path to the Index.xml file
        """
        self.xml_path = Path(xml_path)
        self.tree = ET.parse(str(self.xml_path))
        self.root = self.tree.getroot()
        
    def get_field_positions(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Extract field IDs and their X,Y positions from the Index.xml file.
        
        Returns:
            Dictionary mapping field IDs to lists of (x, y) position tuples
        """
        field_positions = {}
        
        # Find all Field elements
        field_elements = self.root.findall('.//Field')
        
        for field_elem in field_elements:
            # Extract field ID
            field_id_elem = field_elem.find('FieldID')
            if field_id_elem is None:
                continue
                
            field_id = int(field_id_elem.text)
            
            # Extract position data
            pos_x_elem = field_elem.find('PositionX')
            pos_y_elem = field_elem.find('PositionY')
            
            if pos_x_elem is not None and pos_y_elem is not None:
                # Get position values and convert to float
                pos_x = float(pos_x_elem.text)
                pos_y = float(pos_y_elem.text)
                
                # Handle unit conversion if needed
                x_unit = pos_x_elem.get('Unit', 'm')
                y_unit = pos_y_elem.get('Unit', 'm')
                
                # Convert to meters if in different units
                pos_x = self._convert_to_meters(pos_x, x_unit)
                pos_y = self._convert_to_meters(pos_y, y_unit)
                
                # Add to the dictionary, creating a list if this is the first entry
                if field_id not in field_positions:
                    field_positions[field_id] = []
                
                field_positions[field_id].append((pos_x, pos_y))
        
        return field_positions
    
    def _convert_to_meters(self, value: float, unit: str) -> float:
        """
        Convert a value from the given unit to meters.
        
        Args:
            value: The value to convert
            unit: The unit of the value ('m', 'mm', 'um', etc.)
            
        Returns:
            The value converted to meters
        """
        unit = unit.lower()
        if unit == 'm':
            return value
        elif unit == 'mm':
            return value * 0.001
        elif unit == 'um' or unit == 'µm':
            return value * 0.000001
        else:
            logger.warning(f"Unknown unit: {unit}, assuming meters")
            return value
    
    def normalize_field_positions(self, field_positions: Dict[int, List[Tuple[float, float]]]) -> Dict[int, Tuple[float, float]]:
        """
        Process field positions by averaging positions for fields with multiple entries.
        
        Args:
            field_positions: Dictionary mapping field IDs to lists of (x, y) position tuples
            
        Returns:
            Dictionary mapping field IDs to single (x, y) position tuples (averaged if needed)
        """
        normalized_positions = {}
        
        for field_id, positions in field_positions.items():
            if not positions:
                continue
                
            # Calculate average position if there are multiple entries
            if len(positions) > 1:
                avg_x = sum(pos[0] for pos in positions) / len(positions)
                avg_y = sum(pos[1] for pos in positions) / len(positions)
                normalized_positions[field_id] = (avg_x, avg_y)
            else:
                # Just use the single position
                normalized_positions[field_id] = positions[0]
        
        return normalized_positions
    
    def _calculate_position_tolerance(self, positions: List[float]) -> float:
        """
        Calculate a tolerance value for grouping positions.
        
        Args:
            positions: List of position values
            
        Returns:
            Tolerance value (half the minimum distance between positions)
        """
        if len(positions) <= 1:
            return 0.0001  # Default tolerance if only one position
            
        # Calculate minimum distance between positions
        sorted_positions = sorted(positions)
        min_distance = min(sorted_positions[i+1] - sorted_positions[i] 
                          for i in range(len(sorted_positions)-1))
        
        # Use half the minimum distance as tolerance
        return min_distance / 2.0
    
    def sort_fields_by_position(self, normalized_positions: Dict[int, Tuple[float, float]]) -> List[int]:
        """
        Sort fields based on their positions in a top-left to bottom-right raster pattern.
        
        Args:
            normalized_positions: Dictionary mapping field IDs to (x, y) position tuples
            
        Returns:
            List of field IDs sorted in raster scan order
        """
        # Convert to list of (field_id, x, y) tuples for sorting
        position_list = [(field_id, pos[0], pos[1]) for field_id, pos in normalized_positions.items()]
        
        # Determine the grid structure
        # First, find unique Y positions and sort them (descending, as Y increases downward)
        unique_y = sorted(set(pos[2] for pos in position_list), reverse=True)
        
        # Define row tolerance (fields within this distance are considered in the same row)
        y_tolerance = self._calculate_position_tolerance(unique_y)
        
        # Group fields by rows
        rows = []
        for y_pos in unique_y:
            # Find all fields that are approximately at this Y position
            row_fields = [pos for pos in position_list if abs(pos[2] - y_pos) <= y_tolerance]
            # Sort the row by X position (ascending, left to right)
            row_fields.sort(key=lambda pos: pos[1])
            rows.append(row_fields)
        
        # Flatten the rows into a single list of field IDs
        sorted_field_ids = [field[0] for row in rows for field in row]
        
        return sorted_field_ids
    
    def create_field_id_mapping(self, sorted_field_ids: List[int]) -> Dict[int, int]:
        """
        Create a mapping from original field IDs to new field IDs based on sorted order.
        
        Args:
            sorted_field_ids: List of field IDs sorted in raster scan order
            
        Returns:
            Dictionary mapping original field IDs to new field IDs
        """
        # Create mapping where the index in the sorted list + 1 becomes the new field ID
        field_id_mapping = {field_id: i + 1 for i, field_id in enumerate(sorted_field_ids)}
        
        return field_id_mapping
    
    def get_field_id_mapping(self) -> Dict[int, int]:
        """
        Generate a complete mapping from original field IDs to new field IDs.
        This is the main method that users will call.
        
        Returns:
            Dictionary mapping original field IDs to new field IDs
        """
        # Get field positions from XML
        field_positions = self.get_field_positions()
        
        # Normalize positions (average multiple positions for same field)
        normalized_positions = self.normalize_field_positions(field_positions)
        
        # Sort fields by position in raster pattern
        sorted_field_ids = self.sort_fields_by_position(normalized_positions)
        
        # Create mapping from original to new field IDs
        field_id_mapping = self.create_field_id_mapping(sorted_field_ids)
        
        return field_id_mapping
    
    def print_field_mapping(self, mapping: Optional[Dict[int, int]] = None) -> None:
        """
        Print the field ID mapping in a readable format.
        
        Args:
            mapping: Optional mapping to print. If None, generates a new mapping.
        """
        if mapping is None:
            mapping = self.get_field_id_mapping()
            
        print("Original Field ID -> New Field ID")
        print("-------------------------------")
        for orig_id, new_id in sorted(mapping.items(), key=lambda x: x[1]):
            print(f"{orig_id:>13} -> {new_id}")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python opera_phenix_field_renumbering.py path/to/Index.xml")
        sys.exit(1)
        
    xml_path = sys.argv[1]
    remapper = OperaPhenixFieldRemapper(xml_path)
    mapping = remapper.get_field_id_mapping()
    remapper.print_field_mapping(mapping)
