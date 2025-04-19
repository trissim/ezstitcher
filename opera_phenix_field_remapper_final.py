"""
Final implementation of Opera Phenix field renumbering functionality.

This module provides functionality to remap Opera Phenix field IDs to follow
a logical top-left to bottom-right raster pattern based on position data in the Index.xml file.
"""

def get_field_positions(self):
    """
    Extract field IDs and their X,Y positions from the Index.xml file.
    
    Returns:
        dict: Mapping of field IDs to (x, y) position tuples
    """
    field_positions = {}
    
    # Extract namespace from root tag if present
    namespace = ''
    if '}' in self.root.tag:
        namespace = '{' + self.root.tag.split('}')[0][1:] + '}'
    
    # Find all Image elements
    for image_elem in self.root.findall(f'.//{namespace}Image'):
        # Check if this element has FieldID, PositionX, and PositionY children
        field_id_elem = image_elem.find(f'.//{namespace}FieldID')
        pos_x_elem = image_elem.find(f'.//{namespace}PositionX')
        pos_y_elem = image_elem.find(f'.//{namespace}PositionY')
        
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

def sort_fields_by_position(self, positions):
    """
    Sort fields based on their positions in a top-left to bottom-right raster pattern.
    
    Args:
        positions (dict): Mapping of field IDs to (x, y) position tuples
        
    Returns:
        list: Field IDs sorted in raster scan order
    """
    if not positions:
        return []
        
    # Convert to list of (field_id, x, y) tuples for sorting
    pos_list = [(field_id, x, y) for field_id, (x, y) in positions.items()]
    
    # Find unique Y positions
    y_values = sorted(set(y for _, _, y in pos_list))
    
    # Calculate tolerance for grouping rows
    if len(y_values) > 1:
        # Use half the minimum distance between rows as tolerance
        min_y_diff = min(y_values[i+1] - y_values[i] for i in range(len(y_values)-1))
        y_tolerance = min_y_diff / 2
    else:
        y_tolerance = 0.0001  # Default tolerance
    
    # Group fields by rows (Y position)
    rows = []
    for y_pos in sorted(y_values, reverse=True):  # Reverse to start from top
        row_fields = [pos for pos in pos_list if abs(pos[2] - y_pos) <= y_tolerance]
        # Sort each row by X position (left to right)
        row_fields.sort(key=lambda pos: pos[1])
        rows.append(row_fields)
    
    # Flatten the rows into a single list of field IDs
    sorted_field_ids = [field[0] for row in rows for field in row]
    
    return sorted_field_ids

def get_field_id_mapping(self):
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

def remap_field_id(self, field_id, mapping=None):
    """
    Remap a field ID using the position-based mapping.
    
    Args:
        field_id (int): Original field ID
        mapping (dict, optional): Mapping to use. If None, generates a new mapping.
        
    Returns:
        int: New field ID, or original if not in mapping
    """
    if mapping is None:
        mapping = self.get_field_id_mapping()
        
    return mapping.get(field_id, field_id)
