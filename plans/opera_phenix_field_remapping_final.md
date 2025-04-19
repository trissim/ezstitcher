# Opera Phenix Field Remapping - Final Implementation Plan

## Overview
This document outlines the implementation for adding field remapping functionality to the OperaPhenixXmlParser class. The goal is to remap Opera Phenix field IDs to follow a logical top-left to bottom-right raster pattern based on position data in the Index.xml file.

## Implementation

### Methods to Add to OperaPhenixXmlParser

```python
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
```

### Method to Add to OperaPhenixFilenameParser

```python
def remap_field_in_filename(self, filename, xml_parser=None):
    """
    Remap the field ID in a filename to follow a top-left to bottom-right pattern.
    
    Args:
        filename (str): Original filename
        xml_parser (OperaPhenixXmlParser, optional): Parser with XML data
        
    Returns:
        str: New filename with remapped field ID
    """
    if xml_parser is None:
        return filename
        
    # Parse the filename
    metadata = self.parse_filename(filename)
    if not metadata or 'site' not in metadata:
        return filename
        
    # Get the mapping and remap the field ID
    mapping = xml_parser.get_field_id_mapping()
    new_field_id = xml_parser.remap_field_id(metadata['site'], mapping)
    
    # If the field ID didn't change, return the original filename
    if new_field_id == metadata['site']:
        return filename
        
    # Construct a new filename with the remapped field ID
    return self.construct_filename(
        well=metadata['well'],
        site=new_field_id,
        channel=metadata.get('channel'),
        z_index=metadata.get('z_index'),
        extension=metadata['extension']
    )
```

## Integration Steps

1. Add the methods to the OperaPhenixXmlParser class in `ezstitcher/microscopes/opera_phenix.py`
2. Add the remap_field_in_filename method to the OperaPhenixFilenameParser class
3. Update any relevant tests

## Usage Example

```python
# Parse the Index.xml file
xml_parser = OperaPhenixXmlParser('path/to/Index.xml')

# Get the field ID mapping
mapping = xml_parser.get_field_id_mapping()

# Remap a field ID
original_field_id = 3
new_field_id = xml_parser.remap_field_id(original_field_id, mapping)

# Remap a filename
filename_parser = OperaPhenixFilenameParser()
original_filename = 'A01f03d0.tiff'
new_filename = filename_parser.remap_field_in_filename(original_filename, xml_parser)
```

## Status: Complete
Progress: 100%
Last Updated: 2023-04-18
