#!/usr/bin/env python
"""
Test script for Opera Phenix field remapping functionality.

This script demonstrates how to use the field remapping functionality
to remap Opera Phenix field IDs based on their positions in the Index.xml file.
"""

import os
import sys
from pathlib import Path
import logging

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ezstitcher.core.opera_phenix_xml_parser import OperaPhenixXmlParser
from ezstitcher.microscopes.opera_phenix import OperaPhenixFilenameParser

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_field_remapping():
    """
    Test the field remapping functionality.
    """
    # Use the Index.xml file in the project tests directory
    xml_path = Path(__file__).parent / "Index.xml"
    assert xml_path.exists(), f"Index.xml file not found at: {xml_path}"

    logger.info(f"Testing field remapping with XML file: {xml_path}")

    # Parse the XML file
    xml_parser = OperaPhenixXmlParser(xml_path)

    # Get field positions
    field_positions = xml_parser.get_field_positions()
    logger.info(f"Found {len(field_positions)} fields with position data")

    # Print field positions
    for field_id, position in field_positions.items():
        logger.info(f"Field {field_id}: position {position}")

    # Sort fields by position
    sorted_field_ids = xml_parser.sort_fields_by_position(field_positions)
    logger.info(f"Fields sorted by position: {sorted_field_ids}")

    # Get field ID mapping
    field_id_mapping = xml_parser.get_field_id_mapping()
    logger.info("Field ID mapping:")

    # Create a reverse mapping (new_id -> original_id)
    reverse_mapping = {}
    for original_id, new_id in field_id_mapping.items():
        if new_id not in reverse_mapping:
            reverse_mapping[new_id] = []
        reverse_mapping[new_id].append(original_id)

    # Display mapping as (new_id, original_id)
    logger.info("Format: (new_field_id, original_field_id)")
    for new_id in sorted(reverse_mapping.keys()):
        original_ids = reverse_mapping[new_id]
        if len(original_ids) == 1:
            logger.info(f"  ({new_id}, {original_ids[0]})")
        else:
            # Handle case where multiple original IDs map to the same new ID
            logger.info(f"  ({new_id}, {original_ids})")

    # Also show the original mapping for reference
    logger.info("\nOriginal mapping (original_id -> new_id):")
    for original_id, new_id in sorted(field_id_mapping.items()):
        logger.info(f"  {original_id} -> {new_id}")

    # Test remapping a filename
    filename_parser = OperaPhenixFilenameParser()

    # Create a test filename for each field
    for original_id, new_id in field_id_mapping.items():
        test_filename = f"r01c01f{original_id:03d}p01-ch1sk1fk1fl1.tiff"
        remapped_filename = filename_parser.remap_field_in_filename(test_filename, xml_parser)
        logger.info(f"Original filename: {test_filename}")
        logger.info(f"Remapped filename: {remapped_filename}")

        # Verify the remapping
        # The remapped filename will have 3-digit padding for both field and z-index
        expected_filename = f"r01c01f{new_id:03d}p001-ch1sk1fk1fl1.tiff"
        assert remapped_filename == expected_filename, f"Expected {expected_filename}, got {remapped_filename}"

if __name__ == "__main__":
    # Use the Index.xml file in the main directory
    index_xml_path = Path(__file__).parent / "Index.xml"

    if not index_xml_path.exists():
        logger.error(f"Index.xml file not found at: {index_xml_path}")
        sys.exit(1)

    # Test with the Index.xml file in the main directory
    test_field_remapping(index_xml_path)

    logger.info("All tests passed!")
