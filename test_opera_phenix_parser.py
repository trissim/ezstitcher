#!/usr/bin/env python3
"""
Test script for the Opera Phenix XML parser.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the Python path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ezstitcher.core.opera_phenix_xml_parser import OperaPhenixXmlParser

def main():
    # Path to the Opera Phenix Index.xml file
    xml_path = Path("/home/ts/code/projects/ezstitcher/sample_data/Opera/20250407TS-12w_axoTest-2__2025-04-07T15_10_15-Measurement 2/Images/Index.xml")

    # Check if the file exists
    if not xml_path.exists():
        print(f"Error: File not found: {xml_path}")
        return

    print(f"Parsing Opera Phenix Index.xml file: {xml_path}")

    # Create the parser
    parser = OperaPhenixXmlParser(xml_path)

    # Get plate information
    plate_info = parser.get_plate_info()
    print("\nPlate Information:")
    for key, value in plate_info.items():
        print(f"  {key}: {value}")

    # Get grid size
    grid_size = parser.get_grid_size()
    print(f"\nGrid Size: {grid_size[0]}x{grid_size[1]}")

    # Get pixel size
    pixel_size = parser.get_pixel_size()
    print(f"\nPixel Size: {pixel_size} µm")

    # Find ImageResolutionX element
    resolution_elem = parser.root.find(f".//{parser.namespace}ImageResolutionX")
    if resolution_elem is not None and resolution_elem.text:
        unit = resolution_elem.get('Unit', '')
        value = float(resolution_elem.text)
        print(f"\nImageResolutionX: {value} {unit} ({value*1e6:.4f} µm)")
    else:
        print("\nImageResolutionX not found in XML")

    # Get channel information
    channel_info = parser.get_channel_info()
    print("\nChannel Information:")
    for channel in channel_info:
        print(f"  Channel {channel.get('channel_id')}:")
        for key, value in channel.items():
            if key != 'channel_id':
                print(f"    {key}: {value}")

    # Get well positions
    well_positions = parser.get_well_positions()
    print("\nWell Positions:")
    for well_id, position in well_positions.items():
        print(f"  {well_id}: Row {position[0]}, Column {position[1]}")

    # Get Z-step size
    z_step_size = parser.get_z_step_size()
    print(f"\nZ-Step Size: {z_step_size} µm")

    # Get unique Z positions
    position_z_elems = parser.root.findall(f".//{parser.namespace}PositionZ")
    z_positions = set()
    for elem in position_z_elems:
        if elem.text and elem.get('Unit', '').lower() == 'm':
            z_positions.add(float(elem.text))

    # Sort Z positions
    z_positions = sorted(z_positions)
    print(f"\nUnique Z positions ({len(z_positions)}):")
    for i, pos in enumerate(z_positions):
        print(f"  Z{i+1}: {pos} m ({pos*1e6:.2f} µm)")

    # Calculate differences between consecutive Z positions
    if len(z_positions) > 1:
        diffs = [abs(z_positions[i+1] - z_positions[i]) for i in range(len(z_positions)-1)]
        print(f"\nZ-step differences:")
        for i, diff in enumerate(diffs):
            print(f"  Diff {i+1}: {diff} m ({diff*1e6:.2f} µm)")

    # Get image information (limit to first 5 images)
    image_info = parser.get_image_info()
    print(f"\nImage Information (showing first 5 of {len(image_info)} images):")
    for i, (image_id, info) in enumerate(image_info.items()):
        if i >= 5:
            break
        print(f"  Image {image_id}:")
        for key, value in info.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
