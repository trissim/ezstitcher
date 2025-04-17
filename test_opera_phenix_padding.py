#!/usr/bin/env python3

import logging
from pathlib import Path
from ezstitcher.microscopes.opera_phenix import OperaPhenixFilenameParser
from ezstitcher.core.file_system_manager import FileSystemManager

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def main():
    # Create a test directory
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create a parser
    parser = OperaPhenixFilenameParser()
    
    # Test constructing filenames with different padding
    well = "R01C01"
    site = 1
    channel = 1
    z_index = 1
    
    # Test with default padding
    filename1 = parser.construct_filename(well, site, channel, z_index)
    print(f"Default padding: {filename1}")
    
    # Test with explicit padding
    filename2 = parser.construct_filename(well, site, channel, z_index, site_padding=3, z_padding=3)
    print(f"Explicit padding (3,3): {filename2}")
    
    filename3 = parser.construct_filename(well, site, channel, z_index, site_padding=3, z_padding=2)
    print(f"Explicit padding (3,2): {filename3}")
    
    # Test with different z_index values
    for z in range(1, 11):
        filename = parser.construct_filename(well, site, channel, z)
        print(f"Z-index {z}: {filename}")
    
    # Test with placeholder
    filename_placeholder = parser.construct_filename(well, site, channel, '{iii}')
    print(f"Placeholder: {filename_placeholder}")
    
    # Test parsing
    metadata = parser.parse_filename(filename1)
    print(f"Parsed metadata: {metadata}")
    
    # Clean up
    test_dir.rmdir()

if __name__ == "__main__":
    main()
