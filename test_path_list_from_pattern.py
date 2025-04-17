#!/usr/bin/env python3

import logging
from pathlib import Path
from ezstitcher.core.microscope_interfaces import MicroscopeHandler
from ezstitcher.microscopes.opera_phenix import OperaPhenixFilenameParser

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def main():
    # Create a test directory with some sample files
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)

    # Create some sample Opera Phenix files
    sample_files = [
        "r01c01f001p01-ch1sk1fk1fl1.tiff",
        "r01c01f002p01-ch1sk1fk1fl1.tiff",
        "r01c01f003p01-ch1sk1fk1fl1.tiff",
        "r01c01f001p02-ch1sk1fk1fl1.tiff",
        "r01c01f001p01-ch2sk1fk1fl1.tiff",
        "r02c03f001p01-ch1sk1fk1fl1.tiff",
    ]

    print("Sample files created:")
    for filename in sample_files:
        print(f"  {filename}")

    for filename in sample_files:
        (test_dir / filename).touch()

    # Create an Opera Phenix parser
    parser = OperaPhenixFilenameParser()

    # Print the parsed metadata for each file
    print("\nParsed metadata for each file:")
    for filename in sample_files:
        metadata = parser.parse_filename(filename)
        print(f"  {filename}: {metadata}")

    # Test with an Opera Phenix pattern
    pattern = "r01c01f{iii}p01-ch1sk1fk1fl1.tiff"
    print(f"\nTesting pattern: {pattern}")
    pattern_with_dummy = pattern.replace('{iii}', '001')
    pattern_metadata = parser.parse_filename(pattern_with_dummy)
    print(f"Pattern metadata: {pattern_metadata}")
    matching_files = parser.path_list_from_pattern(test_dir, pattern)
    print(f"Matching files: {matching_files}")

    # Test with a different pattern
    pattern = "r01c01f001p{iii}-ch1sk1fk1fl1.tiff"
    print(f"\nTesting pattern: {pattern}")
    pattern_with_dummy = pattern.replace('{iii}', '001')
    pattern_metadata = parser.parse_filename(pattern_with_dummy)
    print(f"Pattern metadata: {pattern_metadata}")
    matching_files = parser.path_list_from_pattern(test_dir, pattern)
    print(f"Matching files: {matching_files}")

    # Test with a channel pattern
    pattern = "r01c01f001p01-ch{iii}sk1fk1fl1.tiff"
    print(f"\nTesting pattern: {pattern}")
    pattern_with_dummy = pattern.replace('{iii}', '001')
    pattern_metadata = parser.parse_filename(pattern_with_dummy)
    print(f"Pattern metadata: {pattern_metadata}")
    matching_files = parser.path_list_from_pattern(test_dir, pattern)
    print(f"Matching files: {matching_files}")

    # Clean up
    for filename in sample_files:
        (test_dir / filename).unlink()
    test_dir.rmdir()

if __name__ == "__main__":
    main()
