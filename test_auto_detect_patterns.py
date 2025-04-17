#!/usr/bin/env python3

import logging
from pathlib import Path
from ezstitcher.core.microscope_interfaces import MicroscopeHandler

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    # Create a MicroscopeHandler with auto-detection
    plate_folder = Path("tests/test_data/synthetic_data/imagexpress")
    handler = MicroscopeHandler(plate_folder=plate_folder, microscope_type='auto')
    
    # Test auto_detect_patterns with different variable components
    print("\nTesting with variable_components=['site']:")
    patterns = handler.auto_detect_patterns(
        plate_folder,
        variable_components=['site']
    )
    print_patterns(patterns)
    
    print("\nTesting with variable_components=['z_index']:")
    patterns = handler.auto_detect_patterns(
        plate_folder,
        variable_components=['z_index']
    )
    print_patterns(patterns)
    
    print("\nTesting with variable_components=['site', 'z_index']:")
    patterns = handler.auto_detect_patterns(
        plate_folder,
        variable_components=['site', 'z_index']
    )
    print_patterns(patterns)

def print_patterns(patterns):
    """Print patterns in a readable format."""
    for well, channel_patterns in patterns.items():
        print(f"Well: {well}")
        for channel, pattern_list in channel_patterns.items():
            print(f"  Channel: {channel}")
            for pattern in pattern_list:
                print(f"    Pattern: {pattern}")

if __name__ == "__main__":
    main()
