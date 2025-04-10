"""
Pattern matcher for ezstitcher.

This module provides a class for matching patterns in filenames and directories.
"""

import re
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from ezstitcher.core.filename_parser import FilenameParser, ImageXpressFilenameParser

logger = logging.getLogger(__name__)


class PatternMatcher:
    """Match patterns in filenames and directories."""

    def __init__(self, filename_parser: Optional[FilenameParser] = None):
        """Initialize with an optional filename parser."""
        self.filename_parser = filename_parser or ImageXpressFilenameParser()

    def path_list_from_pattern(self, directory, pattern):
        """
        Get a list of filenames matching a pattern in a directory.

        Args:
            directory (str or Path): Directory to search
            pattern (str): Pattern to match with {iii} placeholder for site index

        Returns:
            list: List of matching filenames
        """
        directory = Path(directory)

        # Handle substitution of {series} if present (from Ashlar)
        if "{series}" in pattern:
            pattern = pattern.replace("{series}", "{iii}")

        # Convert pattern to regex
        # Replace {iii} with (\d+) to match any number of digits (padded or not)
        regex_pattern = pattern.replace('{iii}', '(\\d+)')

        # Handle _z001 suffix
        if '_z' not in regex_pattern:
            regex_pattern = regex_pattern.replace('_w1', '_w1(?:_z\\d+)?')

        # Handle .tif vs .tiff extension
        if regex_pattern.endswith('.tif'):
            regex_pattern = regex_pattern[:-4] + '(?:\.tif|\.tiff)'

        print(f"Regex pattern: {regex_pattern}")
        regex = re.compile(regex_pattern)

        # Find all matching files
        matching_files = []
        for file_path in directory.glob('*'):
            if file_path.is_file() and regex.match(file_path.name):
                matching_files.append(file_path.name)

        # If no files found, check for TimePoint_1 subdirectory
        if not matching_files and (directory / "TimePoint_1").exists():
            timepoint_dir = directory / "TimePoint_1"
            for file_path in timepoint_dir.glob('*'):
                if file_path.is_file() and regex.match(file_path.name):
                    matching_files.append(file_path.name)
            # If files found in TimePoint_1, use that directory
            if matching_files:
                print(f"Found {len(matching_files)} files in TimePoint_1 directory")
                return sorted(matching_files)

        # Print debug information
        print(f"Pattern: {pattern}, Directory: {directory}, Files found: {len(matching_files)}")
        if matching_files:
            print(f"First file: {matching_files[0]}")

        return sorted(matching_files)

    def auto_detect_patterns(self, folder_path, well_filter=None, extensions=None):
        """
        Automatically detect image patterns in a folder.

        Args:
            folder_path (str or Path): Path to the folder
            well_filter (list): Optional list of wells to include
            extensions (list): Optional list of file extensions to include

        Returns:
            dict: Dictionary mapping wells to wavelength patterns
        """
        folder_path = Path(folder_path)

        # Default extensions
        if extensions is None:
            extensions = ['.tif', '.TIF', '.tiff', '.TIFF']

        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend([f.name for f in folder_path.glob(f"*{ext}")])

        # Group by well and wavelength
        patterns_by_well = {}

        for filename in image_files:
            # Parse the filename using the filename parser
            metadata = self.filename_parser.parse_filename(filename)
            if not metadata:
                logger.warning(f"Unexpected filename format: {filename}")
                continue

            well = metadata['well']
            site = metadata['site']
            channel = metadata['channel']  # or wavelength for backward compatibility

            # Filter wells if needed
            if well_filter and well not in well_filter:
                continue

            # Initialize well pattern dictionary if needed
            if well not in patterns_by_well:
                patterns_by_well[well] = {}

            # Create pattern for this wavelength/channel
            # Use a simple approach that works for all formats
            # For ImageXpress format (which could be converted from Opera Phenix)
            # Example: A01_s001_w1.tif -> A01_s{iii}_w1.tif
            # Example: R01C01_s001_w1.tif -> R01C01_s{iii}_w1.tif

            # Get the base filename without path
            base_filename = os.path.basename(filename)

            # For all formats, replace the site number with {iii}
            # First, try the standard ImageXpress format with zero-padding
            site_str_padded = f"{site:03d}" if site < 100 else str(site)
            site_str_unpadded = str(site)

            # Try both padded and unpadded formats
            if f"_s{site_str_padded}_" in base_filename:
                pattern = base_filename.replace(f"_s{site_str_padded}_", "_s{iii}_")
            elif f"_s{site_str_unpadded}_" in base_filename:
                pattern = base_filename.replace(f"_s{site_str_unpadded}_", "_s{iii}_")
            else:
                # Try Opera Phenix format
                # Example: r01c01f001p01-ch1sk1fk1fl1.tiff -> r01c01f{iii}p01-ch1sk1fk1fl1.tiff
                site_str = f"{site}"
                pattern = re.sub(r'f' + site_str, 'f{iii}', base_filename)

                # If the pattern didn't change, try a more generic approach
                if pattern == base_filename:
                    pattern = re.sub(r'f\d+', 'f{iii}', base_filename)

            # For ImageXpress, remove z-index if present
            if isinstance(self.filename_parser, ImageXpressFilenameParser):
                pattern = re.sub(r'_z\d+', '', pattern)

            # Store the pattern for this wavelength/channel
            channel_str = str(channel)  # Convert to string for dictionary key
            patterns_by_well[well][channel_str] = pattern

        return patterns_by_well
