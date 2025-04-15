import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import re

from ezstitcher.core.image_locator import ImageLocator
from collections import defaultdict
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

        # Handle _z001 suffix if not already in the pattern
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

        # Print debug information
        print(f"Pattern: {pattern}, Directory: {directory}, Files found: {len(matching_files)}")
        if matching_files:
            print(f"First file: {matching_files[0]}")

        # Use natural sorting instead of lexicographical sorting
        return self._natural_sort(matching_files)

    def metadata_from_pattern(self, pattern):
        """
        Extract metadata from a filename pattern.

        Args:
            pattern (str): Filename pattern

        Returns:
            dict: Dictionary with extracted metadata
        """

        return self.filename_parser.parse_filename(pattern.replace('{iii}', '001'))

    def _natural_sort(self, file_list):
        """
        Sort filenames naturally, so that site numbers are sorted numerically.
        E.g., ["s1", "s10", "s2"] -> ["s1", "s2", "s10"]

        Args:
            file_list (list): List of filenames to sort

        Returns:
            list: Naturally sorted list of filenames
        """
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        return sorted(file_list, key=natural_sort_key)

    def group_patterns_by_channel(self, patterns):
        """
        Group patterns by channel/wavelength using the filename parser.

        Args:
            patterns (list): List of patterns to group

        Returns:
            dict: Dictionary mapping channel numbers to patterns
        """
        patterns_by_channel = {}

        for pattern in patterns:
            # Use the filename parser to extract the channel
            # Replace {iii} with a dummy site number for parsing
            pattern_with_site = pattern.replace('{iii}', '001')
            metadata = self.filename_parser.parse_filename(pattern_with_site)

            if metadata and 'channel' in metadata:
                channel = str(metadata['channel'])
                if channel not in patterns_by_channel:
                    patterns_by_channel[channel] = []
                patterns_by_channel[channel].append(pattern)
            else:
                logger.warning(f"Could not extract channel from pattern: {pattern}")
                continue

        return patterns_by_channel

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

        # Use ImageLocator to find all image files (including in subdirectories)
        image_dir = ImageLocator.find_image_directory(folder_path)
        logger.info(f"Using image directory: {image_dir}")
        image_paths = ImageLocator.find_images_in_directory(image_dir, extensions, recursive=True)

        if not image_paths:
            logger.warning(f"No image files found in {folder_path}")
            return {}

        # Use defaultdict to simplify initialization
        # patterns_by_well[well][channel] will automatically create a list when accessed
        patterns_by_well = defaultdict(list)

        # Process all images in a single pass
        for img_path in image_paths:
            # Parse the filename using the filename parser
            metadata = self.filename_parser.parse_filename(img_path.name)
            if not metadata:
                logger.warning(f"Unexpected filename format: {img_path.name}")
                continue

            well = metadata['well']
            site = metadata['site']
            channel = metadata['channel']
            z_index = metadata.get('z_index')  # May be None for non-Z-stack images

            # Filter wells if needed
            if well_filter and well not in well_filter:
                continue

            # Track z-indices for this well and channel
            channel_str = str(channel)

            # Create pattern with proper padding and z-suffixes
            base_filename = img_path.name

            # Always use padded site numbers
            site_str_padded = f"{site:03d}"

            # Create pattern based on format
            if "_s" in base_filename and "_w" in base_filename:  # ImageXpress format
                # Replace site number with {iii} placeholder
                if f"_s{site_str_padded}_" in base_filename:
                    pattern = base_filename.replace(f"_s{site_str_padded}_", "_s{iii}_")
                else:
                    # Handle non-padded site numbers
                    pattern = re.sub(r'_s\d+_', '_s{iii}_', base_filename)
            else:  # Opera Phenix format
                # Example: r01c01f001p01-ch1sk1fk1fl1.tiff -> r01c01f{iii}p01-ch1sk1fk1fl1.tiff
                pattern = re.sub(r'f\d+', 'f{iii}', base_filename)

            # Store the pattern for this wavelength/channel
            # The defaultdict will automatically create a list if it doesn't exist
            patterns_by_well[well].append(pattern)

        # Group patterns by channel for each well
        result = {}
        for well, patterns in patterns_by_well.items():
            # Remove duplicates while preserving order
            unique_patterns = []
            for pattern in patterns:
                if pattern not in unique_patterns:
                    unique_patterns.append(pattern)

            # Group by channel
            result[well] = self.group_patterns_by_channel(unique_patterns)

        return result

