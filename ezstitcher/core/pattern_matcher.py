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
            # Also replace {zzz} with a dummy z-index if present
            if '{zzz}' in pattern_with_site:
                pattern_with_site = pattern_with_site.replace('{zzz}', '001')

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

    def group_patterns_by_z_index(self, patterns):
        """
        Group patterns by z-index using the filename parser.

        Args:
            patterns (list): List of patterns to group

        Returns:
            dict: Dictionary mapping z-indices to patterns
        """
        patterns_by_z = {}

        for pattern in patterns:
            # Check if this is a z-variable pattern (contains {zzz})
            if '{zzz}' in pattern:
                # This is a pattern with variable z-index
                # Add it to a special 'variable' category
                if 'variable' not in patterns_by_z:
                    patterns_by_z['variable'] = []
                patterns_by_z['variable'].append(pattern)
                continue

            # For fixed z-index patterns, extract the z-index
            # Replace {iii} with a dummy site number for parsing
            pattern_with_site = pattern.replace('{iii}', '001')
            metadata = self.filename_parser.parse_filename(pattern_with_site)

            if metadata and 'z_index' in metadata and metadata['z_index'] is not None:
                z_index = str(metadata['z_index'])
                if z_index not in patterns_by_z:
                    patterns_by_z[z_index] = []
                patterns_by_z[z_index].append(pattern)
            else:
                # If no z-index is found, put it in the '1' category (default z-index)
                if '1' not in patterns_by_z:
                    patterns_by_z['1'] = []
                patterns_by_z['1'].append(pattern)

        return patterns_by_z

    def auto_detect_patterns(self, folder_path, well_filter=None, extensions=None, group_by='channel', variable_site=True, variable_z=False):
        """
        Automatically detect image patterns in a folder.

        Args:
            folder_path (str or Path): Path to the folder
            well_filter (list): Optional list of wells to include
            extensions (list): Optional list of file extensions to include
            group_by (str): How to group patterns ('channel' or 'z_index')
            variable_site (bool): Whether to generate patterns with variable site (default: True)
            variable_z (bool): Whether to generate patterns with variable z-index (default: False)

        Returns:
            dict: Dictionary mapping wells to patterns grouped by channel or z-index
        """
        folder_path = Path(folder_path)
        extensions = extensions or ['.tif', '.TIF', '.tiff', '.TIFF']

        # Find all image files
        image_dir = ImageLocator.find_image_directory(folder_path)
        logger.info(f"Using image directory: {image_dir}")
        image_paths = ImageLocator.find_images_in_directory(image_dir, extensions, recursive=True)

        if not image_paths:
            logger.warning(f"No image files found in {folder_path}")
            return {}

        # Process all images
        patterns_by_well = defaultdict(list)
        for img_path in image_paths:
            metadata = self.filename_parser.parse_filename(img_path.name)
            if not metadata or (well_filter and metadata['well'] not in well_filter):
                continue

            well = metadata['well']
            channel = metadata['channel']
            has_z = 'z_index' in metadata

            # Generate patterns based on flags
            if "_s" in img_path.name and "_w" in img_path.name:  # ImageXpress format
                if variable_site and not variable_z:
                    # Only variable site
                    pattern = self.filename_parser.construct_filename(
                        well=well, site="{iii}", channel=channel,
                        z_index=metadata.get('z_index'),
                        extension=metadata['extension']
                    )
                    patterns_by_well[well].append(pattern)
                elif variable_z and not variable_site:
                    # Only variable z-index
                    if has_z:
                        pattern = self.filename_parser.construct_filename(
                            well=well, site=metadata['site'], channel=channel,
                            z_index="{iii}", extension=metadata['extension']
                        )
                        pattern = re.sub(r'_z\d+', '_z{iii}', pattern)
                        patterns_by_well[well].append(pattern)
                elif variable_site and variable_z:
                    # Both variable
                    if has_z:
                        pattern = self.filename_parser.construct_filename(
                            well=well, site="{iii}", channel=channel,
                            z_index="{iii}", extension=metadata['extension']
                        )
                        pattern = re.sub(r'_z\d+', '_z{iii}', pattern)
                        patterns_by_well[well].append(pattern)
                    else:
                        pattern = self.filename_parser.construct_filename(
                            well=well, site="{iii}", channel=channel,
                            extension=metadata['extension']
                        )
                        patterns_by_well[well].append(pattern)
            else:  # Opera Phenix format
                if variable_site and not variable_z:
                    # Only variable site
                    pattern = re.sub(r'f\d+', 'f{iii}', img_path.name)
                    patterns_by_well[well].append(pattern)
                elif variable_z and not variable_site:
                    # Only variable z-index
                    if has_z:
                        pattern = re.sub(r'p\d+', 'p{iii}', img_path.name)
                        patterns_by_well[well].append(pattern)
                elif variable_site and variable_z:
                    # Both variable
                    pattern = re.sub(r'f\d+', 'f{iii}', img_path.name)
                    if has_z:
                        pattern = re.sub(r'p\d+', 'p{iii}', pattern)
                    patterns_by_well[well].append(pattern)

        # Group patterns and remove duplicates
        result = {}
        for well, patterns in patterns_by_well.items():
            # Remove duplicates while preserving order
            unique_patterns = []
            for pattern in patterns:
                if pattern not in unique_patterns:
                    unique_patterns.append(pattern)
            result[well] = unique_patterns

            # Group by channel or z-index
            if group_by == 'z_index':
                result[well] = self.group_patterns_by_z_index(unique_patterns)
            else:  # Default to channel grouping
                result[well] = self.group_patterns_by_channel(unique_patterns)

        return result

