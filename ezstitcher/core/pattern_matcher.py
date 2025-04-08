"""
Pattern matcher for ezstitcher.

This module provides a class for matching patterns in filenames and directories.
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PatternMatcher:
    """Match patterns in filenames and directories."""
    
    @staticmethod
    def path_list_from_pattern(directory, pattern):
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
        regex = re.compile(regex_pattern)
        
        # Find all matching files
        matching_files = []
        for file_path in directory.glob('*'):
            if file_path.is_file() and regex.match(file_path.name):
                matching_files.append(file_path.name)
        
        return sorted(matching_files)
    
    @staticmethod
    def auto_detect_patterns(folder_path, well_filter=None, extensions=None):
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
        
        # Pattern to extract well, site, wavelength from filename
        # Example: A01_s001_w1.tif or A01_s001_w1_z001.tif
        filename_pattern = re.compile(r'([A-Z]\d+)_s(\d+)_w(\d+)(?:_z\d+)?\..*')
        
        for filename in image_files:
            match = filename_pattern.match(filename)
            if not match:
                continue
                
            well = match.group(1)
            site_str = match.group(2)
            site = int(site_str)
            wavelength = match.group(3)
            
            # Filter wells if needed
            if well_filter and well not in well_filter:
                continue
                
            # Initialize well pattern dictionary if needed
            if well not in patterns_by_well:
                patterns_by_well[well] = {}
                
            # Create pattern for this wavelength
            # First, ensure the site number is padded to 3 digits
            padded_site = f"{site:03d}"
            padded_filename = filename.replace(f"_s{site_str}", f"_s{padded_site}")
            
            # Now replace the padded site with the placeholder
            pattern = re.sub(r'_s\d{3}', '_s{iii}', padded_filename)
            
            # Remove z-index if present
            pattern = re.sub(r'_z\d+', '', pattern)
            
            # Add to patterns
            patterns_by_well[well][wavelength] = pattern
        
        return patterns_by_well
