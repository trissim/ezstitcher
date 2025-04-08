"""
Filename parser for ezstitcher.

This module provides a class for parsing microscopy image filenames.
"""

import re
import logging

logger = logging.getLogger(__name__)


class FilenameParser:
    """Parse microscopy image filenames to extract components."""
    
    @staticmethod
    def parse_filename(filename):
        """
        Parse a microscopy image filename to extract well, site, wavelength, and z-index.
        
        Args:
            filename (str): Filename to parse
            
        Returns:
            dict: Dictionary with extracted components or None if parsing fails
        """
        # Common pattern: WellID_sXXX_wY_zZZZ.tif
        # Example: A01_s001_w1_z001.tif
        pattern = re.compile(r'([A-Z]\d+)_s(\d+)_w(\d+)(?:_z(\d+))?\..*')
        match = pattern.match(filename)
        
        if match:
            well = match.group(1)
            site = int(match.group(2))
            wavelength = int(match.group(3))
            z_index = int(match.group(4)) if match.group(4) else None
            
            return {
                'well': well,
                'site': site,
                'wavelength': wavelength,
                'z_index': z_index
            }
        
        return None
    
    @staticmethod
    def pad_site_number(filename):
        """
        Ensure site number is padded to 3 digits.
        
        Args:
            filename (str): Filename to pad
            
        Returns:
            str: Filename with padded site number
        """
        # Match site number pattern
        pattern = re.compile(r'(_s)(\d+)(_)')
        match = pattern.search(filename)
        
        if match:
            prefix = match.group(1)
            site_num = match.group(2)
            suffix = match.group(3)
            
            # Pad site number to 3 digits
            padded_site = site_num.zfill(3)
            
            # Replace in filename
            return filename.replace(f"{prefix}{site_num}{suffix}", f"{prefix}{padded_site}{suffix}")
        
        return filename
    
    @staticmethod
    def construct_filename(well, site, wavelength, z_index=None, extension='.tif'):
        """
        Construct a filename from components.
        
        Args:
            well (str): Well ID (e.g., 'A01')
            site (int): Site number
            wavelength (int): Wavelength number
            z_index (int, optional): Z-index
            extension (str, optional): File extension
            
        Returns:
            str: Constructed filename
        """
        # Pad site number to 3 digits
        site_str = f"{site:03d}"
        
        if z_index is not None:
            return f"{well}_s{site_str}_w{wavelength}_z{z_index:03d}{extension}"
        else:
            return f"{well}_s{site_str}_w{wavelength}{extension}"
