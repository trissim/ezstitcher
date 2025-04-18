"""
ImageXpress microscope implementations for ezstitcher.

This module provides concrete implementations of FilenameParser and MetadataHandler
for ImageXpress microscopes.
"""

import os
import re
import logging
import tifffile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from ezstitcher.core.microscope_interfaces import FilenameParser, MetadataHandler
from ezstitcher.core.image_locator import ImageLocator
from ezstitcher.core.file_system_manager import FileSystemManager as fsman

logger = logging.getLogger(__name__)


class ImageXpressFilenameParser(FilenameParser):
    """
    Parser for ImageXpress microscope filenames.
    
    Handles standard ImageXpress format filenames like:
    - A01_s001_w1.tif
    - A01_s1_w1_z1.tif
    """
    
    # Regular expression pattern for ImageXpress filenames
    #_pattern = re.compile(r'(?:.*?_)?([A-Z]\d+)(?:_s(\d+))?(?:_w(\d+))?(?:_z(\d+))?(\.\w+)?$')
    _pattern = re.compile(r'(?:.*?_)?([A-Z]\d+)(?:_s(\d+|\{[^\}]*\}))?(?:_w(\d+|\{[^\}]*\}))?(?:_z(\d+|\{[^\}]*\}))?(\.\w+)?$')

    @classmethod
    def can_parse(cls, filename: str) -> bool:
        """
        Check if this parser can parse the given filename.
        
        Args:
            filename (str): Filename to check
            
        Returns:
            bool: True if this parser can parse the filename, False otherwise
        """
        # Extract just the basename
        basename = os.path.basename(filename)
        # Check if the filename matches the ImageXpress pattern
        return bool(cls._pattern.match(basename))
    
    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse an ImageXpress filename to extract all components, including extension.
        
        Args:
            filename (str): Filename to parse
            
        Returns:
            dict or None: Dictionary with extracted components or None if parsing fails
        """
        basename = os.path.basename(filename)
        match = self._pattern.match(basename)
        
        if match:
            well, site_str, channel_str, z_str, ext = match.groups()
    
            #handle {} place holders
            parse_comp = lambda s: None if not s or '{' in s else int(s)
            site = parse_comp(site_str)
            channel = parse_comp(channel_str)
            z_index = parse_comp(z_str)
            
            # Use the parsed components in the result
            result = {
                'well': well,
                'site': site,
                'channel': channel,
                'z_index': z_index,
                'extension': ext if ext else '.tif'  # Default if somehow empty
            }
            
            return result
        else:
            logger.debug(f"Could not parse ImageXpress filename: {filename}")
            return None
    
    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None, 
                          channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None, 
                          extension: str = '.tif',
                          site_padding: int = 3, z_padding: int = 3) -> str:
        """
        Construct an ImageXpress filename from components, only including parts if provided.
        
        Args:
            well (str): Well ID (e.g., 'A01')
            site (int or str, optional): Site number or placeholder string (e.g., '{iii}')
            channel (int, optional): Channel number
            z_index (int or str, optional): Z-index or placeholder string (e.g., '{zzz}')
            extension (str, optional): File extension
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 3)
            
        Returns:
            str: Constructed filename
        """
        if not well:
            raise ValueError("Well ID cannot be empty or None.")
        
        parts = [well]
        if site is not None:
            if isinstance(site, str):
                # If site is a string (e.g., '{iii}'), use it directly
                parts.append(f"_s{site}")
            else:
                # Otherwise, format it as a padded integer
                parts.append(f"_s{site:0{site_padding}d}")
        
        if channel is not None:
            parts.append(f"_w{channel}")
        
        if z_index is not None:
            if isinstance(z_index, str):
                # If z_index is a string (e.g., '{zzz}'), use it directly
                parts.append(f"_z{z_index}")
            else:
                # Otherwise, format it as a padded integer
                parts.append(f"_z{z_index:0{z_padding}d}")
        
        base_name = "".join(parts)
        return f"{base_name}{extension}"


class ImageXpressMetadataHandler(MetadataHandler):
    """
    Metadata handler for ImageXpress microscopes.
    
    Handles finding and parsing HTD files for ImageXpress microscopes.
    """
    
    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """
        Find the HTD file for an ImageXpress plate.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            Path to the HTD file, or None if not found
        """
        plate_path = Path(plate_path)
        
        # Look for ImageXpress HTD file in plate directory
        htd_files = list(plate_path.glob("*.HTD"))
        if htd_files:
            for htd_file in htd_files:
                if 'plate' in htd_file.name.lower():
                    return htd_file
            return htd_files[0]
        
        return None
    
    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """
        Get grid dimensions for stitching from HTD file.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            (grid_size_x, grid_size_y)
        """
        htd_file = self.find_metadata_file(plate_path)
        
        if htd_file:
            # Parse HTD file
            try:
                with open(htd_file, 'r') as f:
                    htd_content = f.read()
                
                # Extract grid dimensions - try multiple formats
                # First try the new format with "XSites" and "YSites"
                cols_match = re.search(r'"XSites", (\d+)', htd_content)
                rows_match = re.search(r'"YSites", (\d+)', htd_content)
                
                # If not found, try the old format with SiteColumns and SiteRows
                if not (cols_match and rows_match):
                    cols_match = re.search(r'SiteColumns=(\d+)', htd_content)
                    rows_match = re.search(r'SiteRows=(\d+)', htd_content)
                
                if cols_match and rows_match:
                    grid_size_x = int(cols_match.group(1))
                    grid_size_y = int(rows_match.group(1))
                    logger.info(f"Using grid dimensions from HTD file: {grid_size_x}x{grid_size_y}")
                    return grid_size_x, grid_size_y
            except Exception as e:
                logger.error(f"Error parsing HTD file {htd_file}: {e}")
        
        # Default grid dimensions
        logger.warning("Using default grid dimensions: 2x2")
        return 2, 2
    
    def get_pixel_size(self, plate_path: Union[str, Path]) -> Optional[float]:
        """
        Extract pixel size from TIFF metadata.
        
        Looks for spatial-calibration-x in the ImageDescription tag.
        
        Args:
            image_path: Path to a TIFF image
            
        Returns:
            float: Pixel size in microns (default 1.0 if not found)
        """
        for image_path in ImageLocator.find_images_in_directory(plate_path):
            try:
                # Read TIFF tags
                with tifffile.TiffFile(image_path) as tif:
                    # Try to get ImageDescription tag
                    if tif.pages[0].tags.get('ImageDescription'):
                        desc = tif.pages[0].tags['ImageDescription'].value
                        # Look for spatial calibration using regex
                        match = re.search(r'id="spatial-calibration-x"[^>]*value="([0-9.]+)"', desc)

                        if match:
                            print(f"Found pixel size metadata {str(float(match.group(1)))} in {image_path}")
                            return float(match.group(1))
                        
                        # Alternative pattern for some formats
                        match = re.search(r'Spatial Calibration: ([0-9.]+) [uµ]m', desc)
                        if match:

                            print(f"Found pixel size metadata {str(float(match.group(1)))} in {image_path}")
                            return float(match.group(1))

            except Exception as e:
                print(f"Error reading metadata from {image_path}: {e}")
            
            # Default value if metadata not found
            return 1.0




        """
        Get the pixel size from metadata.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            Pixel size in micrometers, or None if not available
        """
        # ImageXpress HTD files typically don't contain pixel size information
        # We would need to extract it from the image metadata
        return 0.65  # Default value in micrometers
