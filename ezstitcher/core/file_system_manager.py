"""
File system manager for ezstitcher.

This module provides a class for managing file system operations.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Pattern
import tifffile
import numpy as np

from ezstitcher.core.filename_parser import FilenameParser
from ezstitcher.core.csv_handler import CSVHandler
from ezstitcher.core.pattern_matcher import PatternMatcher
from ezstitcher.core.directory_manager import DirectoryManager

logger = logging.getLogger(__name__)


class FileSystemManager:
    """
    Manages file system operations for ezstitcher.
    Abstracts away direct file system interactions for improved testability.
    """
    
    def __init__(self, config=None):
        """
        Initialize the FileSystemManager.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config or {}
        self.default_extensions = ['.tif', '.TIF', '.tiff', '.TIFF', 
                                  '.jpg', '.JPG', '.jpeg', '.JPEG', 
                                  '.png', '.PNG']
        self.filename_parser = FilenameParser()
        self.csv_handler = CSVHandler()
        self.pattern_matcher = PatternMatcher()
        self.directory_manager = DirectoryManager()
    
    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory (str or Path): Directory path to ensure exists
            
        Returns:
            Path: Path object for the directory
        """
        return self.directory_manager.ensure_directory(directory)
    
    def list_image_files(self, directory: Union[str, Path], 
                         extensions: Optional[List[str]] = None) -> List[Path]:
        """
        List all image files in a directory with specified extensions.
        
        Args:
            directory (str or Path): Directory to search
            extensions (list): List of file extensions to include
            
        Returns:
            list: List of Path objects for image files
        """
        if extensions is None:
            extensions = self.default_extensions

        directory = Path(directory)
        image_files = []

        for ext in extensions:
            image_files.extend(list(directory.glob(f"*{ext}")))

        return sorted(image_files)
    
    def path_list_from_pattern(self, directory: Union[str, Path], pattern: str) -> List[str]:
        """
        Get a list of filenames matching a pattern in a directory.
        
        Args:
            directory (str or Path): Directory to search
            pattern (str): Pattern to match with {iii} placeholder for site index
            
        Returns:
            list: List of matching filenames
        """
        return self.pattern_matcher.path_list_from_pattern(directory, pattern)
    
    def load_image(self, file_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load an image and ensure it's 2D grayscale.
        
        Args:
            file_path (str or Path): Path to the image file
            
        Returns:
            numpy.ndarray: 2D grayscale image or None if loading fails
        """
        try:
            img = tifffile.imread(str(file_path))

            # Convert to 2D grayscale if needed
            if img.ndim == 3:
                # Check if it's a channel-first format (C, H, W)
                if img.shape[0] <= 4:  # Assuming max 4 channels (RGBA)
                    # Convert channel-first to 2D by taking mean across channels
                    img = np.mean(img, axis=0).astype(img.dtype)
                # Check if it's a channel-last format (H, W, C)
                elif img.shape[2] <= 4:  # Assuming max 4 channels (RGBA)
                    # Convert channel-last to 2D by taking mean across channels
                    img = np.mean(img, axis=2).astype(img.dtype)
                else:
                    # If it's a 3D image with a different structure, use the first slice
                    img = img[0].astype(img.dtype)

            return img
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None
    
    def save_image(self, file_path: Union[str, Path], image: np.ndarray, 
                  compression: Optional[str] = None) -> bool:
        """
        Save an image to disk.
        
        Args:
            file_path (str or Path): Path to save the image
            image (numpy.ndarray): Image to save
            compression (str or None): Compression method
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            directory = Path(file_path).parent
            directory.mkdir(parents=True, exist_ok=True)

            # Save image
            tifffile.imwrite(str(file_path), image, compression=compression)
            return True
        except Exception as e:
            logger.error(f"Error saving image {file_path}: {e}")
            return False
    
    def find_files_by_pattern(self, directory: Union[str, Path], 
                             pattern: Union[str, Pattern]) -> List[Path]:
        """
        Find files matching a regex pattern.
        
        Args:
            directory (str or Path): Directory to search
            pattern (str or Pattern): Regex pattern to match
            
        Returns:
            list: List of matching Path objects
        """
        directory = Path(directory)
        
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
            
        matching_files = []
        for file_path in directory.glob('*'):
            if pattern.match(file_path.name):
                matching_files.append(file_path)
                
        return sorted(matching_files)
    
    def parse_positions_csv(self, csv_path: Union[str, Path]) -> List[Tuple[str, float, float]]:
        """
        Parse a CSV file with lines of the form:
          file: <filename>; grid: (col, row); position: (x, y)
        
        Args:
            csv_path (str or Path): Path to the CSV file
            
        Returns:
            list: List of tuples (filename, x_float, y_float)
        """
        return self.csv_handler.parse_positions_csv(csv_path)
    
    def generate_positions_df(self, image_files, positions, grid_positions):
        """
        Generate a DataFrame with position information.
        
        Args:
            image_files (list): List of image filenames
            positions (list): List of (x, y) position tuples
            grid_positions (list): List of (row, col) grid position tuples
            
        Returns:
            pandas.DataFrame: DataFrame with position information
        """
        return self.csv_handler.generate_positions_df(image_files, positions, grid_positions)
    
    def save_positions_df(self, df, positions_path):
        """
        Save a positions DataFrame to CSV.
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            positions_path (str or Path): Path to save the CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.csv_handler.save_positions_df(df, positions_path)
    
    def find_htd_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """
        Find the HTD file for a plate.
        
        Args:
            plate_path (str or Path): Path to the plate folder
            
        Returns:
            Path or None: Path to the HTD file, or None if not found
        """
        plate_path = Path(plate_path)
        
        # Look in plate directory
        htd_files = list(plate_path.glob("*.HTD"))
        if htd_files:
            for htd_file in htd_files:
                if 'plate' in htd_file.name.lower():
                    return htd_file
            return htd_files[0]

        # Look in parent directory
        parent_dir = plate_path.parent
        htd_files = list(parent_dir.glob("*.HTD"))
        if htd_files:
            for htd_file in htd_files:
                if 'plate' in htd_file.name.lower():
                    return htd_file
            return htd_files[0]

        return None
    
    def parse_htd_file(self, htd_path: Union[str, Path]) -> Optional[Tuple[int, int]]:
        """
        Parse an HTD file to extract grid dimensions.
        
        Args:
            htd_path (str or Path): Path to the HTD file
            
        Returns:
            tuple: (grid_size_x, grid_size_y) or None if parsing fails
        """
        try:
            with open(htd_path, 'r') as f:
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
                return grid_size_x, grid_size_y
            else:
                logger.warning(f"Could not parse grid dimensions from HTD file: {htd_path}")
                return None
        except Exception as e:
            logger.error(f"Error parsing HTD file {htd_path}: {e}")
            return None
    
    def find_wells(self, timepoint_dir: Union[str, Path]) -> List[str]:
        """
        Find all wells in the timepoint directory.
        
        Args:
            timepoint_dir (str or Path): Path to the TimePoint_1 directory
            
        Returns:
            list: List of well names (e.g., ['A01', 'A02', ...])
        """
        return self.directory_manager.find_wells(timepoint_dir)
    
    def clean_temp_folders(self, parent_dir: Union[str, Path], base_name: str, keep_suffixes=None) -> None:
        """
        Clean up temporary folders created during processing.
        
        Args:
            parent_dir (str or Path): Parent directory
            base_name (str): Base name of the plate folder
            keep_suffixes (list, optional): List of suffixes to keep
        """
        self.directory_manager.clean_temp_folders(parent_dir, base_name, keep_suffixes)
    
    def create_output_directories(self, parent_dir, plate_name, suffixes):
        """
        Create output directories for a plate.
        
        Args:
            parent_dir (str or Path): Parent directory
            plate_name (str): Name of the plate
            suffixes (dict): Dictionary mapping directory types to suffixes
            
        Returns:
            dict: Dictionary mapping directory types to Path objects
        """
        return self.directory_manager.create_output_directories(parent_dir, plate_name, suffixes)
    
    def parse_filename(self, filename):
        """
        Parse a microscopy image filename.
        
        Args:
            filename (str): Filename to parse
            
        Returns:
            dict: Dictionary with extracted components
        """
        return self.filename_parser.parse_filename(filename)
    
    def pad_site_number(self, filename):
        """
        Ensure site number is padded to 3 digits.
        
        Args:
            filename (str): Filename to pad
            
        Returns:
            str: Filename with padded site number
        """
        return self.filename_parser.pad_site_number(filename)
    
    def construct_filename(self, well, site, wavelength, z_index=None, extension='.tif'):
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
        return self.filename_parser.construct_filename(well, site, wavelength, z_index, extension)
    
    def auto_detect_patterns(self, folder_path, well_filter=None):
        """
        Automatically detect image patterns in a folder.
        
        Args:
            folder_path (str or Path): Path to the folder
            well_filter (list): Optional list of wells to include
            
        Returns:
            dict: Dictionary mapping wells to wavelength patterns
        """
        return self.pattern_matcher.auto_detect_patterns(folder_path, well_filter)
