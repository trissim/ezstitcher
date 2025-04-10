"""
Image locator module for ezstitcher.

This module provides a class for locating images in various directory structures.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Pattern

logger = logging.getLogger(__name__)


class ImageLocator:
    """
    Locates images in various directory structures.
    
    This class provides methods to find images in different directory structures:
    - Images directly in the plate folder
    - Images in a TimePoint_1 subfolder
    - Images in Z-stack folders in the plate folder
    - Images in Z-stack folders in the TimePoint_1 subfolder
    - Images in an Images subfolder
    - Images in an Images/TimePoint_1 subfolder
    """
    
    DEFAULT_EXTENSIONS = ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    
    @staticmethod
    def find_images_in_directory(directory: Union[str, Path], 
                                extensions: Optional[List[str]] = None) -> List[Path]:
        """
        Find all images in a directory.
        
        Args:
            directory: Directory to search
            extensions: List of file extensions to include. If None, uses DEFAULT_EXTENSIONS.
            
        Returns:
            List of Path objects for image files
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
            
        if extensions is None:
            extensions = ImageLocator.DEFAULT_EXTENSIONS
            
        image_files = []
        for ext in extensions:
            image_files.extend(list(directory.glob(f"*{ext}")))
            
        return sorted(image_files)
    
    @staticmethod
    def find_images_by_pattern(directory: Union[str, Path], 
                              pattern: Union[str, Pattern],
                              extensions: Optional[List[str]] = None) -> List[Path]:
        """
        Find images matching a pattern in a directory.
        
        Args:
            directory: Directory to search
            pattern: Regex pattern to match
            extensions: List of file extensions to include. If None, uses DEFAULT_EXTENSIONS.
            
        Returns:
            List of Path objects for matching image files
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
            
        if extensions is None:
            extensions = ImageLocator.DEFAULT_EXTENSIONS
            
        # Compile pattern if it's a string
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
            
        image_files = []
        for ext in extensions:
            for file_path in directory.glob(f"*{ext}"):
                if pattern.search(file_path.name):
                    image_files.append(file_path)
                    
        return sorted(image_files)
    
    @staticmethod
    def find_timepoint_dir(plate_folder: Union[str, Path], 
                          timepoint_dir_name: str = "TimePoint_1") -> Optional[Path]:
        """
        Find the TimePoint directory in various locations.
        
        Checks the following locations:
        1. plate_folder/timepoint_dir_name
        2. plate_folder/Images/timepoint_dir_name
        
        Args:
            plate_folder: Path to the plate folder
            timepoint_dir_name: Name of the timepoint directory
            
        Returns:
            Path to the timepoint directory if found, None otherwise
        """
        plate_folder = Path(plate_folder)
        
        # Check if timepoint directory exists directly in plate folder
        timepoint_dir = plate_folder / timepoint_dir_name
        if timepoint_dir.exists() and timepoint_dir.is_dir():
            logger.debug(f"Found timepoint directory: {timepoint_dir}")
            return timepoint_dir
            
        # Check if timepoint directory exists in Images subfolder
        images_dir = plate_folder / "Images"
        if images_dir.exists() and images_dir.is_dir():
            images_timepoint_dir = images_dir / timepoint_dir_name
            if images_timepoint_dir.exists() and images_timepoint_dir.is_dir():
                logger.debug(f"Found timepoint directory in Images: {images_timepoint_dir}")
                return images_timepoint_dir
                
        logger.debug(f"No timepoint directory found in {plate_folder}")
        return None
    
    @staticmethod
    def find_z_stack_dirs(plate_folder: Union[str, Path], 
                         timepoint_dir_name: str = "TimePoint_1") -> List[Tuple[int, Path]]:
        """
        Find Z-stack directories in various locations.
        
        Checks the following locations:
        1. plate_folder/timepoint_dir_name/ZStep_*
        2. plate_folder/Images/timepoint_dir_name/ZStep_*
        3. plate_folder/ZStep_*
        
        Args:
            plate_folder: Path to the plate folder
            timepoint_dir_name: Name of the timepoint directory
            
        Returns:
            List of (z_index, directory) tuples
        """
        plate_folder = Path(plate_folder)
        z_stack_dirs = []
        z_pattern = re.compile(r'ZStep_(\d+)')
        
        # Check if Z-stack directories exist in timepoint directory
        timepoint_dir = ImageLocator.find_timepoint_dir(plate_folder, timepoint_dir_name)
        if timepoint_dir:
            for item in timepoint_dir.iterdir():
                if item.is_dir():
                    match = z_pattern.match(item.name)
                    if match:
                        z_index = int(match.group(1))
                        z_stack_dirs.append((z_index, item))
                        
        # Check if Z-stack directories exist directly in plate folder
        if not z_stack_dirs:
            for item in plate_folder.iterdir():
                if item.is_dir():
                    match = z_pattern.match(item.name)
                    if match:
                        z_index = int(match.group(1))
                        z_stack_dirs.append((z_index, item))
                        
        # Sort by Z-index
        z_stack_dirs.sort(key=lambda x: x[0])
        
        if z_stack_dirs:
            logger.debug(f"Found {len(z_stack_dirs)} Z-stack directories")
        else:
            logger.debug(f"No Z-stack directories found in {plate_folder}")
            
        return z_stack_dirs
    
    @staticmethod
    def find_image_locations(plate_folder: Union[str, Path], 
                            timepoint_dir_name: str = "TimePoint_1",
                            extensions: Optional[List[str]] = None) -> Dict[str, List[Path]]:
        """
        Find all possible image locations in a plate folder.
        
        Args:
            plate_folder: Path to the plate folder
            timepoint_dir_name: Name of the timepoint directory
            extensions: List of file extensions to include. If None, uses DEFAULT_EXTENSIONS.
            
        Returns:
            Dictionary mapping location types to lists of image paths
        """
        plate_folder = Path(plate_folder)
        if extensions is None:
            extensions = ImageLocator.DEFAULT_EXTENSIONS
            
        image_locations = {}
        
        # Check for images directly in plate folder
        direct_images = ImageLocator.find_images_in_directory(plate_folder, extensions)
        if direct_images:
            image_locations['plate'] = direct_images
            
        # Check for images in timepoint directory
        timepoint_dir = ImageLocator.find_timepoint_dir(plate_folder, timepoint_dir_name)
        if timepoint_dir:
            timepoint_images = ImageLocator.find_images_in_directory(timepoint_dir, extensions)
            if timepoint_images:
                image_locations['timepoint'] = timepoint_images
                
        # Check for images in Images directory
        images_dir = plate_folder / "Images"
        if images_dir.exists() and images_dir.is_dir():
            images_dir_images = ImageLocator.find_images_in_directory(images_dir, extensions)
            if images_dir_images:
                image_locations['images'] = images_dir_images
                
        # Check for images in Z-stack directories
        z_stack_dirs = ImageLocator.find_z_stack_dirs(plate_folder, timepoint_dir_name)
        if z_stack_dirs:
            z_stack_images = {}
            for z_index, z_dir in z_stack_dirs:
                z_images = ImageLocator.find_images_in_directory(z_dir, extensions)
                if z_images:
                    z_stack_images[z_index] = z_images
                    
            if z_stack_images:
                image_locations['z_stack'] = z_stack_images
                
        return image_locations
    
    @staticmethod
    def detect_directory_structure(plate_folder: Union[str, Path], 
                                  timepoint_dir_name: str = "TimePoint_1") -> str:
        """
        Detect the directory structure of a plate folder.
        
        Args:
            plate_folder: Path to the plate folder
            timepoint_dir_name: Name of the timepoint directory
            
        Returns:
            String describing the directory structure:
            - 'flat': Images directly in the plate folder
            - 'timepoint': Images in a timepoint directory
            - 'images': Images in an Images directory
            - 'images_timepoint': Images in an Images/timepoint directory
            - 'z_stack': Images in Z-stack directories
            - 'timepoint_z_stack': Images in timepoint/Z-stack directories
            - 'unknown': Unknown directory structure
        """
        plate_folder = Path(plate_folder)
        
        # Check for Z-stack directories
        z_stack_dirs = ImageLocator.find_z_stack_dirs(plate_folder, timepoint_dir_name)
        if z_stack_dirs:
            # Check if Z-stack directories are in timepoint directory
            timepoint_dir = ImageLocator.find_timepoint_dir(plate_folder, timepoint_dir_name)
            if timepoint_dir and any(z_dir.parent == timepoint_dir for _, z_dir in z_stack_dirs):
                return 'timepoint_z_stack'
            else:
                return 'z_stack'
                
        # Check for timepoint directory
        timepoint_dir = ImageLocator.find_timepoint_dir(plate_folder, timepoint_dir_name)
        if timepoint_dir:
            # Check if timepoint directory is in Images directory
            if timepoint_dir.parent.name == "Images":
                return 'images_timepoint'
            else:
                return 'timepoint'
                
        # Check for Images directory
        images_dir = plate_folder / "Images"
        if images_dir.exists() and images_dir.is_dir() and any(images_dir.glob('*')):
            return 'images'
            
        # Check for images directly in plate folder
        if any(plate_folder.glob('*.tif')) or any(plate_folder.glob('*.tiff')):
            return 'flat'
            
        return 'unknown'
