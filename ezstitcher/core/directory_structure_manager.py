"""
Directory structure manager module for ezstitcher.

This module provides a class for managing different directory structures.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set

from ezstitcher.core.image_locator import ImageLocator
from ezstitcher.core.filename_parser import FilenameParser, ImageXpressFilenameParser, detect_parser

logger = logging.getLogger(__name__)


class DirectoryStructureManager:
    """
    Manages different directory structures for microscopy data.
    
    Supports various directory structures:
    - Images directly in the plate folder
    - Images in a TimePoint_1 subfolder
    - Images in Z-stack folders in the plate folder
    - Images in Z-stack folders in the TimePoint_1 subfolder
    - Images in an Images subfolder
    - Images in an Images/TimePoint_1 subfolder
    """
    
    def __init__(self, plate_folder: Union[str, Path], 
                filename_parser: Optional[FilenameParser] = None,
                timepoint_dir_name: str = "TimePoint_1"):
        """
        Initialize the DirectoryStructureManager.
        
        Args:
            plate_folder: Path to the plate folder
            filename_parser: Parser for microscopy filenames
            timepoint_dir_name: Name of the timepoint directory
        """
        self.plate_folder = Path(plate_folder)
        self.timepoint_dir_name = timepoint_dir_name
        self.structure_type = None
        self.image_locations = {}
        
        # Initialize filename parser
        if filename_parser is None:
            # Auto-detect parser from image files
            image_files = self._find_sample_image_files()
            if image_files:
                self.filename_parser = detect_parser([str(f) for f in image_files])
                logger.info(f"Auto-detected parser: {self.filename_parser.__class__.__name__}")
            else:
                self.filename_parser = ImageXpressFilenameParser()
                logger.info("No image files found, defaulting to ImageXpress parser")
        else:
            self.filename_parser = filename_parser
            
        # Detect directory structure
        self._detect_structure()
        
    def _find_sample_image_files(self, max_samples: int = 10) -> List[Path]:
        """
        Find a sample of image files for parser detection.
        
        Args:
            max_samples: Maximum number of sample files to return
            
        Returns:
            List of Path objects for image files
        """
        # Try to find images in various locations
        image_locations = ImageLocator.find_image_locations(self.plate_folder, self.timepoint_dir_name)
        
        # Collect sample files from all locations
        sample_files = []
        
        # Check plate folder
        if 'plate' in image_locations:
            sample_files.extend(image_locations['plate'][:max_samples])
            
        # Check timepoint directory
        if 'timepoint' in image_locations and len(sample_files) < max_samples:
            remaining = max_samples - len(sample_files)
            sample_files.extend(image_locations['timepoint'][:remaining])
            
        # Check Images directory
        if 'images' in image_locations and len(sample_files) < max_samples:
            remaining = max_samples - len(sample_files)
            sample_files.extend(image_locations['images'][:remaining])
            
        # Check Z-stack directories
        if 'z_stack' in image_locations and len(sample_files) < max_samples:
            remaining = max_samples - len(sample_files)
            # Flatten the z_stack images
            z_stack_images = []
            for z_index, images in image_locations['z_stack'].items():
                z_stack_images.extend(images)
            sample_files.extend(z_stack_images[:remaining])
            
        return sample_files[:max_samples]
        
    def _detect_structure(self):
        """
        Detect the directory structure and catalog image locations.
        """
        # Detect directory structure
        self.structure_type = ImageLocator.detect_directory_structure(
            self.plate_folder, self.timepoint_dir_name)
        logger.info(f"Detected directory structure: {self.structure_type}")
        
        # Find image locations
        self.image_locations = ImageLocator.find_image_locations(
            self.plate_folder, self.timepoint_dir_name)
        
        # Log summary of image locations
        for location_type, images in self.image_locations.items():
            if location_type == 'z_stack':
                total_images = sum(len(imgs) for imgs in images.values())
                logger.info(f"Found {total_images} images in {len(images)} Z-stack directories")
            else:
                logger.info(f"Found {len(images)} images in {location_type} location")
                
    def get_image_path(self, well: str, site: int, channel: int, z_index: Optional[int] = None) -> Optional[Path]:
        """
        Get the path to an image based on its metadata.
        
        Args:
            well: Well ID (e.g., 'A01' or 'R01C01')
            site: Site number
            channel: Channel number
            z_index: Z-index (optional)
            
        Returns:
            Path to the image if found, None otherwise
        """
        # If z_index is provided, check Z-stack directories first
        if z_index is not None and 'z_stack' in self.image_locations:
            if z_index in self.image_locations['z_stack']:
                # Check each image in this Z-stack directory
                for img_path in self.image_locations['z_stack'][z_index]:
                    metadata = self.filename_parser.parse_filename(str(img_path))
                    if metadata and metadata.get('well') == well and \
                       metadata.get('site') == site and metadata.get('channel') == channel:
                        return img_path
        
        # Try to construct the filename using the parser
        try:
            # For Z-stack images
            if z_index is not None:
                filename = self.filename_parser.construct_filename(well, site, channel, z_index)
            else:
                # For non-Z-stack images
                filename = self.filename_parser.construct_filename(well, site, channel)
                
            # Check each location for this filename
            for location_type, images in self.image_locations.items():
                if location_type == 'z_stack':
                    # Skip Z-stack location if we're not looking for a Z-stack image
                    if z_index is None:
                        continue
                    
                    # Check each Z-stack directory
                    for z_idx, z_images in images.items():
                        if z_idx == z_index:
                            for img_path in z_images:
                                if img_path.name == filename:
                                    return img_path
                else:
                    # Check regular locations
                    for img_path in images:
                        if img_path.name == filename:
                            return img_path
                            
            # If not found, try alternative extensions
            for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                alt_filename = os.path.splitext(filename)[0] + ext
                
                for location_type, images in self.image_locations.items():
                    if location_type == 'z_stack':
                        if z_index is None:
                            continue
                        
                        for z_idx, z_images in images.items():
                            if z_idx == z_index:
                                for img_path in z_images:
                                    if img_path.name == alt_filename:
                                        return img_path
                    else:
                        for img_path in images:
                            if img_path.name == alt_filename:
                                return img_path
        except Exception as e:
            logger.error(f"Error constructing filename: {e}")
            
        # If not found, try parsing all filenames
        for location_type, images in self.image_locations.items():
            if location_type == 'z_stack':
                # Skip Z-stack location if we're not looking for a Z-stack image
                if z_index is None:
                    continue
                
                # Check each Z-stack directory
                for z_idx, z_images in images.items():
                    if z_index is None or z_idx == z_index:
                        for img_path in z_images:
                            metadata = self.filename_parser.parse_filename(str(img_path))
                            if metadata and metadata.get('well') == well and \
                               metadata.get('site') == site and metadata.get('channel') == channel and \
                               (z_index is None or metadata.get('z_index') == z_index):
                                return img_path
            else:
                # Check regular locations
                for img_path in images:
                    metadata = self.filename_parser.parse_filename(str(img_path))
                    if metadata and metadata.get('well') == well and \
                       metadata.get('site') == site and metadata.get('channel') == channel and \
                       (z_index is None or metadata.get('z_index') == z_index):
                        return img_path
                        
        logger.warning(f"Image not found: well={well}, site={site}, channel={channel}, z_index={z_index}")
        return None
        
    def list_images(self, well: Optional[str] = None, site: Optional[int] = None, 
                   channel: Optional[int] = None, z_index: Optional[int] = None) -> List[Path]:
        """
        List images matching the specified criteria.
        
        Args:
            well: Well ID (optional)
            site: Site number (optional)
            channel: Channel number (optional)
            z_index: Z-index (optional)
            
        Returns:
            List of Path objects for matching images
        """
        matching_images = []
        
        # Check each location
        for location_type, images in self.image_locations.items():
            if location_type == 'z_stack':
                # Check each Z-stack directory
                for z_idx, z_images in images.items():
                    if z_index is None or z_idx == z_index:
                        for img_path in z_images:
                            metadata = self.filename_parser.parse_filename(str(img_path))
                            if metadata and \
                               (well is None or metadata.get('well') == well) and \
                               (site is None or metadata.get('site') == site) and \
                               (channel is None or metadata.get('channel') == channel):
                                matching_images.append(img_path)
            else:
                # Check regular locations
                for img_path in images:
                    metadata = self.filename_parser.parse_filename(str(img_path))
                    if metadata and \
                       (well is None or metadata.get('well') == well) and \
                       (site is None or metadata.get('site') == site) and \
                       (channel is None or metadata.get('channel') == channel) and \
                       (z_index is None or metadata.get('z_index') == z_index):
                        matching_images.append(img_path)
                        
        return sorted(matching_images)
        
    def get_timepoint_dir(self) -> Optional[Path]:
        """
        Get the path to the TimePoint directory if it exists.
        
        Returns:
            Path to the TimePoint directory if found, None otherwise
        """
        return ImageLocator.find_timepoint_dir(self.plate_folder, self.timepoint_dir_name)
        
    def get_z_stack_dirs(self) -> List[Tuple[int, Path]]:
        """
        Get the paths to Z-stack directories if they exist.
        
        Returns:
            List of (z_index, directory) tuples
        """
        return ImageLocator.find_z_stack_dirs(self.plate_folder, self.timepoint_dir_name)
        
    def get_wells(self) -> List[str]:
        """
        Get a list of all wells in the plate.
        
        Returns:
            List of well IDs
        """
        wells = set()
        
        # Check each location
        for location_type, images in self.image_locations.items():
            if location_type == 'z_stack':
                # Check each Z-stack directory
                for z_idx, z_images in images.items():
                    for img_path in z_images:
                        metadata = self.filename_parser.parse_filename(str(img_path))
                        if metadata and 'well' in metadata:
                            wells.add(metadata['well'])
            else:
                # Check regular locations
                for img_path in images:
                    metadata = self.filename_parser.parse_filename(str(img_path))
                    if metadata and 'well' in metadata:
                        wells.add(metadata['well'])
                        
        return sorted(list(wells))
        
    def get_sites(self, well: Optional[str] = None) -> List[int]:
        """
        Get a list of all sites in the plate or a specific well.
        
        Args:
            well: Well ID (optional)
            
        Returns:
            List of site numbers
        """
        sites = set()
        
        # Check each location
        for location_type, images in self.image_locations.items():
            if location_type == 'z_stack':
                # Check each Z-stack directory
                for z_idx, z_images in images.items():
                    for img_path in z_images:
                        metadata = self.filename_parser.parse_filename(str(img_path))
                        if metadata and 'site' in metadata and \
                           (well is None or metadata.get('well') == well):
                            sites.add(metadata['site'])
            else:
                # Check regular locations
                for img_path in images:
                    metadata = self.filename_parser.parse_filename(str(img_path))
                    if metadata and 'site' in metadata and \
                       (well is None or metadata.get('well') == well):
                        sites.add(metadata['site'])
                        
        return sorted(list(sites))
        
    def get_channels(self, well: Optional[str] = None, site: Optional[int] = None) -> List[int]:
        """
        Get a list of all channels in the plate, a specific well, or a specific site.
        
        Args:
            well: Well ID (optional)
            site: Site number (optional)
            
        Returns:
            List of channel numbers
        """
        channels = set()
        
        # Check each location
        for location_type, images in self.image_locations.items():
            if location_type == 'z_stack':
                # Check each Z-stack directory
                for z_idx, z_images in images.items():
                    for img_path in z_images:
                        metadata = self.filename_parser.parse_filename(str(img_path))
                        if metadata and 'channel' in metadata and \
                           (well is None or metadata.get('well') == well) and \
                           (site is None or metadata.get('site') == site):
                            channels.add(metadata['channel'])
            else:
                # Check regular locations
                for img_path in images:
                    metadata = self.filename_parser.parse_filename(str(img_path))
                    if metadata and 'channel' in metadata and \
                       (well is None or metadata.get('well') == well) and \
                       (site is None or metadata.get('site') == site):
                        channels.add(metadata['channel'])
                        
        return sorted(list(channels))
        
    def get_z_indices(self, well: Optional[str] = None, site: Optional[int] = None, 
                     channel: Optional[int] = None) -> List[int]:
        """
        Get a list of all Z-indices in the plate, a specific well, site, or channel.
        
        Args:
            well: Well ID (optional)
            site: Site number (optional)
            channel: Channel number (optional)
            
        Returns:
            List of Z-indices
        """
        z_indices = set()
        
        # Check Z-stack directories
        if 'z_stack' in self.image_locations:
            for z_idx, z_images in self.image_locations['z_stack'].items():
                z_indices.add(z_idx)
                
        # Check for Z-indices in filenames
        for location_type, images in self.image_locations.items():
            if location_type == 'z_stack':
                continue  # Already handled above
                
            for img_path in images:
                metadata = self.filename_parser.parse_filename(str(img_path))
                if metadata and 'z_index' in metadata and metadata['z_index'] is not None and \
                   (well is None or metadata.get('well') == well) and \
                   (site is None or metadata.get('site') == site) and \
                   (channel is None or metadata.get('channel') == channel):
                    z_indices.add(metadata['z_index'])
                    
        return sorted(list(z_indices))
