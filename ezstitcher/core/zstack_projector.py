"""
Z-stack projection module for ezstitcher.

This module provides a class for creating projections from Z-stack images.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Callable

from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.filename_parser import detect_parser, ImageXpressFilenameParser

logger = logging.getLogger(__name__)


class ZStackProjector:
    """
    Handles Z-stack projection operations:
    - Creating maximum intensity projections
    - Creating mean intensity projections
    - Creating custom projections
    """
    def __init__(self, config, image_preprocessor=None, fs_manager=None, filename_parser=None):
        """
        Initialize the ZStackProjector.

        Args:
            config: Configuration object with Z-stack processing settings
            image_preprocessor: Image preprocessor for projection creation
            fs_manager: File system manager for file operations
            filename_parser: Parser for microscopy filenames
        """
        self.config = config
        self.fs_manager = fs_manager or FileSystemManager()
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        self.filename_parser = filename_parser
        
    def create_projections(self, input_dir, output_dir, projection_types=None, preprocessing_funcs=None):
        """Create projections from Z-stack images.

        Args:
            input_dir: Directory containing Z-stack images
            output_dir: Directory to save projections
            projection_types: List of projection types to create. If None, uses ["max"]
            preprocessing_funcs: Dictionary mapping channels to preprocessing functions.
                These functions are applied to individual images before projection creation.

        Returns:
            Tuple of (success, projections_info)
        """
        # Default to max projection if None is provided
        if projection_types is None:
            projection_types = ["max"]
            
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Auto-detect filename parser if not provided
        if self.filename_parser is None:
            # Get a sample of filenames from the input_dir
            all_files = self.fs_manager.list_image_files(input_dir)
            file_paths = [str(f) for f in all_files]
            if file_paths:
                self.filename_parser = detect_parser(file_paths)
                logger.info(f"Auto-detected parser: {self.filename_parser.__class__.__name__}")
            else:
                self.filename_parser = ImageXpressFilenameParser()
                logger.info("No files found, defaulting to ImageXpress parser")

        # Check if folder contains Z-stack images
        has_zstack, z_indices_map = self._detect_zstack_images(input_dir)

        # If no Z-stack images were detected directly, check for Z-stack folders
        if not has_zstack:
            timepoint_dir = getattr(self.config, 'timepoint_dir_name', "TimePoint_1")
            timepoint_path = input_dir / timepoint_dir

            if timepoint_path.exists():
                has_zstack_folders, z_folders = self._detect_zstack_folders(input_dir)

                if has_zstack_folders:
                    # We have Z-stack folders, so we need to detect Z-stack images in each folder
                    z_indices_map = {}

                    # Process each Z-stack folder
                    for z_index, folder in z_folders:
                        # Get all image files in this Z-stack folder
                        image_files = self.fs_manager.list_image_files(folder)

                        # Group files by their base components (without Z-index)
                        for img_file in image_files:
                            # Parse the filename using the filename parser
                            metadata = self.filename_parser.parse_filename(str(img_file))

                            if metadata:
                                # Create a base name without the Z-index
                                well = metadata['well']
                                site = metadata['site']
                                channel = metadata['channel']

                                # Create a consistent base name for grouping
                                base_name = f"{well}_s{site:03d}_w{channel}"

                                # Add this Z-index to the list for this base name
                                if base_name not in z_indices_map:
                                    z_indices_map[base_name] = []

                                if z_index not in z_indices_map[base_name]:
                                    z_indices_map[base_name].append(z_index)

                    # Sort the Z-indices for each base name
                    for base_name in z_indices_map:
                        z_indices_map[base_name].sort()

                    has_zstack = len(z_indices_map) > 0

        if not has_zstack:
            logger.warning(f"No Z-stack images found in {input_dir}")
            return False, None

        # For projections, we'll save directly to the output directory
        # This is because the output directory already includes the projection type in its name
        # (e.g., synthetic_plate_projections_max/TimePoint_1)
        projection_dirs = {}
        for proj_type in projection_types:
            # Always use the output directory directly
            proj_dir = self.fs_manager.ensure_directory(output_dir)
            projection_dirs[proj_type] = proj_dir
            logger.info(f"Saving {proj_type} projections to {proj_dir}")

        # Process each base name
        for base_name, z_indices in z_indices_map.items():
            # Load all Z-stack images for this base name
            image_stack = []
            for z_index in sorted(z_indices):
                # Parse the base_name to extract well, site, and channel
                parts = base_name.split('_')
                if len(parts) >= 3 and parts[1].startswith('s') and parts[2].startswith('w'):
                    well = parts[0]
                    site = int(parts[1][1:])
                    channel = int(parts[2][1:])

                    # Construct the filename using the filename parser
                    filename = self.filename_parser.construct_filename(well, site, channel, z_index)
                    img_path = input_dir / filename

                    # If the file doesn't exist, try alternative extensions
                    if not img_path.exists():
                        for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                            alt_filename = self.filename_parser.construct_filename(well, site, channel, z_index, extension=ext)
                            alt_path = input_dir / alt_filename
                            if alt_path.exists():
                                img_path = alt_path
                                break
                else:
                    # Fallback if we can't parse the base_name
                    img_path = input_dir / f"{base_name}_z{z_index:03d}.tif"

                img = self.fs_manager.load_image(img_path)
                if img is not None:
                    # Apply preprocessing if available for this channel
                    if preprocessing_funcs and str(channel) in preprocessing_funcs:
                        img = preprocessing_funcs[str(channel)](img)
                    image_stack.append(img)

            if not image_stack:
                logger.warning(f"No valid images found for {base_name}")
                continue

            # Convert to numpy array
            image_stack = np.array(image_stack)

            # Create and save projections
            for proj_type in projection_types:
                if proj_type == 'max':
                    projection = self.image_preprocessor.max_projection(image_stack)
                elif proj_type == 'mean':
                    projection = self.image_preprocessor.mean_projection(image_stack)
                else:
                    logger.warning(f"Unknown projection type: {proj_type}")
                    continue

                # Save projection
                output_filename = f"{base_name}.tif"
                output_path = projection_dirs[proj_type] / output_filename
                self.fs_manager.save_image(output_path, projection)
                logger.info(f"Saved {proj_type} projection for {base_name} to {output_path}")

        return True, projection_dirs
        
    def _detect_zstack_images(self, folder_path):
        """
        Detect if a folder contains Z-stack images based on filename patterns.
        
        This is a helper method that delegates to a ZStackOrganizer if available,
        or implements the detection logic directly if not.
        
        Args:
            folder_path: Path to the folder to check
            
        Returns:
            tuple: (has_zstack, z_indices_map)
        """
        # Try to import ZStackOrganizer
        try:
            from ezstitcher.core.zstack_organizer import ZStackOrganizer
            organizer = ZStackOrganizer(self.config, self.filename_parser, self.fs_manager)
            return organizer.detect_zstack_images(folder_path)
        except ImportError:
            # Implement detection logic directly
            folder_path = Path(folder_path)
            all_files = self.fs_manager.list_image_files(folder_path)
            
            # Auto-detect parser if needed
            if self.filename_parser is None:
                file_paths = [str(f) for f in all_files]
                if file_paths:
                    self.filename_parser = detect_parser(file_paths)
                else:
                    self.filename_parser = ImageXpressFilenameParser()
                    
            z_indices = {}
            
            for img_file in all_files:
                metadata = self.filename_parser.parse_filename(str(img_file))
                
                if metadata and 'z_index' in metadata and metadata['z_index'] is not None:
                    well = metadata['well']
                    site = metadata['site']
                    channel = metadata['channel']
                    
                    base_name = f"{well}_s{site:03d}_w{channel}"
                    
                    if base_name not in z_indices:
                        z_indices[base_name] = []
                        
                    z_indices[base_name].append(metadata['z_index'])
            
            # Sort z_indices
            for base_name in z_indices:
                z_indices[base_name].sort()
                
            has_zstack = len(z_indices) > 0
            
            return has_zstack, z_indices
            
    def _detect_zstack_folders(self, plate_folder):
        """
        Detect Z-stack folders in a plate folder.
        
        This is a helper method that delegates to a ZStackOrganizer if available,
        or implements the detection logic directly if not.
        
        Args:
            plate_folder: Path to the plate folder
            
        Returns:
            tuple: (has_zstack, z_folders)
        """
        # Try to import ZStackOrganizer
        try:
            from ezstitcher.core.zstack_organizer import ZStackOrganizer
            organizer = ZStackOrganizer(self.config, self.filename_parser, self.fs_manager)
            return organizer.detect_zstack_folders(plate_folder)
        except ImportError:
            # Implement detection logic directly
            plate_path = Path(plate_folder)
            timepoint_dir = getattr(self.config, 'timepoint_dir_name', "TimePoint_1")
            timepoint_path = plate_path / timepoint_dir
            
            if not timepoint_path.exists():
                return False, []
                
            z_pattern = re.compile(r'ZStep_(\d+)')
            z_folders = []
            
            for item in timepoint_path.iterdir():
                if item.is_dir():
                    match = z_pattern.match(item.name)
                    if match:
                        z_index = int(match.group(1))
                        z_folders.append((z_index, item))
                        
            z_folders.sort(key=lambda x: x[0])
            
            has_zstack = len(z_folders) > 0
            
            return has_zstack, z_folders
