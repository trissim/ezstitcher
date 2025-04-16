import re
import os
import csv
import shutil
import logging
import numpy as np
import inspect
from pathlib import Path
from collections import defaultdict
from typing import List, Callable, Union, Optional, Any
from ezstitcher.core.config import ZStackProcessorConfig, FocusAnalyzerConfig
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.filename_parser import ImageXpressFilenameParser
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_locator import ImageLocator

logger = logging.getLogger(__name__)

class ZStackProcessor:
    """
    Handles Z-stack specific operations:
    - Detection
    - Projection creation
    - Best focus selection
    - Per-plane stitching
    """
    def __init__(self, config: ZStackProcessorConfig, filename_parser=None, preprocessing_funcs=None):
        self.config = config
        self.fs_manager = FileSystemManager()
        self._z_info = None
        self._z_indices = []
        self.preprocessing_funcs = preprocessing_funcs or {}

        # Use focus_config from config if available, otherwise create one with the focus_method
        focus_config = config.focus_config or FocusAnalyzerConfig(method=config.focus_method)
        self.focus_analyzer = FocusAnalyzer(focus_config)

        # Initialize the filename parser
        if filename_parser is None:
            self.filename_parser = ImageXpressFilenameParser()
        else:
            self.filename_parser = filename_parser

        # Initialize the image preprocessor
        self.image_preprocessor = ImagePreprocessor()

    def _adapt_function(self, func: Callable) -> Callable:
        """
        Adapt a function to the reference function interface.

        This allows both:
        - Functions that take a single image and return a processed image
        - Functions that take a stack and return a single image

        to be used as reference functions.

        Args:
            func: The function to adapt

        Returns:
            A function that takes a stack and returns a single image
        """
        # Try to determine if the function works on stacks or single images
        try:
            # Create a small test stack (2 tiny images)
            test_stack = [np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)]

            # Try calling the function with the stack
            result = func(test_stack)

            # If it returns a single image, it's a stack function
            if isinstance(result, np.ndarray) and result.ndim == 2:
                logger.debug(f"Function {func.__name__} detected as stack function")
                return func

            # If it returns something else, we can't use it directly
            raise ValueError(f"Function {func.__name__} doesn't return a 2D image when given a stack")

        except Exception:
            # If it fails, try with a single image
            try:
                # Try with a single image
                result = func(test_stack[0])

                # If it returns an image, it's an image function
                if isinstance(result, np.ndarray) and result.ndim == 2:
                    logger.debug(f"Function {func.__name__} detected as image function")

                    def adapter(stack):
                        # Apply the function to each image in the stack
                        processed_stack = [func(img) for img in stack]
                        # Return the max projection of the processed stack
                        return np.max(np.array(processed_stack), axis=0)

                    # Copy metadata from original function
                    adapter.__name__ = f"{func.__name__}_adapted"
                    adapter.__doc__ = f"Adapted version of {func.__name__} that works on stacks."
                    return adapter

                # If it returns something else, we can't use it
                raise ValueError(f"Function {func.__name__} doesn't return a 2D image when given a single image")

            except Exception as e:
                # If both attempts fail, raise an error
                raise ValueError(f"Cannot adapt function {func.__name__}: {str(e)}")

    def detect_z_stacks(self, plate_folder: str):
        has_zstack, self._z_info = self.preprocess_plate_folder(plate_folder)

        if has_zstack and 'z_indices_map' in self._z_info and self._z_info['z_indices_map']:
            all_z_indices = set()
            for base_name, indices in self._z_info['z_indices_map'].items():
                all_z_indices.update(indices)
            self._z_indices = sorted(list(all_z_indices))

        return has_zstack

    def preprocess_plate_folder(self, plate_folder):
        plate_folder = ImageLocator.find_image_directory(Path(plate_folder))

        has_zstack_folders, z_folders = self.detect_zstack_folders(plate_folder)

        if has_zstack_folders:
            logger.info(f"Organizing Z-stack folders in {plate_folder}")
            self.organize_zstack_folders(plate_folder)

        has_zstack_images, z_indices_map = self.detect_zstack_images(plate_folder)


        has_zstack = has_zstack_folders or has_zstack_images

        z_info = {
            'has_zstack_folders': has_zstack_folders,
            'z_folders': z_folders,
            'has_zstack_images': has_zstack_images,
            'z_indices_map': z_indices_map
        }

        if has_zstack:
            logger.info(f"Z-stack detected in {plate_folder}")
        else:
            logger.info(f"No Z-stack detected in {plate_folder}")
            z_indices_map = {}

        return has_zstack, z_info

    def detect_zstack_folders(self, plate_folder, pattern=None):
        """
        Detect Z-stack folders in a plate folder.

        Args:
            plate_folder (str or Path): Path to the plate folder
            pattern (str or Pattern, optional): Regex pattern to match Z-stack folders

        Returns:
            tuple: (has_zstack, z_folders) where z_folders is a list of (z_index, folder_path) tuples
        """
        plate_path = ImageLocator.find_image_directory(Path(plate_folder))

        # Use ImageLocator to find Z-stack directories
        z_folders = ImageLocator.find_z_stack_dirs(
            plate_path,
            pattern=pattern or r'ZStep_\d+',
            recursive=False  # Only look in the immediate directory
        )

        # Store Z-indices for later use
        if z_folders:
            self._z_indices = [z[0] for z in z_folders]

        return bool(z_folders), z_folders

    def organize_zstack_folders(self, plate_folder):
        """
        Organize Z-stack folders by moving files to the plate folder with proper naming.

        Args:
            plate_folder (str or Path): Path to the plate folder

        Returns:
            bool: True if Z-stack was organized, False otherwise
        """
        has_zstack_folders, z_folders = self.detect_zstack_folders(plate_folder)
        if not has_zstack_folders:
            return False

        plate_path = Path(plate_folder)
        plate_path = ImageLocator.find_image_directory(plate_folder)

        # Process each Z-stack folder
        for z_index, z_folder in z_folders:
            # Get all image files in this folder
            image_files = self.fs_manager.list_image_files(z_folder)

            for img_file in image_files:
                # Parse the filename
                metadata = self.filename_parser.parse_filename(str(img_file))
                if not metadata:
                    continue

                # Construct new filename with Z-index
                new_name = self.filename_parser.construct_filename(
                    well=metadata['well'],
                    site=metadata['site'],
                    channel=metadata['channel'],
                    z_index=z_index,
                    extension=metadata['extension']
                )

                # Copy file to plate folder
                new_path = plate_path / new_name
                self.fs_manager.copy_file(img_file, new_path)

        # Remove Z-stack folders
        for _, z_folder in z_folders:
            self.fs_manager.remove_directory(z_folder)

        return True

    def get_z_indices(self):
        """
        Return the list of detected Z indices after calling detect_z_stacks().
        """
        return getattr(self, '_z_indices', [])


    def detect_zstack_images(self, folder_path):
        """
        Detect if a folder contains Z-stack images based on filename patterns.

        Args:
            folder_path (str or Path): Path to the folder

        Returns:
            tuple: (has_zstack, z_indices_map) where z_indices_map is a dict mapping base filenames to Z-indices
        """
        folder_path = Path(folder_path)
        all_files = self.fs_manager.list_image_files(folder_path)

        # Group files by their base components and collect z-indices
        z_indices = defaultdict(list)
        for img_file in all_files:
            metadata = self.filename_parser.parse_filename(str(img_file))
            if metadata and 'z_index' in metadata and metadata['z_index'] is not None:
                well = metadata['well']
                site = metadata['site']
                channel = metadata['channel']
                base_key = f"{well}_s{site:03d}_w{channel}"
                z_indices[base_key].append(metadata['z_index'])

        # Sort z-indices for each base name
        for indices in z_indices.values():
            indices.sort()

        has_zstack = bool(z_indices)
        return has_zstack, dict(z_indices)

    def load_z_stacks(self, input_dir, z_indices_map, filter_func=None, wells=None, sites=None, channels=None):
        """
        Load Z-stack images into memory with optional filtering.

        Args:
            input_dir (str or Path): Directory containing Z-stack images
            z_indices_map (dict): Dictionary mapping base names to Z-indices
            filter_func (callable, optional): Custom filter function that takes base_name and metadata
                                             and returns True if the stack should be loaded
            wells (list, optional): List of well IDs to include (e.g., ['A01', 'B02'])
            sites (list, optional): List of site numbers to include
            channels (list, optional): List of channel numbers to include

        Returns:
            dict: Dictionary mapping base names to loaded image stacks (as numpy arrays)
        """
        input_dir = Path(input_dir)
        loaded_stacks = {}

        # Process each base name
        for base_name, z_indices in z_indices_map.items():
            # Extract metadata from the base_name pattern
            metadata = self.filename_parser.parse_filename(base_name)

            if not metadata:
                logger.warning(f"Could not parse base_name: {base_name}, skipping")
                continue

            well = metadata['well']
            site = metadata['site']
            channel = metadata['channel']

            # Apply filters if provided
            if (wells and well not in wells) or \
               (sites and site not in sites) or \
               (channels and channel not in channels):
                continue

            # Load all Z-stack images for this base name
            image_stack = []
            for z_index in sorted(z_indices):
                # Use the filename parser to construct the base filename with the correct z-index
                filename = self.filename_parser.construct_filename(
                    well=well,
                    site=site,
                    channel=channel,
                    z_index=z_index
                )

                # Create a pattern that matches the exact filename (without extension)
                base_filename = Path(filename).stem
                pattern = f"^{re.escape(base_filename)}\\."

                # Use ImageLocator to find the file with any supported extension
                matching_files = ImageLocator.find_images_by_pattern(input_dir, pattern)

                if matching_files:
                    # Use the first matching file
                    img_path = matching_files[0]
                    img = self.fs_manager.load_image(img_path)
                    if img is not None:
                        image_stack.append(img)
                else:
                    logger.debug(f"Could not find image for {well}, site {site}, channel {channel}, z-index {z_index}")

            if image_stack:
                # Convert to numpy array
                loaded_stacks[base_name] = np.array(image_stack)
            else:
                logger.warning(f"No valid images found for {base_name}")

        return loaded_stacks

    def create_zstack_projections(self, input_dir, output_dir, projection_types=None):
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

        # Check if folder contains Z-stack images
        has_zstack, z_indices_map = self.detect_zstack_images(input_dir)

        if not has_zstack:
            logger.warning(f"No Z-stack images found in {input_dir}")
            return False, None

        # For projections, we'll save directly to the output directory
        projection_dirs = {}
        for proj_type in projection_types:
            proj_dir = self.fs_manager.ensure_directory(output_dir)
            projection_dirs[proj_type] = proj_dir
            logger.info(f"Saving {proj_type} projections to {proj_dir}")

        # Load all Z-stacks into memory
        loaded_stacks = self.load_z_stacks(input_dir, z_indices_map)

        # Create and save projections for each stack
        for base_name, image_stack in loaded_stacks.items():
            # Create and save projections
            for proj_type in projection_types:
                if proj_type == 'max':
                    projection = self.image_preprocessor.max_projection(image_stack)
                elif proj_type == 'mean':
                    projection = self.image_preprocessor.mean_projection(image_stack)
                else:
                    logger.warning(f"Unknown projection type: {proj_type}")
                    continue

                # Save projection with the standard .tif extension
                output_filename = f"{base_name}.tif"
                output_path = projection_dirs[proj_type] / output_filename
                self.fs_manager.save_image(output_path, projection)
                logger.info(f"Saved {proj_type} projection for {base_name} to {output_path}")

        return True, projection_dirs

    def process_stack(self, base_name, stack, output_dir):
        """
        Process a z-stack or single image uniformly.
        
        Args:
            base_name (str): Base name for output files (without extension)
            stack (list): List of images (numpy arrays)
            output_dir (Path): Directory to save results
            
        Returns:
            list: List of output file paths
        """
        # Handle single images the same way as stacks - just apply projection
        # This simplifies the code and ensures consistent behavior
        if self.config.projection_method != "none":
            # Apply projection (for single images, this just returns the image)
            if len(stack) == 1:
                result = stack[0]  # No need for projection with single image
            else:
                result = self.create_projection(stack, self.config.projection_method)

            # Use z001 suffix for projections
            # This ensures consistency with per-plane files
            filename = f"{base_name}.tif"
            output_path = output_dir / filename
            self.fs_manager.save_image(output_path, result)
            return [output_path]
        else:
            # Save each z-plane with proper z-suffix
            output_paths = []
            for i, z_index in enumerate(range(1, len(stack) + 1)):
                # Use filename parser to add z-suffix
                filename = f"{base_name}_z{z_index:03d}.tif"
                output_path = output_dir / filename
                self.fs_manager.save_image(output_path, stack[i])
                output_paths.append(output_path)
        return output_paths

    def create_projection(self, stack, method="max"):
        """
        Create a projection from a stack using the specified method.
        
        Args:
            stack (list): List of images
            method (str): Projection method (max, mean, best_focus)
            
        Returns:
            numpy.ndarray: Projected image
        """
        if method == "max":
            return self.image_preprocessor.max_projection(stack)
        elif method == "mean":
            return self.image_preprocessor.mean_projection(stack)
        elif method == "best_focus":
            best_idx, _ = self.focus_analyzer.find_best_focus(stack)
            return stack[best_idx]
        else:
            logger.warning(f"Unknown projection method: {method}, using max")
        return self.image_preprocessor.max_projection(stack)
