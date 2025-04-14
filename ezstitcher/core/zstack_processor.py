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
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.focus_analyzer import FocusAnalyzer

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

        # Initialize the focus analyzer
        from ezstitcher.core.focus_analyzer import FocusAnalyzer
        # Use focus_config from config if available, otherwise create one with the focus_method
        focus_config = config.focus_config or FocusAnalyzerConfig(method=config.focus_method)
        self.focus_analyzer = FocusAnalyzer(focus_config)

        # Initialize the filename parser
        if filename_parser is None:
            # Import here to avoid circular imports
            from ezstitcher.core.filename_parser import ImageXpressFilenameParser
            self.filename_parser = ImageXpressFilenameParser()
        else:
            self.filename_parser = filename_parser

        # Initialize the image preprocessor
        from ezstitcher.core.image_preprocessor import ImagePreprocessor
        self.image_preprocessor = ImagePreprocessor()

        # Initialize the reference function
        self._reference_function = self._create_reference_function(config.z_reference_function)

    def _create_reference_function(self, func_or_name: Union[str, Callable]) -> Callable:
        """
        Create a reference function from a string name or callable.

        This function adapts various types of functions to the reference function interface,
        which takes a Z-stack (list of images) and returns a single 2D image.

        Args:
            func_or_name: String name of a standard function or a callable
                Standard names: "max_projection", "mean_projection", "best_focus"
                Can also be a custom function that takes a Z-stack and returns a 2D image

        Returns:
            A function that takes a stack and returns a single image
        """
        # Handle string names
        if isinstance(func_or_name, str):
            if func_or_name == "max_projection":
                return self._adapt_function(self.image_preprocessor.max_projection)
            elif func_or_name == "mean_projection":
                return self._adapt_function(self.image_preprocessor.mean_projection)
            elif func_or_name == "best_focus":
                return lambda stack: self.focus_analyzer.select_best_focus(stack, method=self.config.focus_method)[0]
            else:
                raise ValueError(f"Unknown reference function name: {func_or_name}")

        # Handle callables
        if callable(func_or_name):
            return self._adapt_function(func_or_name)

        raise ValueError(f"Reference function must be a string or callable, got {type(func_or_name)}")

    def _preprocess_stack(self, stack, channel):
        """
        Apply preprocessing to each image in a Z-stack.

        Args:
            stack: List of images in the Z-stack
            channel: Channel identifier for selecting the preprocessing function

        Returns:
            List of preprocessed images
        """
        if channel in self.preprocessing_funcs:
            func = self.preprocessing_funcs[channel]
            return [func(img) for img in stack]
        return stack

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
        plate_path = Path(plate_folder)

        has_zstack_folders, z_folders = self.detect_zstack_folders(plate_folder)

        if has_zstack_folders:
            logger.info(f"Organizing Z-stack folders in {plate_folder}")
            self.organize_zstack_folders(plate_folder)

        timepoint_path = plate_path / "TimePoint_1"
        if timepoint_path.exists():
            has_zstack_images, z_indices_map = self.detect_zstack_images(timepoint_path)
        else:
            has_zstack_images = False
            z_indices_map = {}

        has_zstack = has_zstack_folders or has_zstack_images

        if has_zstack:
            logger.info(f"Z-stack detected in {plate_folder}")
        else:
            logger.info(f"No Z-stack detected in {plate_folder}")

        z_info = {
            'has_zstack_folders': has_zstack_folders,
            'z_folders': z_folders,
            'has_zstack_images': has_zstack_images,
            'z_indices_map': z_indices_map
        }

        return has_zstack, z_info

    def detect_zstack_folders(self, plate_folder):
        logger.debug(f"Called detect_zstack_folders with plate_folder={plate_folder}")

        try:
            plate_path = Path(plate_folder)
            timepoint_dir = self.config.timepoint_dir_name if hasattr(self.config, 'timepoint_dir_name') else "TimePoint_1"
            timepoint_path = plate_path / timepoint_dir

            if not timepoint_path.exists():
                logger.error(f"{timepoint_dir} folder does not exist in {plate_folder}")
                logger.debug("Returning (False, []) due to missing TimePoint_1")
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
            if has_zstack:
                logger.info(f"Found {len(z_folders)} Z-stack folders in {plate_folder}")
                for z_index, folder in z_folders[:3]:  # Log first 3 for brevity
                    logger.info(f"Z-stack folder: {folder.name}, Z-index: {z_index}")

                # Store the Z-indices for later use
                self._z_indices = [z[0] for z in z_folders]
            else:
                logger.info(f"No Z-stack folders found in {plate_folder}")

            logger.debug(f"Returning ({has_zstack}, z_folders with {len(z_folders)} items)")
            return has_zstack, z_folders

        except Exception as e:
            logger.error(f"Exception in detect_zstack_folders: {e}", exc_info=True)
            raise


    def organize_zstack_folders(self, plate_folder):
        """
        Organize Z-stack folders by moving files to TimePoint_1 with proper naming.

        Args:
            plate_folder (str or Path): Path to the plate folder

        Returns:
            bool: True if Z-stack was organized, False otherwise
        """
        has_zstack, z_folders = self.detect_zstack_folders(plate_folder)

        if not has_zstack:
            return False

        plate_path = Path(plate_folder)
        timepoint_dir = self.config.timepoint_dir_name if hasattr(self.config, 'timepoint_dir_name') else "TimePoint_1"
        timepoint_path = plate_path / timepoint_dir
        self.fs_manager.ensure_directory(timepoint_path)

        for z_index, z_folder in z_folders:
            logger.info(f"Processing Z-stack folder: {z_folder.name}")

            # Use FileSystemManager to list image files
            image_files = self.fs_manager.list_image_files(z_folder)

            for img_file in image_files:
                # Use the instance's filename parser to parse the filename
                if not hasattr(self, 'filename_parser') or self.filename_parser is None:
                    # Import here to avoid circular imports
                    from ezstitcher.core.filename_parser import detect_parser
                    self.filename_parser = detect_parser([str(img_file)])
                    logger.info(f"Auto-detected parser: {self.filename_parser.__class__.__name__}")

                metadata = self.filename_parser.parse_filename(str(img_file))

                if metadata:
                    well = metadata['well']
                    site = str(metadata['site']).zfill(3)
                    wavelength = str(metadata['channel'])
                    extension = os.path.splitext(img_file.name)[1]

                    # Use the filename parser's construct_filename method for all microscope types
                    # This is the proper OOP approach - let the parser handle the format-specific details
                    new_name = self.filename_parser.construct_filename(well, int(site), int(wavelength), z_index, extension=extension)

                    new_path = timepoint_path / new_name

                    self.fs_manager.copy_file(img_file, new_path)
                    logger.info(f"Copied {img_file.name} to {new_path.name}")
                else:
                    logger.warning(f"Could not parse filename: {img_file.name}")

        for z_index, z_folder in z_folders:
            if self.fs_manager.remove_directory(z_folder):
                logger.info(f"Removed Z-stack folder: {z_folder}")
            else:
                logger.warning(f"Failed to remove Z-stack folder {z_folder}")

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

        # Use FileSystemManager to list image files
        all_files = self.fs_manager.list_image_files(folder_path)

        # Check if we need to auto-detect the parser
        if not hasattr(self, 'filename_parser') or self.filename_parser is None:
            # Import here to avoid circular imports
            from ezstitcher.core.filename_parser import detect_parser
            file_paths = [str(f) for f in all_files]
            if file_paths:
                self.filename_parser = detect_parser(file_paths)
                logger.info(f"Auto-detected parser: {self.filename_parser.__class__.__name__}")
            else:
                from ezstitcher.core.filename_parser import ImageXpressFilenameParser
                self.filename_parser = ImageXpressFilenameParser()
                logger.info("No files found, defaulting to ImageXpress parser")

        from collections import defaultdict
        z_indices = defaultdict(list)

        # Group files by their base components (well, site, channel)
        for img_file in all_files:
            # Parse the filename using the appropriate parser
            metadata = self.filename_parser.parse_filename(str(img_file))

            if metadata and 'z_index' in metadata and metadata['z_index'] is not None:
                # Create a base name without the Z-index
                well = metadata['well']
                site = metadata['site']
                channel = metadata['channel']

                # Create a consistent base name for grouping
                base_name = f"{well}_s{site:03d}_w{channel}"

                # Add the Z-index to the list for this base name
                z_indices[base_name].append(metadata['z_index'])
                logger.debug(f"Matched z-index: {img_file.name} -> base:{base_name}, z:{metadata['z_index']}")
            else:
                logger.debug(f"No z-index found for file: {img_file.name}")

        has_zstack = len(z_indices) > 0
        if has_zstack:
            for base_name in z_indices:
                z_indices[base_name].sort()

            logger.info(f"Found Z-stack images in {folder_path}")
            logger.info(f"Detected {len(z_indices)} unique image stacks")

            for i, (base_name, indices) in enumerate(list(z_indices.items())[:3]):
                logger.info(f"Example {i+1}: {base_name} has {len(indices)} z-planes: {indices}")
        else:
            logger.info(f"No Z-stack images detected in {folder_path}")

        return has_zstack, dict(z_indices)


    def create_zstack_projections(self, input_dir, output_dir, projection_types=None, preprocessing_funcs=None):
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
        # Use provided preprocessing functions or the ones from initialization
        preprocessing_funcs = preprocessing_funcs or self.preprocessing_funcs
        # Default to max projection if None is provided
        if projection_types is None:
            projection_types = ["max"]
        import numpy as np
        from pathlib import Path

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Check if folder contains Z-stack images
        has_zstack, z_indices_map = self.detect_zstack_images(input_dir)

        # If no Z-stack images were detected directly, check for Z-stack folders
        if not has_zstack:
            timepoint_dir = self.config.timepoint_dir_name if hasattr(self.config, 'timepoint_dir_name') else "TimePoint_1"
            timepoint_path = input_dir / timepoint_dir

            if timepoint_path.exists():
                has_zstack_folders, z_folders = self.detect_zstack_folders(input_dir)

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
        # All intermediate images used for stitching should go in the processed directory
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
