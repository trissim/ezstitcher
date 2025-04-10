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
    def pad_site_number(self, filename):
        """
        Pad site number in filename to 3 digits.

        Args:
            filename (str): Filename to pad

        Returns:
            str: Filename with padded site number
        """
        site_match = re.search(r'_s(\d{1,3})(?=_|\.)', filename)
        if site_match:
            site_num = site_match.group(1)
            if len(site_num) < 3:
                padded = site_num.zfill(3)
                filename = filename.replace(f"_s{site_num}", f"_s{padded}")
        return filename

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

    def get_zstack_info(self, folder_path):
        """
        Get detailed information about Z-stacks in a folder.

        Args:
            folder_path (str or Path): Path to the folder containing Z-stack images

        Returns:
            dict: Dictionary mapping stack IDs to information about each stack
        """
        folder_path = Path(folder_path)

        # Detect Z-stack images
        has_zstack, z_indices_map = self.detect_zstack_images(folder_path)

        if not has_zstack:
            return {}

        # Get all image files
        all_files = self.fs_manager.list_image_files(folder_path)

        # Group files by stack ID, wavelength, and Z-index
        z_info = {}

        for img_file in all_files:
            metadata = self.filename_parser.parse_filename(str(img_file))

            if not metadata or 'z_index' not in metadata or metadata['z_index'] is None:
                continue

            well = metadata['well']
            site = metadata['site']
            channel = metadata['channel']
            z_index = metadata['z_index']

            # Create a consistent base name for grouping
            stack_id = f"{well}_s{site:03d}"

            # Initialize the stack info if it doesn't exist
            if stack_id not in z_info:
                z_info[stack_id] = {
                    'z_indices': sorted(z_indices_map.get(f"{stack_id}_w{channel}", [])),
                    'wavelengths': set(),
                    'files': {}
                }

            # Add the wavelength
            z_info[stack_id]['wavelengths'].add(channel)

            # Initialize the wavelength dict if it doesn't exist
            if channel not in z_info[stack_id]['files']:
                z_info[stack_id]['files'][channel] = {}

            # Add the file
            z_info[stack_id]['files'][channel][z_index] = img_file

        # Convert wavelengths sets to sorted lists
        for stack_id in z_info:
            z_info[stack_id]['wavelengths'] = sorted(z_info[stack_id]['wavelengths'])

        return z_info
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

    def create_best_focus_images(self, input_dir, output_dir=None, focus_method='combined', focus_wavelength='all'):
        """
        Select the best focused image from each Z-stack and save to output directory.

        Args:
            input_dir (str or Path): Directory with Z-stack images
            output_dir (str or Path): Directory to save best focus images. If None, creates a directory named {plate_name}_best_focus
            focus_method (str): Focus detection method
            focus_wavelength (str): Wavelength to use for focus detection

        Returns:
            tuple: (success, output_dir) where success is a boolean and output_dir is the path to the output directory
        """
        input_dir = Path(input_dir)

        # If output_dir is None, create a directory named {plate_name}_best_focus
        if output_dir is None:
            plate_path = input_dir.parent if input_dir.name == "TimePoint_1" else input_dir
            parent_dir = plate_path.parent
            plate_name = plate_path.name
            best_focus_suffix = self.config.best_focus_dir_suffix if hasattr(self.config, 'best_focus_dir_suffix') else "_best_focus"
            output_dir = parent_dir / f"{plate_name}{best_focus_suffix}"

        # Create TimePoint_1 directory in output_dir
        timepoint_dir = self.config.timepoint_dir_name if hasattr(self.config, 'timepoint_dir_name') else "TimePoint_1"
        timepoint_dir = output_dir / timepoint_dir
        self.fs_manager.ensure_directory(timepoint_dir)

        # Check if folder contains Z-stack images
        has_zstack, z_indices_map = self.detect_zstack_images(input_dir)
        if not has_zstack:
            logger.warning(f"No Z-stack images found in {input_dir}")
            return False, None

        # Group images by well, site, and wavelength
        images_by_coordinates = defaultdict(list)

        # Check if we need to auto-detect the parser
        if not hasattr(self, 'filename_parser') or self.filename_parser is None:
            # Import here to avoid circular imports
            from ezstitcher.core.filename_parser import detect_parser
            all_files = self.fs_manager.list_image_files(input_dir)
            file_paths = [str(f) for f in all_files]
            if file_paths:
                self.filename_parser = detect_parser(file_paths)
                logger.info(f"Auto-detected parser: {self.filename_parser.__class__.__name__}")
            else:
                from ezstitcher.core.filename_parser import ImageXpressFilenameParser
                self.filename_parser = ImageXpressFilenameParser()
                logger.info("No files found, defaulting to ImageXpress parser")

        # Group Z-indices by coordinates
        for base_name, z_indices in z_indices_map.items():
            # Parse the base_name to extract well, site, and channel
            # The base_name is in the format "well_sXXX_wY"
            parts = base_name.split('_')
            if len(parts) >= 3 and parts[1].startswith('s') and parts[2].startswith('w'):
                well = parts[0]
                site = int(parts[1][1:])
                wavelength = int(parts[2][1:])

                # Create coordinates key
                coordinates = (well, site, wavelength)

                # Add to dictionary
                images_by_coordinates[coordinates] = (base_name, z_indices)
            else:
                # If we get here, we couldn't parse the base_name
                logger.warning(f"Could not parse coordinates from {base_name}")

        # Process each set of coordinates
        best_focus_results = {}

        # Filter by wavelength if specified
        if focus_wavelength != 'all':
            focus_wavelength = int(focus_wavelength)
            focus_coordinates = [coords for coords in images_by_coordinates.keys() if coords[2] == focus_wavelength]
        else:
            focus_coordinates = list(images_by_coordinates.keys())

        # Process each set of focus coordinates
        for coordinates in focus_coordinates:
            well, site, wavelength = coordinates
            base_name, z_indices = images_by_coordinates[coordinates]

            # Load all Z-stack images for this coordinate
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

            # Find best focus using FocusAnalyzer
            best_img, best_z, scores = self.focus_analyzer.select_best_focus(image_stack, method=focus_method)
            z_index = sorted(z_indices)[best_z]

            # Save best focus image
            output_filename = f"{well}_s{site:03d}_w{wavelength}.tif"
            output_path = timepoint_dir / output_filename
            self.fs_manager.save_image(output_path, best_img)
            logger.info(f"Saved best focus image for {base_name} (z={z_index}) to {output_path}")

            # Store best Z-index for this coordinate
            best_focus_results[coordinates] = z_index

        # If focus_wavelength is not 'all', use the same Z-index for other wavelengths
        if focus_wavelength != 'all':
            for coordinates in images_by_coordinates.keys():
                well, site, wavelength = coordinates
                if wavelength != focus_wavelength:
                    # Find the corresponding focus coordinates
                    focus_coords = (well, site, focus_wavelength)
                    if focus_coords in best_focus_results:
                        # Use the same Z-index as the focus wavelength
                        best_z = best_focus_results[focus_coords]
                        base_name, z_indices = images_by_coordinates[coordinates]

                        # Load the image at the best Z-index
                        # Parse the base_name to extract well, site, and channel
                        parts = base_name.split('_')
                        if len(parts) >= 3 and parts[1].startswith('s') and parts[2].startswith('w'):
                            well = parts[0]
                            site = int(parts[1][1:])
                            channel = int(parts[2][1:])

                            # Construct the filename using the filename parser
                            filename = self.filename_parser.construct_filename(well, site, channel, best_z)
                            img_path = input_dir / filename

                            # If the file doesn't exist, try alternative extensions
                            if not img_path.exists():
                                for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                                    alt_filename = self.filename_parser.construct_filename(well, site, channel, best_z, extension=ext)
                                    alt_path = input_dir / alt_filename
                                    if alt_path.exists():
                                        img_path = alt_path
                                        break
                        else:
                            # Fallback if we can't parse the base_name
                            img_path = input_dir / f"{base_name}_z{best_z:03d}.tif"

                        img = self.fs_manager.load_image(img_path)
                        if img is not None:
                            # Save the image
                            output_filename = f"{well}_s{site:03d}_w{wavelength}.tif"
                            output_path = timepoint_dir / output_filename
                            self.fs_manager.save_image(output_path, img)
                            logger.info(f"Saved best focus image for {base_name} (z={best_z}) to {output_path}")
                            best_focus_results[coordinates] = best_z

        return len(best_focus_results) > 0, output_dir

    def find_best_focus(self, timepoint_dir, output_dir):
        """
        Find the best focus plane for each Z-stack and save it to the output directory.

        Args:
            timepoint_dir (Path): Path to the TimePoint_1 directory
            output_dir (Path): Path to the output directory

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure the output directory exists
            self.fs_manager.ensure_directory(output_dir)

            # Get all Z-stack info
            z_info = self.get_zstack_info(timepoint_dir)
            if not z_info:
                logger.error(f"No Z-stack info found in {timepoint_dir}")
                return False

            # Process each unique image stack
            for stack_id, stack_info in z_info.items():
                # Skip stacks with only one Z-plane
                if len(stack_info['z_indices']) <= 1:
                    # Just copy the single image to the output directory
                    for wavelength in stack_info['wavelengths']:
                        src_path = stack_info['files'][wavelength][stack_info['z_indices'][0]]
                        dst_path = output_dir / f"{stack_id}_w{wavelength}.tif"
                        import shutil
                        shutil.copy(src_path, dst_path)
                        logger.info(f"Copied single Z-plane image to {dst_path}")
                    continue

                # Process each wavelength separately
                for wavelength in stack_info['wavelengths']:
                    # Load all images for this wavelength
                    images = []
                    for z_idx in stack_info['z_indices']:
                        img_path = stack_info['files'][wavelength][z_idx]
                        import cv2
                        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                        if img is None:
                            logger.error(f"Failed to read image: {img_path}")
                            continue
                        images.append(img)

                    if not images:
                        logger.error(f"No images loaded for {stack_id}_w{wavelength}")
                        continue

                    # Find the best focus plane
                    best_idx, _ = self.focus_analyzer.find_best_focus(images, method=self.config.focus_method)
                    best_z = stack_info['z_indices'][best_idx]

                    # Save the best focus image
                    best_img = images[best_idx]
                    output_path = output_dir / f"{stack_id}_w{wavelength}.tif"
                    self.fs_manager.save_image(output_path, best_img)
                    logger.info(f"Saved best focus for {stack_id}_w{wavelength} to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Error in find_best_focus: {e}", exc_info=True)
            return False

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

    def stitch_across_z(self, plate_folder, reference_z=None, stitch_all_z_planes=True, processor=None, preprocessing_funcs=None):
        """
        Stitch all Z-planes in a plate using a reference Z-plane for positions.

        Args:
            plate_folder (str or Path): Path to the plate folder
            reference_z (str or callable, optional): Reference Z-plane to use for positions.
                If str: 'max', 'mean', or 'best_focus'
                If callable: A function that takes a Z-stack (list of images) and returns a 2D image
                If None: Uses the z_reference_function from the config
            stitch_all_z_planes (bool): Whether to stitch all Z-planes
            processor (PlateProcessor): Processor to use for stitching
            preprocessing_funcs (dict, optional): Dictionary mapping channels to preprocessing functions.
                These functions are applied to individual images before Z-stack processing.

        Returns:
            bool: True if successful, False otherwise
        """
        # Use provided preprocessing functions or the ones from initialization
        preprocessing_funcs = preprocessing_funcs or self.preprocessing_funcs
        try:
            plate_path = Path(plate_folder)
            timepoint_dir = "TimePoint_1"
            timepoint_path = plate_path / timepoint_dir

            if not timepoint_path.exists():
                logger.error(f"{timepoint_dir} folder does not exist in {plate_folder}")
                return False

            # Check if folder contains Z-stack images
            has_zstack, z_indices_map = self.detect_zstack_images(timepoint_path)
            if not has_zstack:
                logger.warning(f"No Z-stack images found in {timepoint_path}")
                return False

            # Get all unique Z-indices
            all_z_indices = set()
            for base_name, indices in z_indices_map.items():
                all_z_indices.update(indices)
            z_indices = sorted(list(all_z_indices))
            logger.info(f"Found {len(z_indices)} Z-planes: {z_indices}")

            # Get reference positions
            parent_dir = plate_path.parent
            plate_name = plate_path.name

            # If reference_z is not provided, use the z_reference_function from the config
            if reference_z is None:
                # Use the reference function from the config
                reference_function = self._reference_function

                # For backward compatibility, convert the function to a string name if possible
                if reference_function == self._adapt_function(self.image_preprocessor.max_projection):
                    reference_z = 'max'
                elif reference_function == self._adapt_function(self.image_preprocessor.mean_projection):
                    reference_z = 'mean'
                elif callable(reference_function) and reference_function.__name__ == '<lambda>' and 'select_best_focus' in str(reference_function):
                    reference_z = 'best_focus'
                else:
                    # Use the function directly
                    reference_z = reference_function

            # For string names, convert to the appropriate function
            if isinstance(reference_z, str):
                logger.info(f"Using reference_z: {reference_z}")

                # Determine reference directory based on reference_z
                if reference_z == 'max':
                    reference_dir = parent_dir / f"{plate_name}_projections_max" / timepoint_dir
                    # Ensure the directory exists
                    self.fs_manager.ensure_directory(reference_dir)

                    # Create max projections for each Z-stack
                    logger.info(f"Creating max projections for reference")
                    self.create_zstack_projections(plate_path / timepoint_dir, reference_dir)
                elif reference_z == 'mean':
                    reference_dir = parent_dir / f"{plate_name}_projections_mean" / timepoint_dir
                    # Ensure the directory exists
                    self.fs_manager.ensure_directory(reference_dir)

                    # Create mean projections for each Z-stack
                    logger.info(f"Creating mean projections for reference")
                    self.create_zstack_projections(plate_path / timepoint_dir, reference_dir, projection_types=['mean'])
                elif reference_z == 'best_focus':
                    reference_dir = parent_dir / f"{plate_name}_best_focus" / timepoint_dir
                    # Ensure the directory exists
                    self.fs_manager.ensure_directory(reference_dir)

                    # Create best focus projections for each Z-stack
                    logger.info(f"Creating best focus projections for reference")
                    # For best focus, we need to use the find_best_focus method
                    self.find_best_focus(plate_path / timepoint_dir, reference_dir)
                else:
                    logger.error(f"Invalid reference_z string: {reference_z}. Must be 'max', 'mean', or 'best_focus'.")
                    return False
            elif callable(reference_z):
                # If reference_z is a function, we need to create a custom projection directory
                custom_proj_dir = parent_dir / f"{plate_name}_projections_custom" / timepoint_dir
                self.fs_manager.ensure_directory(custom_proj_dir)

                # Create custom projections for each Z-stack
                logger.info(f"Creating custom projections using provided function")

                # Process each Z-stack
                for base_name, z_indices in z_indices_map.items():
                    # Load all images in the Z-stack
                    image_stack = []
                    channel = '1'  # Default channel

                    for z_idx in sorted(z_indices):
                        # Check if images are in ZStep folders or have _z in the filename
                        zstep_folder = timepoint_path / f"ZStep_{z_idx}"
                        if zstep_folder.exists():
                            # Images are in ZStep folders
                            filename = f"{base_name}.tif"
                            file_path = zstep_folder / filename
                        else:
                            # Check if this is an Opera Phenix file
                            opera_match = re.match(r'(r\d{1,2}c\d{1,2}f\d+).*', base_name)
                            if opera_match:
                                # Opera Phenix format
                                opera_base = opera_match.group(1)
                                # Extract channel using the filename parser
                                metadata = self.filename_parser.parse_filename(base_name)
                                channel = str(metadata.get('channel', '1')) if metadata else '1'
                                filename = f"{opera_base}p{z_idx:02d}-ch{channel}sk1fk1fl1.tiff"
                                file_path = timepoint_path / filename
                            else:
                                # ImageXpress format
                                filename = f"{base_name}_z{z_idx:03d}.tif"
                                file_path = timepoint_path / filename
                                # Extract channel using the filename parser
                                metadata = self.filename_parser.parse_filename(base_name)
                                channel = str(metadata.get('channel', '1')) if metadata else '1'

                        if file_path.exists():
                            img = self.fs_manager.load_image(file_path)
                            if img is not None:
                                # Apply preprocessing if available for this channel
                                if channel in preprocessing_funcs:
                                    img = preprocessing_funcs[channel](img)
                                image_stack.append(img)

                    if not image_stack:
                        logger.warning(f"No images found for Z-stack {base_name}")
                        continue

                    # Apply the custom function to create the projection
                    try:
                        custom_projection = reference_z(image_stack)

                        # Save the custom projection
                        output_filename = f"{base_name}.tif"
                        output_path = custom_proj_dir / output_filename
                        self.fs_manager.save_image(output_path, custom_projection)
                        logger.info(f"Saved custom projection for {base_name} to {output_path}")
                    except Exception as e:
                        logger.error(f"Error creating custom projection for {base_name}: {e}")
                        continue

                # Use the custom projections directory as reference
                reference_dir = custom_proj_dir
            else:
                logger.error(f"Invalid reference_z type: {type(reference_z)}. Must be str or callable.")
                return False

            if not reference_dir.exists():
                logger.error(f"Reference directory does not exist: {reference_dir}")
                return False

            # Get positions from reference directory
            positions_dir = parent_dir / f"{plate_name}_positions"

            # If positions directory doesn't exist or is empty, we need to stitch the projections first
            if not positions_dir.exists() or not list(positions_dir.glob("*.csv")):
                logger.info(f"No position files found in {positions_dir}, stitching projections first")

                # Create a new processor with the same config as the one passed in
                if processor is None:
                    logger.error(f"No processor provided for stitching projections")
                    return False

                # Stitch the projections to generate position files
                success = processor.run(
                    str(reference_dir.parent)
                )

                if not success:
                    logger.error(f"Failed to stitch projections")
                    return False

                # Check if positions directory exists now
                # The positions directory might be named differently depending on the plate name
                # Try to find the positions directory by looking for directories with 'positions' in the name
                proj_positions_dirs = [d for d in parent_dir.glob(f"*positions*") if d.is_dir()]
                if not proj_positions_dirs:
                    logger.error(f"No positions directory found after stitching projections")
                    return False

                # Use the first positions directory found
                proj_positions_dir = proj_positions_dirs[0]
                logger.info(f"Found positions directory: {proj_positions_dir}")

                # Copy position files from projections positions directory to positions directory
                import shutil
                # Make sure the positions directory exists
                self.fs_manager.ensure_directory(positions_dir)

                # List all position files in the projections positions directory
                position_files = list(proj_positions_dir.glob("*.csv"))
                logger.info(f"Found {len(position_files)} position files in {proj_positions_dir}")

                # Copy all position files from the projections positions directory to the positions directory
                position_files_copied = 0
                for position_file in position_files:
                    # Get the well name from the position file name
                    well_name = position_file.stem.split('_')[0]
                    # Copy the position file to the positions directory
                    dest_file = positions_dir / f"{well_name}_w1.csv"
                    shutil.copy(position_file, dest_file)
                    position_files_copied += 1
                    logger.info(f"Copied position file from {position_file} to {dest_file}")

                # Check if position files were copied
                if position_files_copied == 0:
                    # Try to find position files in other directories
                    all_position_dirs = [d for d in parent_dir.glob(f"*positions*") if d.is_dir()]
                    logger.info(f"Found {len(all_position_dirs)} position directories: {all_position_dirs}")

                    # Try each directory
                    for pos_dir in all_position_dirs:
                        if pos_dir == positions_dir:
                            continue

                        position_files = list(pos_dir.glob("*.csv"))
                        logger.info(f"Found {len(position_files)} position files in {pos_dir}")

                        for position_file in position_files:
                            # Get the well name from the position file name
                            well_name = position_file.stem.split('_')[0]
                            # Copy the position file to the positions directory
                            dest_file = positions_dir / f"{well_name}_w1.csv"
                            shutil.copy(position_file, dest_file)
                            position_files_copied += 1
                            logger.info(f"Copied position file from {position_file} to {dest_file}")

                    if position_files_copied == 0:
                        logger.error(f"No position files found in any positions directory")
                        return False

                logger.info(f"Copied {position_files_copied} position files to {positions_dir}")

            # Get all position files
            position_files = list(positions_dir.glob("*.csv"))
            if not position_files:
                logger.error(f"No position files found in {positions_dir} after stitching projections")
                return False

            logger.info(f"Found {len(position_files)} position files in {positions_dir}")

            # Stitch each Z-plane using the reference positions
            stitched_dir = parent_dir / f"{plate_name}_stitched" / timepoint_dir
            self.fs_manager.ensure_directory(stitched_dir)

            # For each Z-plane, stitch all wells and wavelengths
            for z_index in z_indices:
                logger.info(f"Stitching Z-plane {z_index}")

                # For each position file (which corresponds to a well and wavelength)
                for pos_file in position_files:
                    # Extract well and wavelength from position file name
                    match = re.match(r'(.+)_w(\d+)\.csv', pos_file.name)
                    if not match:
                        logger.warning(f"Could not parse position file name: {pos_file.name}")
                        continue

                    well_pattern = match.group(1)
                    wavelength = match.group(2)

                    # Read positions from CSV
                    positions = []
                    with open(pos_file, 'r') as f:
                        for line in f:
                            # Parse the line format: "file: C01_s001_w1.tif; grid: (0, 0); position: (0.0, 0.0)"
                            if 'file:' in line and 'position:' in line:
                                # Extract the site number from the filename
                                file_part = line.split(';')[0].strip()
                                filename = file_part.split('file:')[1].strip()
                                site_match = re.search(r's(\d+)_', filename)
                                if site_match:
                                    site = int(site_match.group(1))

                                    # Extract the position coordinates
                                    pos_part = line.split('position:')[1].strip()
                                    pos_match = re.search(r'\((\d+\.\d+), (\d+\.\d+)\)', pos_part)
                                    if pos_match:
                                        x = float(pos_match.group(1))
                                        y = float(pos_match.group(2))
                                        positions.append((site, x, y))

                    if not positions:
                        logger.warning(f"No positions found in {pos_file}")
                        continue

                    # Get all tiles for this well, wavelength, and Z-plane
                    tiles = []
                    for site, x, y in positions:
                        # Construct the filename and path
                        # First, check if files are in ZStep folders
                        zstep_folder = timepoint_path / f"ZStep_{z_index}"
                        if zstep_folder.exists():
                            # Files are in ZStep folders
                            filename = f"{well_pattern}_s{site:03d}_w{wavelength}.tif"
                            file_path = zstep_folder / filename
                            logger.info(f"Looking for file in ZStep folder: {file_path}")
                        else:
                            # Use the filename parser to construct the filename
                            # Check if we need to auto-detect the parser
                            if not hasattr(self, 'filename_parser') or self.filename_parser is None:
                                # Import here to avoid circular imports
                                from ezstitcher.core.filename_parser import detect_parser
                                # Get a sample of filenames from the timepoint_path
                                sample_files = []
                                for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                                    sample_files.extend([str(f) for f in timepoint_path.glob(f"*{ext}")][:10])
                                if sample_files:
                                    self.filename_parser = detect_parser(sample_files)
                                    logger.info(f"Auto-detected parser: {self.filename_parser.__class__.__name__}")
                                else:
                                    from ezstitcher.core.filename_parser import ImageXpressFilenameParser
                                    self.filename_parser = ImageXpressFilenameParser()
                                    logger.info("No files found, defaulting to ImageXpress parser")

                            # Construct the filename using the filename parser
                            try:
                                filename = self.filename_parser.construct_filename(well_pattern, site, int(wavelength), z_index)
                                file_path = timepoint_path / filename
                                logger.info(f"Looking for file: {file_path}")

                                # If the file doesn't exist, try alternative extensions
                                if not file_path.exists():
                                    for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                                        alt_filename = self.filename_parser.construct_filename(well_pattern, site, int(wavelength), z_index, extension=ext)
                                        alt_path = timepoint_path / alt_filename
                                        if alt_path.exists():
                                            file_path = alt_path
                                            break
                            except Exception as e:
                                # Fallback to ImageXpress format if the parser fails
                                logger.warning(f"Error constructing filename: {e}. Falling back to default format.")
                                filename = f"{well_pattern}_s{site:03d}_w{wavelength}_z{z_index:03d}.tif"
                                file_path = timepoint_path / filename

                        if file_path.exists():
                            # Load the image
                            img = self.fs_manager.load_image(file_path)
                            if img is not None:
                                tiles.append((site, x, y, img))
                        else:
                            logger.warning(f"Tile not found: {file_path}")

                    if not tiles:
                        logger.warning(f"No tiles found for {well_pattern}_w{wavelength}_z{z_index}")
                        continue

                    # Stitch the tiles
                    logger.info(f"Stitching {len(tiles)} tiles for {well_pattern}_w{wavelength}_z{z_index}")

                    # Determine canvas size
                    max_x = max(x + img.shape[1] for _, x, _, img in tiles)
                    max_y = max(y + img.shape[0] for _, _, y, img in tiles)
                    canvas = np.zeros((int(max_y), int(max_x)), dtype=np.uint16)

                    # Place tiles on canvas
                    for site, x, y, img in tiles:
                        x_start, y_start = int(x), int(y)
                        x_end, y_end = x_start + img.shape[1], y_start + img.shape[0]

                        # Ensure we don't go out of bounds
                        x_end = min(x_end, canvas.shape[1])
                        y_end = min(y_end, canvas.shape[0])

                        # Place the tile
                        canvas[y_start:y_end, x_start:x_end] = img[:y_end-y_start, :x_end-x_start]

                    # Save the stitched image
                    output_filename = f"{well_pattern}_w{wavelength}_z{z_index:03d}.tif"
                    output_path = stitched_dir / output_filename
                    self.fs_manager.save_image(output_path, canvas)
                    logger.info(f"Saved stitched image to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Error in stitch_across_z: {e}", exc_info=True)
            return False
            k