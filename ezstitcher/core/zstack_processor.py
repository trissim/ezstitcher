"""
Z-stack processing module for ezstitcher.

This module provides a class for processing Z-stack images.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Callable, Tuple

from ezstitcher.core.config import ZStackProcessorConfig, FocusConfig
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.filename_parser import ImageXpressFilenameParser
from ezstitcher.core.zstack_organizer import ZStackOrganizer
from ezstitcher.core.zstack_projector import ZStackProjector
from ezstitcher.core.zstack_focus_manager import ZStackFocusManager
from ezstitcher.core.zstack_stitcher import ZStackStitcher
from ezstitcher.core.zstack_reference_adapter import ZStackReferenceAdapter

logger = logging.getLogger(__name__)


class ZStackProcessor:
    """
    Handles Z-stack specific operations:
    - Detection
    - Projection creation
    - Best focus selection
    - Per-plane stitching

    This class acts as a coordinator for the various Z-stack processing components.
    """
    def __init__(self, config: ZStackProcessorConfig, filename_parser=None, preprocessing_funcs=None):
        """
        Initialize the ZStackProcessor.

        Args:
            config: Configuration for Z-stack processing
            filename_parser: Parser for microscopy filenames
            preprocessing_funcs: Dictionary mapping channels to preprocessing functions
        """
        self.config = config
        self.fs_manager = FileSystemManager()
        self._z_info = None
        self._z_indices = []
        self.preprocessing_funcs = preprocessing_funcs or {}

        # Initialize the focus analyzer
        focus_config = config.focus_config or FocusConfig(method=config.focus_method)
        self.focus_analyzer = FocusAnalyzer(focus_config)

        # Initialize the filename parser
        if filename_parser is None:
            self.filename_parser = ImageXpressFilenameParser()
        else:
            self.filename_parser = filename_parser

        # Initialize the image preprocessor
        self.image_preprocessor = ImagePreprocessor()

        # Initialize component classes
        self.organizer = ZStackOrganizer(config, self.filename_parser, self.fs_manager)
        self.projector = ZStackProjector(config, self.image_preprocessor, self.fs_manager, self.filename_parser)
        self.focus_manager = ZStackFocusManager(config, self.focus_analyzer, self.fs_manager, self.filename_parser)
        self.stitcher = ZStackStitcher(config, self.fs_manager, self.filename_parser)
        self.reference_adapter = ZStackReferenceAdapter(self.image_preprocessor, self.focus_analyzer)

        # Initialize the reference function
        self._reference_function = self.reference_adapter.create_reference_function(config.z_reference_function, config.focus_method)

    def detect_z_stacks(self, plate_folder):
        """
        Detect Z-stacks in a plate folder.

        Args:
            plate_folder: Path to the plate folder

        Returns:
            bool: True if Z-stacks were detected, False otherwise
        """
        has_zstack, self._z_info = self.preprocess_plate_folder(plate_folder)

        if has_zstack and 'z_indices_map' in self._z_info and self._z_info['z_indices_map']:
            all_z_indices = set()
            for base_name, indices in self._z_info['z_indices_map'].items():
                all_z_indices.update(indices)
            self._z_indices = sorted(list(all_z_indices))

        return has_zstack

    def preprocess_plate_folder(self, plate_folder):
        """
        Preprocess a plate folder to detect and organize Z-stacks.

        Args:
            plate_folder: Path to the plate folder

        Returns:
            tuple: (has_zstack, z_info)
        """
        plate_path = Path(plate_folder)

        # Initialize directory structure manager
        dir_structure = self.fs_manager.initialize_dir_structure(plate_folder)
        logger.info(f"Detected directory structure: {dir_structure.structure_type}")

        has_zstack_folders, z_folders = self.detect_zstack_folders(plate_folder)

        if has_zstack_folders:
            logger.info(f"Organizing Z-stack folders in {plate_folder}")
            self.organize_zstack_folders(plate_folder)

        # Get timepoint directory from directory structure manager
        timepoint_path = dir_structure.get_timepoint_dir()
        if timepoint_path:
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

    # Delegate to ZStackOrganizer
    def detect_zstack_folders(self, plate_folder):
        """
        Detect Z-stack folders in a plate folder.

        Args:
            plate_folder: Path to the plate folder

        Returns:
            tuple: (has_zstack, z_folders)
        """
        return self.organizer.detect_zstack_folders(plate_folder)

    def detect_zstack_images(self, folder_path):
        """
        Detect Z-stack images in a folder.

        Args:
            folder_path: Path to the folder

        Returns:
            tuple: (has_zstack, z_indices_map)
        """
        return self.organizer.detect_zstack_images(folder_path)

    def organize_zstack_folders(self, plate_folder):
        """
        Organize Z-stack folders by moving files to TimePoint_1 with proper naming.

        Args:
            plate_folder: Path to the plate folder

        Returns:
            bool: True if successful, False otherwise
        """
        return self.organizer.organize_zstack_folders(plate_folder)

    def get_z_indices(self):
        """
        Return the list of detected Z indices after calling detect_z_stacks().

        Returns:
            list: List of Z indices
        """
        return self.organizer.get_z_indices()

    def get_zstack_info(self, folder_path):
        """
        Get detailed information about Z-stacks in a folder.

        Args:
            folder_path: Path to the folder

        Returns:
            dict: Dictionary mapping stack IDs to information about each stack
        """
        return self.organizer.get_zstack_info(folder_path)

    # Delegate to ZStackFocusManager
    def create_best_focus_images(self, input_dir, output_dir=None, focus_method='combined', focus_wavelength='all'):
        """
        Select the best focused image from each Z-stack and save to output directory.

        Args:
            input_dir: Directory with Z-stack images
            output_dir: Directory to save best focus images
            focus_method: Focus detection method
            focus_wavelength: Wavelength to use for focus detection

        Returns:
            tuple: (success, output_dir)
        """
        return self.focus_manager.create_best_focus_images(input_dir, output_dir, focus_method, focus_wavelength)

    def find_best_focus(self, timepoint_dir, output_dir):
        """
        Find the best focus plane for each Z-stack and save it to the output directory.

        Args:
            timepoint_dir: Path to the TimePoint_1 directory
            output_dir: Path to the output directory

        Returns:
            bool: True if successful, False otherwise
        """
        return self.focus_manager.find_best_focus(timepoint_dir, output_dir)

    # Delegate to ZStackProjector
    def create_zstack_projections(self, input_dir, output_dir, projection_types=None, preprocessing_funcs=None):
        """
        Create projections from Z-stack images.

        Args:
            input_dir: Directory containing Z-stack images
            output_dir: Directory to save projections
            projection_types: List of projection types to create
            preprocessing_funcs: Dictionary mapping channels to preprocessing functions

        Returns:
            tuple: (success, projections_info)
        """
        return self.projector.create_projections(input_dir, output_dir, projection_types, preprocessing_funcs)

    # Delegate to ZStackStitcher
    def stitch_across_z(self, plate_folder, reference_z=None, stitch_all_z_planes=True, processor=None, preprocessing_funcs=None):
        """
        Stitch all Z-planes in a plate using a reference Z-plane for positions.

        Args:
            plate_folder: Path to the plate folder
            reference_z: Reference Z-plane to use for positions
            stitch_all_z_planes: Whether to stitch all Z-planes
            processor: Processor to use for stitching
            preprocessing_funcs: Dictionary mapping channels to preprocessing functions

        Returns:
            bool: True if successful, False otherwise
        """
        return self.stitcher.stitch_across_z(plate_folder, reference_z, stitch_all_z_planes, processor, preprocessing_funcs)

    # Delegate to ZStackReferenceAdapter
    def _create_reference_function(self, func_or_name):
        """
        Create a reference function from a string name or callable.

        Args:
            func_or_name: String name of a standard function or a callable

        Returns:
            callable: A function that takes a stack and returns a single image
        """
        return self.reference_adapter.create_reference_function(func_or_name, self.config.focus_method)

    def _adapt_function(self, func):
        """
        Adapt a function to the reference function interface.

        Args:
            func: The function to adapt

        Returns:
            callable: A function that takes a stack and returns a single image
        """
        return self.reference_adapter.adapt_function(func)

    def _preprocess_stack(self, stack, channel):
        """
        Apply preprocessing to each image in a Z-stack.

        Args:
            stack: List of images in the Z-stack
            channel: Channel identifier for selecting the preprocessing function

        Returns:
            list: List of preprocessed images
        """
        return self.reference_adapter.preprocess_stack(stack, channel, self.preprocessing_funcs)

    def pad_site_number(self, filename):
        """
        Pad site number in filename to 3 digits.

        This method is kept for backward compatibility.
        New code should use the filename parser's methods directly.

        Args:
            filename: Filename to pad

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
