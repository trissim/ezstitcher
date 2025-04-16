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
