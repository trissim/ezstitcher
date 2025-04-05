"""Core module for ezstitcher.

This module provides the core functionality for the ezstitcher package.
"""

# Import main functions for backward compatibility
from ezstitcher.core.main import (
    process_plate_folder,
    modified_process_plate_folder,
    process_bf,
    find_best_focus
)

# Import classes for new class-based API
from ezstitcher.core.image_processor import ImageProcessor
from ezstitcher.core.focus_detector import FocusDetector
from ezstitcher.core.z_stack_manager import ZStackManager
from ezstitcher.core.stitcher_manager import StitcherManager