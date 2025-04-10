"""Core module for ezstitcher.

This module provides the core functionality for the ezstitcher package.
"""

# Import main functions for backward compatibility
from ezstitcher.core.main import (
    process_plate_folder,
    modified_process_plate_folder,
    process_bf,
    find_best_focus,
    process_plate_folder_with_config
)

# Import classes for instance-based API
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.zstack_processor import ZStackProcessor
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.plate_processor import PlateProcessor

# Import configuration classes
from ezstitcher.core.config import (
    StitcherConfig,
    FocusAnalyzerConfig,
    ImagePreprocessorConfig,
    ZStackProcessorConfig,
    PlateProcessorConfig
)

# Import Pydantic configuration classes
from ezstitcher.core.pydantic_config import (
    PlateProcessorConfig as PydanticPlateProcessorConfig,
    StitcherConfig as PydanticStitcherConfig,
    ZStackProcessorConfig as PydanticZStackProcessorConfig,
    FocusAnalyzerConfig as PydanticFocusAnalyzerConfig,
    ImagePreprocessorConfig as PydanticImagePreprocessorConfig,
    ConfigPresets
)

# Import utility classes
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.filename_parser import FilenameParser
from ezstitcher.core.csv_handler import CSVHandler
from ezstitcher.core.pattern_matcher import PatternMatcher
from ezstitcher.core.directory_structure_manager import DirectoryStructureManager
from ezstitcher.core.image_locator import ImageLocator