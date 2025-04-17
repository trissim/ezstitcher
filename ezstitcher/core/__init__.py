"""Core module for ezstitcher.

This module provides the core functionality for the ezstitcher package.
"""

# Import main functions for backward compatibility
# Removed obsolete imports of non-existent functions from main.py

# Import classes for instance-based API
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Import configuration classes
from ezstitcher.core.config import (
    StitcherConfig,
    FocusAnalyzerConfig,
    ImagePreprocessorConfig,
    ZStackProcessorConfig,
    PlateProcessorConfig,
    PipelineConfig
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
from ezstitcher.core.microscope_interfaces import FilenameParser
from ezstitcher.core.image_locator import ImageLocator
