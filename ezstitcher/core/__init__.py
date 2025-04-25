"""Core module for ezstitcher.

This module provides the core functionality for the ezstitcher package.
"""

# Import main functions for backward compatibility
# Removed obsolete imports of non-existent functions from main.py

# Import classes for instance-based API
from ezstitcher.core.image_processor import ImageProcessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

# Import configuration classes
from ezstitcher.core.config import (
    StitcherConfig,
    FocusAnalyzerConfig,
    PipelineConfig
)

# Import utility classes
from ezstitcher.core.microscope_interfaces import FilenameParser
from ezstitcher.core.image_locator import ImageLocator
