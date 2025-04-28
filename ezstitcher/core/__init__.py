"""Core module for ezstitcher."""

# Import main classes for instance-based API
from ezstitcher.core.image_processor import ImageProcessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

# Import configuration classes
from ezstitcher.core.config import (
    StitcherConfig,
    PipelineConfig
)

# Import pipeline factory class
from ezstitcher.core.pipeline_factories import AutoPipelineFactory

__all__ = [
    'ImageProcessor',
    'FocusAnalyzer',
    'Stitcher',
    'FileSystemManager',
    'PipelineOrchestrator',
    'StitcherConfig',
    'PipelineConfig',
    'AutoPipelineFactory',
]
