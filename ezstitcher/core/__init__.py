"""Core module for ezstitcher."""

# Import main classes for instance-based API
from ezstitcher.core.image_processor import ImageProcessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.pipeline import Pipeline, Step, ProcessingContext # Added missing imports from previous diff

# Import configuration classes
from ezstitcher.core.config import (
    StitcherConfig,
    PipelineConfig
)

# Import pipeline factory class
from ezstitcher.core.pipeline_factories import AutoPipelineFactory

__all__ = [
    'ProcessingContext',
    'ImageProcessor',
    'FocusAnalyzer',
    'Stitcher',
    'PipelineOrchestrator',
    'StitcherConfig',
    'PipelineConfig',
    'AutoPipelineFactory',
    'Pipeline', # Added missing export
    'Step', # Added missing export
    'ProcessingContext', # Added missing export
]
