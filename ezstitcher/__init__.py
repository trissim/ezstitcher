"""EZStitcher - Easy Microscopy Image Stitching Tool.

EZStitcher is a Python package for stitching microscopy images with support for
Z-stacks, multi-channel fluorescence, and advanced focus detection.
"""

__version__ = "0.1.0"

# Import main functions for backward compatibility
# Removed obsolete imports of non-existent functions from ezstitcher.core

# Import classes for instance-based API
from ezstitcher.core import (
    ImageProcessor,
    FocusAnalyzer,
    Stitcher,
    PipelineOrchestrator
)

# Import configuration classes
from ezstitcher.core import (
    StitcherConfig,
    FocusAnalyzerConfig,
    PipelineConfig
)
