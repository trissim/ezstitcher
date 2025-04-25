"""EZStitcher: An easy-to-use microscopy image stitching and processing tool."""

__version__ = "0.1.1"  # Update version number

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
