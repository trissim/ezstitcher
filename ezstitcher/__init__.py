"""EZStitcher - Easy Microscopy Image Stitching Tool.

EZStitcher is a Python package for stitching microscopy images with support for
Z-stacks, multi-channel fluorescence, and advanced focus detection.
"""

__version__ = "0.1.0"

# Import main functions for backward compatibility
from ezstitcher.core import (
    process_plate_folder,
    modified_process_plate_folder,
    process_bf,
    find_best_focus,
    process_plate_folder_with_config
)

# Import classes for instance-based API
from ezstitcher.core import (
    ImagePreprocessor,
    FocusAnalyzer,
    ZStackProcessor,
    Stitcher,
    PlateProcessor
)

# Import configuration classes
from ezstitcher.core import (
    StitcherConfig,
    FocusAnalyzerConfig,
    ImagePreprocessorConfig,
    ZStackProcessorConfig,
    PlateProcessorConfig
)

# Import Pydantic configuration classes
from ezstitcher.core import (
    PydanticPlateProcessorConfig,
    PydanticStitcherConfig,
    PydanticZStackProcessorConfig,
    PydanticFocusAnalyzerConfig,
    PydanticImagePreprocessorConfig,
    ConfigPresets
)