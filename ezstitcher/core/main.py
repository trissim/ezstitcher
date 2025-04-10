"""
Main module for ezstitcher.

This module provides the main entry point for the ezstitcher package.
"""

import logging
from pathlib import Path

# Import both dataclass and pydantic configs for backward compatibility
from ezstitcher.core.config import (
    PlateProcessorConfig, StitcherConfig, ZStackProcessorConfig,
    FocusAnalyzerConfig, ImagePreprocessorConfig
)

# Import Pydantic configs for new code
from ezstitcher.core.pydantic_config import (
    PlateProcessorConfig as PydanticPlateProcessorConfig,
    StitcherConfig as PydanticStitcherConfig,
    ZStackProcessorConfig as PydanticZStackProcessorConfig,
    FocusAnalyzerConfig as PydanticFocusAnalyzerConfig,
    ImagePreprocessorConfig as PydanticImagePreprocessorConfig,
    ConfigPresets
)
from ezstitcher.core.plate_processor import PlateProcessor
from ezstitcher.core.zstack_processor import ZStackProcessor
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor
# Removed obsolete import of non-existent ZStackDetector

# ... existing functions (process_plate_folder, etc.) ...

# (existing content up to line 382)
# (content omitted for brevity)

def process_plate_auto(
    plate_folder: str | Path,
    config: PlateProcessorConfig | None = None,
    **kwargs
) -> bool:
    """
    High-level function to process a plate folder.

    Automatically detects if the plate contains Z-stacks and runs the appropriate workflow
    using the modular OOP components internally.

    Args:
        plate_folder: Path to the plate folder.
        config: Optional PlateProcessorConfig. If None, a default config is created.
        **kwargs: Optional overrides for config parameters.

    Returns:
        True if processing succeeded, False otherwise.
    """
    plate_folder = Path(plate_folder)

    # Prepare config
    if config is None:
        config = PlateProcessorConfig()

    # Override config with kwargs if provided
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Instantiate PlateProcessor
    processor = PlateProcessor(config)

    # Detect if Z-stacks are present
    detector = ZStackProcessor(config.z_stack_processor)
    has_zstack, _ = detector.detect_zstack_images(plate_folder / "TimePoint_1")

    if has_zstack:
        logging.info("Z-stacks detected. Running full Z-stack processing pipeline.")
        # Run full Z-stack pipeline
        success = processor.run(plate_folder)
    else:
        logging.info("No Z-stacks detected. Running standard 2D stitching pipeline.")
        # Optionally adjust config to skip Z-stack steps
        config.z_stack_processor.stitch_all_z_planes = False
        processor = PlateProcessor(config)
        success = processor.run(plate_folder)

    return success
