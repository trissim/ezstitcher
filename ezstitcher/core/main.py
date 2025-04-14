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
from ezstitcher.core.image_locator import ImageLocator

def apply_nested_overrides(config_obj, overrides):
    """
    Recursively apply overrides to a nested config object.

    Args:
        config_obj: The root config object.
        overrides: Dict of overrides, possibly with dot notation keys.
    """
    for key, value in overrides.items():
        parts = key.split(".")
        target = config_obj
        for part in parts[:-1]:
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                # Invalid path, skip
                target = None
                break
        if target is not None and hasattr(target, parts[-1]):
            setattr(target, parts[-1], value)



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

    # Create default config if none provided
    if config is None:
        config = PlateProcessorConfig()
        # Ensure microscope_type is 'auto' for auto-detection
        config.microscope_type = 'auto'
        logging.info("No config provided, using default configuration with auto-detection")

    # Apply any config overrides in kwargs
    apply_nested_overrides(config, kwargs)
    
    processor = PlateProcessor(config)

    #processor._current_plate_folder = plate_folder

    return processor.run(plate_folder)
