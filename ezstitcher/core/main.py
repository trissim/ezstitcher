"""
Main module for ezstitcher.

This module provides the main entry point for the ezstitcher package.
"""

import logging
from pathlib import Path

# Import configuration classes
from ezstitcher.core.config import PipelineConfig

# Import the pipeline orchestrator
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

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



def process_plate(
    plate_folder: str | Path,
    config = None,
    **kwargs
) -> bool:
    """
    High-level function to process a plate folder using the PipelineOrchestrator.

    Args:
        plate_folder: Path to the plate folder.
        config: Optional PipelineConfig. If None, a default config is created.
        **kwargs: Optional overrides for config parameters.

    Returns:
        True if processing succeeded, False otherwise.
    """
    plate_folder = Path(plate_folder)

    # Create default config if none provided
    if config is None:
        config = PipelineConfig()
        logging.info("No config provided, using default pipeline configuration")

    # Apply any config overrides in kwargs
    apply_nested_overrides(config, kwargs)

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    return pipeline.run(plate_folder)
