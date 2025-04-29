"""
Function-based interface for the EZ module.

This module provides simple function-based interfaces for common stitching workflows.
"""

from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from .core import EZStitcher


def stitch_plate(input_path: Union[str, Path],
                output_path: Optional[Union[str, Path]] = None,
                **kwargs) -> Path:
    """
    One-liner function to stitch a plate of microscopy images.

    Args:
        input_path: Path to the plate folder
        output_path: Path for output (default: input_path + "_stitched")
        **kwargs: Additional options passed to EZStitcher
            normalize (bool): Whether to apply normalization
            flatten_z (bool): Whether to flatten Z-stacks
            z_method (str): Method for Z-flattening
            channel_weights (List[float]): Weights for channel compositing
            well_filter (List[str]): List of wells to process

    Returns:
        Path: Path to the stitched output
    """
    # Pass output_path as a keyword argument
    if output_path is not None:
        kwargs['output_path'] = output_path
    stitcher = EZStitcher(input_path, **kwargs)
    return stitcher.stitch()
