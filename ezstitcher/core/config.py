"""
Configuration classes for ezstitcher.

This module contains dataclasses for configuration of different components.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class StitcherConfig:
    """Configuration for the Stitcher class."""
    tile_overlap: float = 10.0
    max_shift: int = 50
    margin_ratio: float = 0.1
    pixel_size: float = 1.0


# FocusAnalyzerConfig has been removed in favor of direct parameters to FocusAnalyzer


@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""
    # Directory configuration
    out_dir_suffix: str = "_out"  # Default suffix for processing steps
    positions_dir_suffix: str = "_positions"  # Suffix for position generation step
    stitched_dir_suffix: str = "_stitched"  # Suffix for stitching step

    # Processing configuration
    num_workers: int = 1
    well_filter: Optional[List[str]] = None

    # Stitching configuration
    stitcher: StitcherConfig = field(default_factory=StitcherConfig)
