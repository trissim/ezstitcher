"""
Configuration classes for ezstitcher.

This module contains dataclasses for configuration of different components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class StitcherConfig:
    """Configuration for the Stitcher class."""
    tile_overlap: float = 10.0
    max_shift: int = 50
    margin_ratio: float = 0.1
    pixel_size: float = 1.0


@dataclass
class FocusAnalyzerConfig:
    """Configuration for the FocusAnalyzer class."""
    method: str = "combined"
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    weights: Optional[Dict[str, float]] = None


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
    focus_config: FocusAnalyzerConfig = field(default_factory=FocusAnalyzerConfig)
