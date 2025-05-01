"""
Configuration classes for ezstitcher.

This module contains dataclasses for configuration of different components.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path
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

    # Microscope configuration
    force_parser: Optional[str] = None  # Force a specific parser type (e.g., "OperaPhenix")

    # Intermediate storage configuration
    storage_mode: Literal["legacy", "memory", "zarr"] = "legacy"
    """Mode for storing intermediate pipeline results ('legacy', 'memory', 'zarr').
    'legacy': Default, uses existing in-memory dict within Pipeline.
    'memory': Uses MemoryStorageAdapter (persists .npy on completion).
    'zarr': Uses ZarrStorageAdapter (persists to disk immediately).
    """
    storage_root: Optional[Path] = None
    """Root directory for storage backends that require it (e.g., Zarr).
    If None and required (like for 'zarr' mode), behavior might depend on implementation
    (e.g., error or default temp location). For 'memory' mode, this path might be
    used as the default target for the final 'persist' operation if not otherwise specified.
    """

    def copy(self):
        """Create a deep copy of this configuration object."""
        import copy
        return copy.deepcopy(self)
