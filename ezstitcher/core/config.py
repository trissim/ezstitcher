"""
Configuration classes for ezstitcher.

This module contains dataclasses for configuration of different components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from pathlib import Path


@dataclass
class StitcherConfig:
    """Configuration for the Stitcher class."""
    tile_overlap: float = 6.5
    tile_overlap_x: Optional[float] = None
    tile_overlap_y: Optional[float] = None
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
class ImagePreprocessorConfig:
    """Configuration for the ImagePreprocessor class."""
    preprocessing_funcs: Dict[str, Callable] = field(default_factory=dict)
    composite_weights: Optional[Dict[str, float]] = None


@dataclass
class ZStackProcessorConfig:
    """Configuration for the ZStackProcessor class."""
    focus_detect: bool = False
    focus_method: str = "combined"
    create_projections: bool = False
    stitch_z_reference: str = "best_focus"
    save_projections: bool = True
    stitch_all_z_planes: bool = False
    projection_types: List[str] = field(default_factory=lambda: ["max"])


@dataclass
class PlateProcessorConfig:
    """Configuration for the PlateProcessor class."""
    # Basic parameters
    reference_channels: List[str] = field(default_factory=lambda: ["1"])
    well_filter: Optional[List[str]] = None
    use_reference_positions: bool = False

    # File system parameters
    output_dir_suffix: str = "_processed"
    positions_dir_suffix: str = "_positions"
    stitched_dir_suffix: str = "_stitched"
    best_focus_dir_suffix: str = "_best_focus"
    projections_dir_suffix: str = "_Projections"
    timepoint_dir_name: str = "TimePoint_1"

    # Preprocessing parameters
    preprocessing_funcs: Optional[Dict[str, Callable]] = None
    composite_weights: Optional[Dict[str, float]] = None

    # Nested configurations
    stitcher: StitcherConfig = field(default_factory=StitcherConfig)
    focus_analyzer: FocusAnalyzerConfig = field(default_factory=FocusAnalyzerConfig)
    image_preprocessor: ImagePreprocessorConfig = field(default_factory=ImagePreprocessorConfig)
    z_stack_processor: ZStackProcessorConfig = field(default_factory=ZStackProcessorConfig)


# Legacy configs for backward compatibility
@dataclass
class StitchingConfig:
    reference_channels: List[str] = field(default_factory=lambda: ["1"])
    tile_overlap: float = 10.0
    max_shift: int = 50
    focus_detect: bool = False
    focus_method: str = "combined"
    create_projections: bool = False
    stitch_z_reference: str = "best_focus"
    save_projections: bool = True
    stitch_all_z_planes: bool = False
    well_filter: Optional[List[str]] = None
    composite_weights: Optional[Dict] = None
    preprocessing_funcs: Optional[Dict] = None
    margin_ratio: float = 0.1


@dataclass
class ZStackConfig:
    focus_method: str = "combined"
    projection_types: List[str] = field(default_factory=lambda: ["max"])
    select_best_focus: bool = True


@dataclass
class FocusConfig:
    method: str = "combined"
    roi: Optional[List[int]] = None  # [x, y, width, height]


@dataclass
class PlateConfig:
    plate_folder: str = ""
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    zstack: ZStackConfig = field(default_factory=ZStackConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
