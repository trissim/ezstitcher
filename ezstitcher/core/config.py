"""
Configuration classes for ezstitcher.

This module contains dataclasses for configuration of different components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class StitcherConfig:
    """Configuration for the Stitcher class."""
    tile_overlap: float = 10
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
    #"""Configuration for Z-stack processing."""
    #focus_method: str = "combined"  # Method for focus detection
    #focus_config: Optional[FocusAnalyzerConfig] = None
    #projection_method: str = "max"  # Method for z-stack projection (max, mean, best_focus, none)
    """
    Configuration for the ZStackProcessor class.

    Attributes:
        z_reference_function: Function that converts a 3D stack to a 2D image.
            Can be a string name of a standard function or a callable.
            Standard functions: "max_projection", "mean_projection", "best_focus".
            Can also be a custom function that takes a Z-stack and returns a 2D image.
        save_reference: Whether to save the reference image.
        stitch_all_z_planes: Whether to stitch all Z-planes using reference positions.
        additional_projections: Types of additional projections to create.
        focus_method: Focus detection method to use when using best_focus.
        focus_config: Configuration for the FocusAnalyzer.

        # Deprecated parameters (kept for backward compatibility)
        reference_method: Deprecated. Use z_reference_function instead.
        focus_detect: Deprecated. Use z_reference_function="best_focus" instead.
        stitch_z_reference: Deprecated. Use z_reference_function instead.
        create_projections: Deprecated. Use save_reference instead.
        save_projections: Deprecated. Use save_reference instead.
        projection_types: Deprecated. Use additional_projections instead.
    """

#    reference_flatten="max"
#    stitch_flatten=None

    # Configuration for Z-stack processing
    reference_flatten: Union[str, Callable[[List[Any]], Any]] = "max_projection"
    stitch_flatten: Optional[Union[str, Callable[[List[Any]], Any]]] = None
    save_reference: bool = True
    additional_projections: Optional[List[str]] = None
    focus_method: str = "combined"
    focus_config: FocusAnalyzerConfig = field(default_factory=FocusAnalyzerConfig)

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Validate reference_flatten
        if isinstance(self.reference_flatten, str):
            valid_methods = ["max_projection", "mean_projection", "best_focus"]
            if self.reference_flatten not in valid_methods:
                logger.warning("Invalid reference_flatten method: %s, using max_projection", self.reference_flatten)
                self.reference_flatten = "max_projection"

        # Validate stitch_flatten
        if self.stitch_flatten is not None:
            if isinstance(self.stitch_flatten, str):
                valid_methods = ["max_projection", "mean_projection", "best_focus"]
                if self.stitch_flatten not in valid_methods:
                    logger.warning("Invalid stitch_flatten method: %s, using max_projection", self.stitch_flatten)
                    self.stitch_flatten = "max_projection"


@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""
    # Input/output configuration
    processed_dir_suffix: str = "_processed"
    post_processed_dir_suffix: str = "_post_processed"
    positions_dir_suffix: str = "_positions"
    stitched_dir_suffix: str = "_stitched"

    # Well filtering
    well_filter: Optional[List[str]] = None

    # Reference processing (for position generation)
    reference_channels: List[str] = field(default_factory=lambda: ["1"])
    reference_preprocessing: Optional[Dict[str, Callable]] = None
    reference_composite_weights: Optional[Dict[str, float]] = None

    # Final processing (for stitched output)
    # Note: All available channels are always processed and stitched
    final_preprocessing: Optional[Dict[str, Callable]] = None
    final_composite_weights: Optional[Dict[str, float]] = None

    # Stitching configuration
    stitcher: StitcherConfig = field(default_factory=StitcherConfig)

    # Z-stack processing configuration
    zstack_config: ZStackProcessorConfig = field(default_factory=ZStackProcessorConfig)


@dataclass
class PlateProcessorConfig:
    """Configuration for the PlateProcessor class."""
    # Basic parameters
    reference_channels: List[str] = field(default_factory=lambda: ["1"])
    well_filter: Optional[List[str]] = None
    use_reference_positions: bool = False

    # Microscope type - can be 'auto', 'ImageXpress', 'OperaPhenix', etc.
    microscope_type: str = 'auto'

    # File renaming parameters
    rename_files: bool = True
    padding_width: int = 3
    dry_run: bool = False

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


# ZStackConfig has been removed as it's been replaced by ZStackProcessorConfig


@dataclass
class FocusConfig:
    method: str = "combined"
    roi: Optional[List[int]] = None  # [x, y, width, height]


@dataclass
class PlateConfig:
    plate_folder: str = ""
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    zstack: ZStackProcessorConfig = field(default_factory=ZStackProcessorConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
