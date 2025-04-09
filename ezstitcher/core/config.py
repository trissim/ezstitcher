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
    # New primary parameters
    z_reference_function: Union[str, Callable[[List[Any]], Any]] = "max_projection"
    save_reference: bool = True
    stitch_all_z_planes: bool = False
    additional_projections: Optional[List[str]] = None
    focus_method: str = "combined"
    focus_config: FocusAnalyzerConfig = field(default_factory=lambda: FocusAnalyzerConfig())

    # Deprecated parameters (kept for backward compatibility)
    reference_method: Optional[Union[str, Callable[[List[Any]], Any]]] = None
    focus_detect: Optional[bool] = None
    stitch_z_reference: Optional[Union[str, Callable[[List[Any]], Any]]] = None
    create_projections: Optional[bool] = None
    save_projections: Optional[bool] = None
    projection_types: Optional[List[str]] = None

    def __post_init__(self):
        """Handle backward compatibility and parameter validation."""
        # Handle deprecated parameters

        # First, handle reference_method if it's set (from previous version)
        if self.reference_method is not None:
            if isinstance(self.reference_method, str):
                if self.reference_method == "best_focus":
                    self.z_reference_function = "best_focus"
                elif self.reference_method == "max_projection":
                    self.z_reference_function = "max_projection"
                elif self.reference_method == "mean_projection":
                    self.z_reference_function = "mean_projection"
                else:
                    raise ValueError(f"Unknown reference_method: {self.reference_method}")
            elif callable(self.reference_method):
                self.z_reference_function = self.reference_method

        # Then handle older stitch_z_reference and focus_detect parameters
        if self.focus_detect is not None or self.stitch_z_reference is not None:
            # Only override z_reference_function if at least one deprecated parameter is explicitly set
            if self.stitch_z_reference is not None:
                if self.stitch_z_reference == "best_focus":
                    self.z_reference_function = "best_focus"
                elif self.stitch_z_reference == "max":
                    self.z_reference_function = "max_projection"
                elif self.stitch_z_reference == "mean":
                    self.z_reference_function = "mean_projection"
                elif callable(self.stitch_z_reference):
                    self.z_reference_function = self.stitch_z_reference

            # If focus_detect is True and stitch_z_reference is not set, use best_focus
            if self.focus_detect is True and self.stitch_z_reference is None:
                self.z_reference_function = "best_focus"

        # Handle deprecated create_projections and save_projections
        if self.create_projections is not None:
            self.save_reference = self.create_projections
        if self.save_projections is not None:
            self.save_reference = self.save_projections

        # Handle deprecated projection_types
        if self.projection_types is not None:
            self.additional_projections = self.projection_types

        # If additional_projections is None, use default value for internal processing
        if self.additional_projections is None:
            self.additional_projections = ["max"]

        # Set deprecated parameters for backward compatibility
        if isinstance(self.z_reference_function, str):
            if self.z_reference_function == "max_projection":
                self.reference_method = "max_projection"
                self.stitch_z_reference = "max"
                self.focus_detect = False
            elif self.z_reference_function == "mean_projection":
                self.reference_method = "mean_projection"
                self.stitch_z_reference = "mean"
                self.focus_detect = False
            elif self.z_reference_function == "best_focus":
                self.reference_method = "best_focus"
                self.stitch_z_reference = "best_focus"
                self.focus_detect = True
        elif callable(self.z_reference_function):
            self.reference_method = self.z_reference_function
            self.stitch_z_reference = self.z_reference_function
            self.focus_detect = False

        self.create_projections = self.save_reference
        self.save_projections = self.save_reference
        self.projection_types = self.additional_projections


@dataclass
class PlateProcessorConfig:
    """Configuration for the PlateProcessor class."""
    # Basic parameters
    reference_channels: List[str] = field(default_factory=lambda: ["1"])
    well_filter: Optional[List[str]] = None
    use_reference_positions: bool = False

    # Microscope type - can be 'auto', 'ImageXpress', 'OperaPhenix', etc.
    microscope_type: str = 'auto'

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
