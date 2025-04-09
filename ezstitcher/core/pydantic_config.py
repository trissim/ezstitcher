"""
Pydantic configuration models for ezstitcher.

This module contains Pydantic models for configuration of different components,
providing validation, serialization, and hierarchical configuration management.
"""

from typing import Dict, List, Optional, Union, Callable, Any, Tuple, ClassVar
from pathlib import Path
import json
import yaml
from pydantic import BaseModel, Field, validator, model_validator


class StitcherConfig(BaseModel):
    """
    Configuration for the Stitcher class.

    Attributes:
        tile_overlap: Default percentage overlap between tiles (applied to both x and y if specific values not provided)
        tile_overlap_x: Specific horizontal overlap percentage (overrides tile_overlap for x-axis)
        tile_overlap_y: Specific vertical overlap percentage (overrides tile_overlap for y-axis)
        max_shift: Maximum shift allowed between tiles in pixels
        margin_ratio: Blending margin ratio for stitching (0.0-1.0)
        pixel_size: Pixel size in microns
    """
    tile_overlap: float = Field(6.5, description="Default percentage overlap between tiles")
    tile_overlap_x: Optional[float] = Field(None, description="Specific horizontal overlap percentage")
    tile_overlap_y: Optional[float] = Field(None, description="Specific vertical overlap percentage")
    max_shift: int = Field(50, description="Maximum shift allowed between tiles in pixels")
    margin_ratio: float = Field(0.1, description="Blending margin ratio for stitching (0.0-1.0)")
    pixel_size: float = Field(1.0, description="Pixel size in microns")

    @validator('margin_ratio')
    def validate_margin_ratio(cls, v):
        """Validate that margin_ratio is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"margin_ratio must be between 0 and 1, got {v}")
        return v

    @validator('tile_overlap', 'tile_overlap_x', 'tile_overlap_y')
    def validate_overlap(cls, v):
        """Validate that overlap percentages are reasonable."""
        if v is not None and not 0 <= v <= 50:
            raise ValueError(f"Tile overlap should be between 0 and 50 percent, got {v}")
        return v


class FocusAnalyzerConfig(BaseModel):
    """
    Configuration for the FocusAnalyzer class.

    Attributes:
        method: Focus detection method to use
        roi: Optional region of interest (x, y, width, height)
        weights: Optional weights for different focus metrics when using 'combined' method
    """
    method: str = Field("combined", description="Focus detection method to use")
    roi: Optional[Tuple[int, int, int, int]] = Field(None, description="Region of interest (x, y, width, height)")
    weights: Optional[Dict[str, float]] = Field(None, description="Weights for different focus metrics")

    @validator('method')
    def validate_method(cls, v):
        """Validate that the focus method is supported."""
        valid_methods = ["combined", "laplacian", "normalized_variance", "tenengrad", "fft", "adaptive_fft"]
        if v not in valid_methods:
            raise ValueError(f"Focus method must be one of {valid_methods}, got {v}")
        return v


class ImagePreprocessorConfig(BaseModel):
    """
    Configuration for the ImagePreprocessor class.

    Attributes:
        preprocessing_funcs: Dictionary mapping channel/wavelength to preprocessing function
        composite_weights: Optional weights for creating composite images
    """
    preprocessing_funcs: Dict[str, Callable] = Field(default_factory=dict,
                                                    description="Preprocessing functions by channel")
    composite_weights: Optional[Dict[str, float]] = Field(None,
                                                         description="Weights for creating composite images")

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class ZStackProcessorConfig(BaseModel):
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

        # Deprecated parameters (kept for backward compatibility)
        reference_method: Deprecated. Use z_reference_function instead.
        focus_detect: Deprecated. Use z_reference_function="best_focus" instead.
        stitch_z_reference: Deprecated. Use z_reference_function instead.
        create_projections: Deprecated. Use save_reference instead.
        save_projections: Deprecated. Use save_reference instead.
        projection_types: Deprecated. Use additional_projections instead.
    """
    # New primary parameters
    z_reference_function: Union[str, Callable[[List[Any]], Any]] = Field(
        "max_projection",
        description="Function that converts a 3D stack to a 2D image"
    )
    save_reference: bool = Field(
        True,
        description="Whether to save the reference image"
    )
    stitch_all_z_planes: bool = Field(
        False,
        description="Whether to stitch all Z-planes using reference positions"
    )
    additional_projections: List[str] = Field(
        default_factory=lambda: ["max"],
        description="Types of additional projections to create"
    )
    focus_method: str = Field(
        "combined",
        description="Focus detection method to use when using best_focus"
    )

    # Deprecated parameters (kept for backward compatibility)
    reference_method: Optional[Union[str, Callable[[List[Any]], Any]]] = Field(
        None,
        description="Deprecated. Use z_reference_function instead"
    )
    focus_detect: Optional[bool] = Field(
        None,
        description="Deprecated. Use z_reference_function='best_focus' instead"
    )
    stitch_z_reference: Optional[Union[str, Callable[[List[Any]], Any]]] = Field(
        None,
        description="Deprecated. Use z_reference_function instead"
    )
    create_projections: Optional[bool] = Field(
        None,
        description="Deprecated. Use save_reference instead"
    )
    save_projections: Optional[bool] = Field(
        None,
        description="Deprecated. Use save_reference instead"
    )
    projection_types: Optional[List[str]] = Field(
        None,
        description="Deprecated. Use additional_projections instead"
    )

    @validator('focus_method')
    def validate_focus_method(cls, v):
        """Validate that the focus method is supported."""
        valid_methods = ["combined", "laplacian", "normalized_variance", "tenengrad", "fft", "adaptive_fft"]
        if v not in valid_methods:
            raise ValueError(f"Focus method must be one of {valid_methods}, got {v}")
        return v

    @validator('additional_projections')
    def validate_additional_projections(cls, v):
        """Validate that the projection types are supported."""
        valid_types = ["max", "mean", "std", "median", "min"]
        for proj_type in v:
            if proj_type not in valid_types:
                raise ValueError(f"Projection type must be one of {valid_types}, got {proj_type}")
        return v

    @validator('z_reference_function')
    def validate_z_reference_function(cls, v):
        """Validate that the z_reference_function is valid."""
        if isinstance(v, str):
            valid_refs = ["max_projection", "mean_projection", "best_focus"]
            if v not in valid_refs:
                raise ValueError(f"z_reference_function must be one of {valid_refs} or a callable, got {v}")
        return v

    @validator('projection_types')
    def validate_projection_types(cls, v):
        """Validate that the projection types are supported."""
        if v is not None:
            valid_types = ["max", "mean", "std", "median", "min"]
            for proj_type in v:
                if proj_type not in valid_types:
                    raise ValueError(f"Projection type must be one of {valid_types}, got {proj_type}")
        return v

    @validator('reference_method')
    def validate_reference_method(cls, v):
        """Validate that the reference_method is valid."""
        if v is not None and isinstance(v, str):
            valid_refs = ["max_projection", "mean_projection", "best_focus"]
            if v not in valid_refs:
                raise ValueError(f"reference_method must be one of {valid_refs} or a callable, got {v}")
        return v

    @validator('stitch_z_reference')
    def validate_stitch_z_reference(cls, v):
        """Validate that the stitch_z_reference is valid."""
        if v is not None and isinstance(v, str):
            valid_refs = ["max", "mean", "best_focus", "median", "min"]
            if v not in valid_refs:
                raise ValueError(f"stitch_z_reference must be one of {valid_refs} or a callable, got {v}")
        return v

    @model_validator(mode='after')
    def handle_deprecated_params(self):
        """Handle backward compatibility between old and new parameters."""
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

        return self

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class PlateProcessorConfig(BaseModel):
    """
    Configuration for the PlateProcessor class.

    This is the main configuration class that contains nested configurations for all components.

    Attributes:
        reference_channels: List of channels to use as reference for alignment
        well_filter: Optional list of wells to process
        use_reference_positions: Whether to use existing reference positions
        output_dir_suffix: Suffix for the output directory
        positions_dir_suffix: Suffix for the positions directory
        stitched_dir_suffix: Suffix for the stitched directory
        best_focus_dir_suffix: Suffix for the best focus directory
        projections_dir_suffix: Suffix for the projections directory
        timepoint_dir_name: Name of the timepoint directory
        preprocessing_funcs: Optional preprocessing functions by channel
        composite_weights: Optional weights for creating composite images
        stitcher: Configuration for the Stitcher
        focus_analyzer: Configuration for the FocusAnalyzer
        image_preprocessor: Configuration for the ImagePreprocessor
        z_stack_processor: Configuration for the ZStackProcessor
    """
    # Basic parameters
    reference_channels: List[str] = Field(default_factory=lambda: ["1"],
                                         description="Channels to use as reference")
    well_filter: Optional[List[str]] = Field(None, description="List of wells to process")
    use_reference_positions: bool = Field(False, description="Use existing reference positions")

    # Microscope type - can be 'auto', 'ImageXpress', 'OperaPhenix', etc.
    microscope_type: str = Field('auto', description="Type of microscope ('auto', 'ImageXpress', 'OperaPhenix')")

    # File system parameters
    output_dir_suffix: str = Field("_processed", description="Suffix for the output directory")
    positions_dir_suffix: str = Field("_positions", description="Suffix for the positions directory")
    stitched_dir_suffix: str = Field("_stitched", description="Suffix for the stitched directory")
    best_focus_dir_suffix: str = Field("_best_focus", description="Suffix for the best focus directory")
    projections_dir_suffix: str = Field("_projections", description="Suffix for the projections directory")
    timepoint_dir_name: str = Field("TimePoint_1", description="Name of the timepoint directory")

    # Preprocessing parameters
    preprocessing_funcs: Optional[Dict[str, Callable]] = Field(None,
                                                              description="Preprocessing functions by channel")
    composite_weights: Optional[Dict[str, float]] = Field(None,
                                                         description="Weights for creating composite images")

    # Nested configurations
    stitcher: StitcherConfig = Field(default_factory=StitcherConfig,
                                    description="Configuration for the Stitcher")
    focus_analyzer: FocusAnalyzerConfig = Field(default_factory=FocusAnalyzerConfig,
                                              description="Configuration for the FocusAnalyzer")
    image_preprocessor: ImagePreprocessorConfig = Field(default_factory=ImagePreprocessorConfig,
                                                      description="Configuration for the ImagePreprocessor")
    z_stack_processor: ZStackProcessorConfig = Field(default_factory=ZStackProcessorConfig,
                                                   description="Configuration for the ZStackProcessor")

    @validator('reference_channels')
    def validate_reference_channels(cls, v):
        """Validate that reference channels are not empty."""
        if not v:
            raise ValueError("reference_channels cannot be empty")
        return v

    @model_validator(mode='after')
    def validate_config(self):
        """Validate the entire configuration for consistency."""
        # Ensure focus_method is consistent between focus_analyzer and z_stack_processor
        focus_method = self.focus_analyzer.method
        z_focus_method = self.z_stack_processor.focus_method

        if focus_method != z_focus_method:
            self.z_stack_processor.focus_method = focus_method

        return self

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

    def to_json(self, path: Union[str, Path]) -> None:
        """
        Save the configuration to a JSON file.

        Args:
            path: Path to save the JSON file
        """
        # Convert to dict, excluding callable objects
        config_dict = self.model_dump(exclude={'preprocessing_funcs', 'stitch_z_reference', 'z_reference_function'})

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save the configuration to a YAML file.

        Args:
            path: Path to save the YAML file
        """
        # Convert to dict, excluding callable objects
        config_dict = self.model_dump(exclude={'preprocessing_funcs', 'stitch_z_reference', 'z_reference_function'})

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'PlateProcessorConfig':
        """
        Load the configuration from a JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            PlateProcessorConfig: Loaded configuration
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'PlateProcessorConfig':
        """
        Load the configuration from a YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            PlateProcessorConfig: Loaded configuration
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)


# Example configuration presets
class ConfigPresets:
    """Predefined configuration presets for common use cases."""

    @staticmethod
    def default() -> PlateProcessorConfig:
        """Default configuration for general use."""
        return PlateProcessorConfig()

    @staticmethod
    def z_stack_best_focus() -> PlateProcessorConfig:
        """Configuration for Z-stack processing with best focus detection."""
        return PlateProcessorConfig(
            z_stack_processor=ZStackProcessorConfig(
                z_reference_function="best_focus",
                focus_method="combined",
                save_reference=True,
                additional_projections=["max", "mean"]
            )
        )

    @staticmethod
    def z_stack_per_plane() -> PlateProcessorConfig:
        """Configuration for Z-stack processing with per-plane stitching."""
        return PlateProcessorConfig(
            z_stack_processor=ZStackProcessorConfig(
                z_reference_function="max_projection",
                save_reference=True,
                stitch_all_z_planes=True,
                additional_projections=["max"]
            )
        )

    @staticmethod
    def high_resolution() -> PlateProcessorConfig:
        """Configuration for high-resolution stitching."""
        return PlateProcessorConfig(
            stitcher=StitcherConfig(
                tile_overlap=10.0,
                max_shift=100,
                margin_ratio=0.15,
                pixel_size=0.5
            )
        )
