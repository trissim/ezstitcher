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
        focus_detect: Whether to enable focus detection for Z-stacks
        focus_method: Focus detection method to use
        create_projections: Whether to create projections from Z-stacks
        stitch_z_reference: Z-plane to use for stitching ('best_focus', 'max', 'mean', or custom function)
        save_projections: Whether to save projection images
        stitch_all_z_planes: Whether to stitch all Z-planes using reference positions
        projection_types: Types of projections to create
    """
    focus_detect: bool = Field(False, description="Enable focus detection for Z-stacks")
    focus_method: str = Field("combined", description="Focus detection method to use")
    create_projections: bool = Field(False, description="Create projections from Z-stacks")
    stitch_z_reference: Union[str, Callable[[List[Any]], Any]] = Field("max",
                                                                      description="Z-plane to use for stitching")
    save_projections: bool = Field(True, description="Save projection images")
    stitch_all_z_planes: bool = Field(False, description="Stitch all Z-planes using reference positions")
    projection_types: List[str] = Field(default_factory=lambda: ["max"],
                                       description="Types of projections to create")

    @validator('focus_method')
    def validate_focus_method(cls, v):
        """Validate that the focus method is supported."""
        valid_methods = ["combined", "laplacian", "normalized_variance", "tenengrad", "fft", "adaptive_fft"]
        if v not in valid_methods:
            raise ValueError(f"Focus method must be one of {valid_methods}, got {v}")
        return v

    @validator('projection_types')
    def validate_projection_types(cls, v):
        """Validate that the projection types are supported."""
        valid_types = ["max", "mean", "std", "median", "min"]
        for proj_type in v:
            if proj_type not in valid_types:
                raise ValueError(f"Projection type must be one of {valid_types}, got {proj_type}")
        return v

    @validator('stitch_z_reference')
    def validate_stitch_z_reference(cls, v):
        """Validate that the stitch_z_reference is valid."""
        if isinstance(v, str):
            valid_refs = ["max", "mean", "best_focus", "median", "min"]
            if v not in valid_refs:
                raise ValueError(f"stitch_z_reference must be one of {valid_refs} or a callable, got {v}")
        return v

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
        config_dict = self.model_dump(exclude={'preprocessing_funcs', 'stitch_z_reference'})

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save the configuration to a YAML file.

        Args:
            path: Path to save the YAML file
        """
        # Convert to dict, excluding callable objects
        config_dict = self.model_dump(exclude={'preprocessing_funcs', 'stitch_z_reference'})

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
                focus_detect=True,
                focus_method="combined",
                create_projections=True,
                stitch_z_reference="best_focus",
                save_projections=True,
                projection_types=["max", "mean"]
            )
        )

    @staticmethod
    def z_stack_per_plane() -> PlateProcessorConfig:
        """Configuration for Z-stack processing with per-plane stitching."""
        return PlateProcessorConfig(
            z_stack_processor=ZStackProcessorConfig(
                create_projections=True,
                stitch_z_reference="max",
                save_projections=True,
                stitch_all_z_planes=True,
                projection_types=["max"]
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
