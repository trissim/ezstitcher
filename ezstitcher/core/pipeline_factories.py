"""
Pipeline factory system for the EZStitcher pipeline architecture.

This module contains factory classes that create pre-configured pipelines 
for common workflows, leveraging specialized steps to reduce boilerplate code.

Available factory classes:
- PipelineFactory: Base class for creating pipeline configurations
- BasicPipelineFactory: For single-channel, single-Z stitching
- MultichannelPipelineFactory: For multi-channel stitching
- ZStackPipelineFactory: For Z-stack stitching with projection
- FocusPipelineFactory: For Z-stack stitching with focus selection
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod

from .pipeline import Pipeline
from .steps import Step, PositionGenerationStep, ImageStitchingStep
from .step_factories import ZFlatStep, FocusStep, CompositeStep
from .image_processor import ImageProcessor as IP


class PipelineFactory:
    """Base class for creating pipeline configurations."""

    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        normalize: bool = True,
        normalization_params: Optional[Dict[str, Any]] = None,
        preprocessing_steps: Optional[List[Step]] = None,
        well_filter: Optional[List[str]] = None,
        weights: Optional[Union[List[float], Dict[str, float]]] = None,
        roi: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with common pipeline parameters.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for stitched images
            normalize: Whether to include normalization
            normalization_params: Parameters for normalization
            preprocessing_steps: Additional preprocessing steps
            well_filter: Wells to process
            stitch_original: Whether to stitch original stack
            weights: Weights for channel compositing or focus metrics
            roi: Region of interest for processing
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir.parent / f"{self.input_dir.name}_stitched"
        self.normalize = normalize
        self.normalization_params = normalization_params or {}
        self.preprocessing_steps = preprocessing_steps or []
        self.well_filter = well_filter
        self.weights = weights
        self.roi = roi

    def create_normalization_step(self) -> Optional[Step]:
        """Create normalization step if enabled."""
        if not self.normalize:
            return None
        
        norm_params = self.normalization_params or {
            'low_percentile': 1.0,
            'high_percentile': 99.0
        }
        
        return Step(
            func=(IP.stack_percentile_normalize, norm_params),
            name="Normalization"
        )

    def create_pipelines(self) -> List[Pipeline]:
        """Create pipeline configuration based on parameters."""
        position_steps = []
        
        norm_step = self.create_normalization_step()
        if norm_step:
            position_steps.append(norm_step)

        self._add_specialized_steps(position_steps)
        position_steps.append(PositionGenerationStep())

        return [
            Pipeline(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                steps=position_steps,
                well_filter=self.well_filter,
                name="Position Generation Pipeline"
            ),
            Pipeline(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                steps=[ImageStitchingStep(well_filter=self.well_filter)],
                well_filter=self.well_filter,
                name=self._get_stitching_name()
            )
        ]

    def _add_specialized_steps(self, position_steps: List[Step]) -> None:
        """Add specialized processing steps. Override in subclasses."""
        pass

    def _get_stitching_name(self) -> str:
        """Get name for stitching pipeline."""
        return "Image Stitching Pipeline"


class BasicPipelineFactory(PipelineFactory):
    """Factory for creating basic single-channel, single-Z pipelines."""
    pass


class MultichannelPipelineFactory(PipelineFactory):
    """Factory for creating multi-channel pipelines."""

    def _add_specialized_steps(self, position_steps: List[Step]) -> None:
        """Add channel compositing step."""
        if self.weights:
            position_steps.append(CompositeStep(
                weights=self.weights,
                well_filter=self.well_filter
            ))

    def _get_stitching_name(self) -> str:
        return "Channel-Specific Stitching Pipeline"


class ZStackPipelineFactory(PipelineFactory):
    """Factory for creating Z-stack pipelines with projection."""

    def __init__(
        self,
        method: str = "max",
        method_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize with Z-stack specific parameters."""
        super().__init__(**kwargs)
        self.method = method
        self.method_options = method_options or {'method': 'max'}

    def _add_specialized_steps(self, position_steps: List[Step]) -> None:
        """Add Z-stack processing step."""
        position_steps.append(ZFlatStep(
            method=self.method_options.get('method', 'max'),
            well_filter=self.well_filter
        ))

    def _get_stitching_name(self) -> str:
        return "Z-Stack Stitching Pipeline"


class FocusPipelineFactory(PipelineFactory):
    """Factory for creating focus-based Z-stack pipelines."""

    def __init__(
        self,
        focus_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize with focus-specific parameters."""
        super().__init__(**kwargs)
        self.focus_options = focus_options or {'metric': 'variance_of_laplacian'}

    def _add_specialized_steps(self, position_steps: List[Step]) -> None:
        """Add focus selection step."""
        position_steps.append(FocusStep(
            focus_options=self.focus_options,
            well_filter=self.well_filter
        ))

    def _get_stitching_name(self) -> str:
        return "Focus-Based Stitching Pipeline"
