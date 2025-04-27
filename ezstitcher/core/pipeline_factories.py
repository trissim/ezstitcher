"""
Pipeline factory system for the EZStitcher pipeline architecture.

This module contains the AutoPipelineFactory class that creates pre-configured pipelines
for all common workflows, leveraging specialized steps to reduce boilerplate code.

The AutoPipelineFactory uses a unified approach that handles 2D multichannel, z-stack per plane stitch,
and z-stack projection stitch with a single implementation, simplifying the pipeline architecture.
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from .pipeline import Pipeline
from .steps import Step, PositionGenerationStep, ImageStitchingStep
from .step_factories import ZFlatStep, CompositeStep
from .image_processor import ImageProcessor as IP


class AutoPipelineFactory:
    """
    Unified factory for creating pipelines for all common use cases.

    This factory handles all types of stitching workflows with a single implementation:
    - 2D multichannel stitching
    - Z-stack per plane stitching
    - Z-stack projection stitching

    It automatically configures the appropriate steps based on the input parameters,
    with no need to differentiate between different types of pipelines.
    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        normalize: bool = True,
        normalization_params: Optional[Dict[str, Any]] = None,
        well_filter: Optional[List[str]] = None,
        flatten_z: bool = False,
        z_method: str = "max",
        channel_weights: Optional[Union[List[float], Dict[str, float]]] = None,
    ):
        """
        Initialize with pipeline parameters.

        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for stitched images
            normalize: Whether to include normalization
            normalization_params: Parameters for normalization
            well_filter: Wells to process
            flatten_z: Whether to flatten Z-stacks (if Z-stacks are present)
            z_method: Z-stack flattening method ("max", "mean", "median", etc.)
            channel_weights: Weights for channel compositing (for reference image only).
                           Should be a list with length equal to the number of channels.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir.parent / f"{self.input_dir.name}_stitched"
        self.normalize = normalize
        self.normalization_params = normalization_params or {'low_percentile': 1.0, 'high_percentile': 99.0}
        self.well_filter = well_filter
        self.flatten_z = flatten_z
        self.z_method = z_method
        self.channel_weights = channel_weights

    def create_pipelines(self) -> List[Pipeline]:
        """
        Create pipeline configuration based on parameters.

        This method creates two pipelines:
        1. Position generation pipeline - Creates position files for stitching
        2. Image assembly pipeline - Stitches images using the position files

        The method automatically configures the appropriate steps based on the input parameters,
        handling 2D multichannel, z-stack per plane, and z-stack projection stitching with a
        single unified implementation.
        """
        # Create steps for reuse
        norm_step = Step(
            func=(IP.stack_percentile_normalize, self.normalization_params),
            name="Normalization"
        ) if self.normalize else None

        z_flat_step = ZFlatStep(
            method=self.z_method,
            well_filter=self.well_filter
        ) if self.flatten_z else None

        # Position generation pipeline with hardwired order:
        # [flatten Z, normalize, create_composite, generate positions]
        position_steps = []
        if z_flat_step:
            position_steps.append(z_flat_step)
        if norm_step:
            position_steps.append(norm_step)

        # Always include CompositeStep for channel compositing (for reference image)
        position_steps.append(CompositeStep(
            weights=self.channel_weights,
            well_filter=self.well_filter
        ))

        # Always include PositionGenerationStep
        position_steps.append(PositionGenerationStep())

        # Create position generation pipeline
        pos_pipeline = Pipeline(
            input_dir=self.input_dir,
            steps=position_steps,
            well_filter=self.well_filter,
            name="Position Generation Pipeline"
        )

        # Create image assembly pipeline (with output_dir)
        assembly_pipeline = Pipeline(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            steps=[
                step for step in [
                    norm_step,
                    z_flat_step,
                    ImageStitchingStep(well_filter=self.well_filter)
                ] if step is not None
            ],
            well_filter=self.well_filter,
            name="Image Assembly Pipeline"  # Use a single name for all pipeline types
        )

        # Return both pipelines
        return [pos_pipeline, assembly_pipeline]
