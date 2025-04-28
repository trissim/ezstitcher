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
from .specialized_steps import ZFlatStep, FocusStep, CompositeStep
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
            z_method: Z-stack processing method:
                      - "max", "mean", "median" for projection methods
                      - "combined", "laplacian", "tenengrad" for focus detection methods
            channel_weights: Weights for channel compositing (for reference image only).
                           Should be a list with length equal to the number of channels.
        """
        self.input_dir = Path(input_dir)
        # Only use output_dir if explicitly provided, otherwise let the pipeline handle it
        self.output_dir = Path(output_dir) if output_dir else None

        self.normalize = normalize
        # Default normalization parameters
        self.normalization_params = normalization_params or {
            'low_percentile': 1.0,
            'high_percentile': 99.0
        }
        self.well_filter = well_filter
        self.flatten_z = flatten_z
        self.z_method = z_method
        self.channel_weights = channel_weights

        # Determine if z_method is a focus method or projection method
        self.focus_methods = ["combined", "laplacian", "tenengrad", "normalized_variance", "fft"]
        self.is_focus_method = self.z_method in self.focus_methods

    def create_pipelines(self) -> List[Pipeline]:
        """
        Create pipeline configuration based on parameters.

        This method creates two pipelines:
        1. Position generation pipeline - Creates position files for stitching
        2. Image assembly pipeline - Stitches images using the position files
        """
        # Create position generation pipeline
        pos_pipeline = Pipeline(
            input_dir=self.input_dir,
            steps=[
                # Always include Z-flattening for position generation
                ZFlatStep(method="max"),

                # Include normalization if enabled
                Step(
                    func=(IP.stack_percentile_normalize, self.normalization_params),
                    name="Normalization"
                ) if self.normalize else None,

                # Always include channel compositing for reference image
                CompositeStep(weights=self.channel_weights),

                # Always include position generation
                PositionGenerationStep()
            ],
            well_filter=self.well_filter,
            name="Position Generation Pipeline"
        )

        # Get the position generation step's output directory
        positions_dir = pos_pipeline.steps[-1].output_dir

        # Create assembly pipeline
        assembly_pipeline = Pipeline(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            steps=[
                # Include normalization if enabled (create new instance)
                Step(
                    func=(IP.stack_percentile_normalize, self.normalization_params),
                    name="Normalization"
                ) if self.normalize else None,

                # Include Z-flattening for assembly if enabled
                (FocusStep(
                    focus_options={'metric': self.z_method}
                ) if self.is_focus_method else
                ZFlatStep(
                    method=self.z_method
                )) if self.flatten_z else None,

                # Always include image stitching with explicit positions directory
                ImageStitchingStep(positions_dir=positions_dir)
            ],
            well_filter=self.well_filter,
            name="Image Assembly Pipeline"
        )

        # Return both pipelines
        return [pos_pipeline, assembly_pipeline]
