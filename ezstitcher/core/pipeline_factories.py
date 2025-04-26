"""
Pipeline factory functions for the EZStitcher pipeline architecture.

This module provides concise factory functions that create pre-configured pipelines
for common microscopy image stitching workflows.
"""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path

from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.step_factories import ZFlatStep, FocusStep, CompositeStep
from ezstitcher.core.image_processor import ImageProcessor as IP


def create_basic_pipeline(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    normalize: bool = True,
    well_filter: Optional[List[str]] = None
) -> List[Pipeline]:
    """
    Create a basic single-channel, single-Z stitching pipeline.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for stitched images
        normalize: Whether to include normalization (default: True)
        well_filter: Wells to process
        
    Returns:
        List of pipelines: [position_pipeline, stitching_pipeline]
    """
    # Convert input_dir and output_dir to Path objects if they're strings
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if output_dir is not None and isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    # Create position generation pipeline
    position_steps = []
    
    # Add normalization if requested
    if normalize:
        position_steps.append(
            Step(
                func=(IP.stack_percentile_normalize, 
                      {'low_percentile': 0.1, 'high_percentile': 99.9}),
                name="Normalization"
            )
        )
    
    # Add position generation step
    position_steps.append(PositionGenerationStep())
    
    # Create pipelines
    position_pipeline = Pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        steps=position_steps,
        name="Position Generation",
        well_filter=well_filter
    )
    
    stitching_pipeline = Pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        steps=[ImageStitchingStep()],
        name="Image Stitching",
        well_filter=well_filter
    )
    
    return [position_pipeline, stitching_pipeline]


def create_multichannel_pipeline(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    weights: Optional[List[float]] = None,
    normalize: bool = True,
    stitch_channels_separately: bool = False,
    well_filter: Optional[List[str]] = None
) -> List[Pipeline]:
    """
    Create a multi-channel stitching pipeline.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for stitched images
        weights: Channel weights for compositing (default: equal weights)
        normalize: Whether to include normalization (default: True)
        stitch_channels_separately: Whether to stitch each channel separately
        well_filter: Wells to process
        
    Returns:
        List of pipelines
    """
    # Convert input_dir and output_dir to Path objects if they're strings
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if output_dir is not None and isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    # Create position generation pipeline
    position_steps = []
    
    # Add normalization if requested
    if normalize:
        position_steps.append(
            Step(
                func=(IP.stack_percentile_normalize, 
                      {'low_percentile': 0.1, 'high_percentile': 99.9}),
                name="Normalization"
            )
        )
    
    # Add channel compositing step
    position_steps.append(CompositeStep(weights=weights))
    
    # Add position generation step
    position_steps.append(PositionGenerationStep())
    
    # Create position pipeline
    position_pipeline = Pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        steps=position_steps,
        name="Position Generation",
        well_filter=well_filter
    )
    
    pipelines = [position_pipeline]
    
    # Create stitching pipeline(s)
    if stitch_channels_separately:
        channel_pipeline = Pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            steps=[
                Step(
                    func=lambda x: x,
                    variable_components=['channel'],
                    name="Channel Selector"
                ),
                ImageStitchingStep()
            ],
            name="Channel-Specific Stitching",
            well_filter=well_filter
        )
        pipelines.append(channel_pipeline)
    else:
        stitching_pipeline = Pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            steps=[ImageStitchingStep()],
            name="Composite Stitching",
            well_filter=well_filter
        )
        pipelines.append(stitching_pipeline)
    
    return pipelines


def create_zstack_pipeline(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    method: str = "max",
    normalize: bool = True,
    stitch_original: bool = False,
    well_filter: Optional[List[str]] = None
) -> List[Pipeline]:
    """
    Create a Z-stack stitching pipeline with projection.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for stitched images
        method: Projection method ("max", "mean", "median", etc.)
        normalize: Whether to include normalization (default: True)
        stitch_original: Whether to stitch original Z-stack
        well_filter: Wells to process
        
    Returns:
        List of pipelines
    """
    # Convert input_dir and output_dir to Path objects if they're strings
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if output_dir is not None and isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    # Create position generation pipeline
    position_steps = []
    
    # Add Z-stack flattening step
    position_steps.append(ZFlatStep(method=method))
    
    # Add normalization if requested
    if normalize:
        position_steps.append(
            Step(
                func=(IP.stack_percentile_normalize, 
                      {'low_percentile': 0.1, 'high_percentile': 99.9}),
                name="Normalization"
            )
        )
    
    # Add position generation step
    position_steps.append(PositionGenerationStep())
    
    # Create position pipeline
    position_pipeline = Pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        steps=position_steps,
        name="Position Generation",
        well_filter=well_filter
    )
    
    pipelines = [position_pipeline]
    
    # Create stitching pipeline
    if stitch_original:
        z_pipeline = Pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            steps=[
                Step(
                    func=lambda x: x,
                    variable_components=['z_index'],
                    name="Z-Index Selector"
                ),
                ImageStitchingStep()
            ],
            name="Z-Stack Stitching",
            well_filter=well_filter
        )
        pipelines.append(z_pipeline)
    else:
        stitching_pipeline = Pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            steps=[ImageStitchingStep()],
            name="Flattened Z-Stack Stitching",
            well_filter=well_filter
        )
        pipelines.append(stitching_pipeline)
    
    return pipelines


def create_focus_pipeline(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    metric: str = "combined",
    normalize: bool = True,
    well_filter: Optional[List[str]] = None
) -> List[Pipeline]:
    """
    Create a Z-stack stitching pipeline with focus selection.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for stitched images
        metric: Focus metric ("combined", "laplacian", etc.)
        normalize: Whether to include normalization (default: True)
        well_filter: Wells to process
        
    Returns:
        List of pipelines
    """
    # Convert input_dir and output_dir to Path objects if they're strings
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if output_dir is not None and isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    # Create position generation pipeline
    position_steps = []
    
    # Add focus selection step
    position_steps.append(
        FocusStep(focus_options={'metric': metric})
    )
    
    # Add normalization if requested
    if normalize:
        position_steps.append(
            Step(
                func=(IP.stack_percentile_normalize, 
                      {'low_percentile': 0.1, 'high_percentile': 99.9}),
                name="Normalization"
            )
        )
    
    # Add position generation step
    position_steps.append(PositionGenerationStep())
    
    # Create pipelines
    position_pipeline = Pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        steps=position_steps,
        name="Position Generation",
        well_filter=well_filter
    )
    
    stitching_pipeline = Pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        steps=[ImageStitchingStep()],
        name="Focus-Selected Stitching",
        well_filter=well_filter
    )
    
    return [position_pipeline, stitching_pipeline]
