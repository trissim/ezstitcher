import pytest
from ezstitcher.core.steps import ImageStitchingStep, PositionGenerationStep
from ezstitcher.core.pipeline import Pipeline, Step

# Unit tests for the pipeline module

def test_image_stitching_default_input_dir():
    """Test that ImageStitchingStep uses previous step's output_dir by default."""

    # Create a pipeline with multiple steps
    pipeline = Pipeline(
        input_dir='input_dir',
        output_dir='output_dir',
        steps=[
            Step(name="Step 1", func=lambda x: x, output_dir='step1_output'),
            PositionGenerationStep(output_dir='positions_dir'),
            ImageStitchingStep()
        ]
    )

    # Verify that ImageStitchingStep uses the previous step's output_dir
    stitching_step = pipeline.steps[2]
    assert stitching_step.input_dir == 'positions_dir'

    # Verify that explicitly specifying input_dir overrides the default
    pipeline = Pipeline(
        input_dir='input_dir',
        output_dir='output_dir',
        steps=[
            Step(name="Step 1", func=lambda x: x, output_dir='step1_output'),
            PositionGenerationStep(output_dir='positions_dir'),
            ImageStitchingStep(input_dir='custom_input')
        ]
    )

    stitching_step = pipeline.steps[2]
    assert stitching_step.input_dir == 'custom_input'