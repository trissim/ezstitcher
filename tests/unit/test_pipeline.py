import pytest
from ezstitcher.core.steps import ImageStitchingStep, PositionGenerationStep
from ezstitcher.core.pipeline import Pipeline, Step, ProcessingContext
from pathlib import Path

# Unit tests for the pipeline module

def test_step_path_resolution():
    """Test that paths are correctly resolved through the context."""

    # Create a pipeline with multiple steps
    pipeline = Pipeline(
        steps=[
            Step(name="Step 1", func=lambda x: x),
            PositionGenerationStep(),
            ImageStitchingStep()
        ]
    )

    # Create a mock context with path overrides
    step1_id = id(pipeline.steps[0])
    step2_id = id(pipeline.steps[1])
    step3_id = id(pipeline.steps[2])

    # Create a context with path overrides
    context = ProcessingContext()

    # Add step plans to the context
    from ezstitcher.core.pipeline import StepExecutionPlan

    # Step 1 plan
    context.add_step_plan(
        pipeline.steps[0],
        StepExecutionPlan(
            step_id=step1_id,
            step_name="Step 1",
            step_type="Step",
            input_dir=Path("input_dir"),
            output_dir=Path("step1_output")
        )
    )

    # Step 2 plan (PositionGenerationStep)
    context.add_step_plan(
        pipeline.steps[1],
        StepExecutionPlan(
            step_id=step2_id,
            step_name="Position Generation",
            step_type="PositionGenerationStep",
            input_dir=Path("step1_output"),
            output_dir=Path("positions_dir")
        )
    )

    # Step 3 plan (ImageStitchingStep)
    context.add_step_plan(
        pipeline.steps[2],
        StepExecutionPlan(
            step_id=step3_id,
            step_name="Image Stitching",
            step_type="ImageStitchingStep",
            input_dir=Path("positions_dir"),
            output_dir=Path("output_dir")
        )
    )

    # Verify that paths are correctly resolved through the context
    assert context.get_step_input_dir(pipeline.steps[0]) == Path("input_dir")
    assert context.get_step_output_dir(pipeline.steps[0]) == Path("step1_output")
    assert context.get_step_input_dir(pipeline.steps[1]) == Path("step1_output")
    assert context.get_step_output_dir(pipeline.steps[1]) == Path("positions_dir")
    assert context.get_step_input_dir(pipeline.steps[2]) == Path("positions_dir")
    assert context.get_step_output_dir(pipeline.steps[2]) == Path("output_dir")