"""
Unit tests for path resolution in the pipeline architecture.
"""

import pytest
from pathlib import Path
from ezstitcher.core.pipeline import Pipeline, ProcessingContext, StepExecutionPlan
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.config import PipelineConfig


class TestPathResolution:
    """Tests for path resolution in the pipeline architecture."""

    def test_default_path_resolution(self):
        """Test that paths are correctly resolved with default rules."""
        # Create a mock orchestrator
        orchestrator = PipelineOrchestrator(plate_path="test_plate")

        # Create a pipeline with multiple steps
        pipeline = Pipeline(
            steps=[
                Step(name="Step 1", func=lambda x: x),
                Step(name="Step 2", func=lambda x: x),
                PositionGenerationStep(),
                ImageStitchingStep()
            ]
        )

        # Create context with resolved paths
        context = orchestrator.create_context(pipeline)

        # Verify paths are correctly set
        assert context.get_step_input_dir(pipeline.steps[0]) == orchestrator.workspace_path
        assert context.get_step_output_dir(pipeline.steps[0]).name.endswith(orchestrator.config.out_dir_suffix)

        # Second step should use first step's output as input
        assert context.get_step_input_dir(pipeline.steps[1]) == context.get_step_output_dir(pipeline.steps[0])
        # Second step should use in-place processing by default
        assert context.get_step_input_dir(pipeline.steps[1]) == context.get_step_output_dir(pipeline.steps[1])

        # Position generation step
        assert context.get_step_input_dir(pipeline.steps[2]) == context.get_step_output_dir(pipeline.steps[1])
        assert context.get_step_output_dir(pipeline.steps[2]).name.endswith(orchestrator.config.positions_dir_suffix)

        # Image stitching step
        assert context.get_step_input_dir(pipeline.steps[3]) == context.get_step_output_dir(pipeline.steps[2])
        assert context.get_step_output_dir(pipeline.steps[3]).name.endswith(orchestrator.config.stitched_dir_suffix)

    def test_inline_path_overrides(self):
        """Test that inline path overrides in step constructors are respected."""
        # Create a mock orchestrator
        orchestrator = PipelineOrchestrator(plate_path="test_plate")

        # Create custom paths
        custom_input = Path("/custom/input")
        custom_output = Path("/custom/output")
        custom_positions = Path("/custom/positions")

        # Create a pipeline with steps that have inline path overrides
        pipeline = Pipeline(
            steps=[
                Step(name="Step 1", func=lambda x: x),
                Step(name="Step 2", func=lambda x: x, output_dir=custom_output),
                PositionGenerationStep(output_dir=custom_positions),
                ImageStitchingStep(input_dir=custom_input)
            ]
        )

        # Create context with resolved paths and path_overrides
        context = orchestrator.create_context(pipeline, path_overrides=pipeline.path_overrides)

        # Verify inline overrides are applied
        assert context.get_step_output_dir(pipeline.steps[1]) == custom_output
        assert context.get_step_output_dir(pipeline.steps[2]) == custom_positions
        assert context.get_step_input_dir(pipeline.steps[3]) == custom_input

        # Verify chaining is preserved where appropriate
        assert context.get_step_input_dir(pipeline.steps[1]) == context.get_step_output_dir(pipeline.steps[0])
        assert context.get_step_input_dir(pipeline.steps[2]) == custom_output

    def test_explicit_path_overrides(self):
        """Test that explicit path overrides via path_overrides dict are respected."""
        # Create a mock orchestrator
        orchestrator = PipelineOrchestrator(plate_path="test_plate")

        # Create a pipeline with multiple steps
        pipeline = Pipeline(
            steps=[
                Step(name="Step 1", func=lambda x: x),
                Step(name="Step 2", func=lambda x: x),
                PositionGenerationStep(),
                ImageStitchingStep()
            ]
        )

        # Create path overrides
        step1_id = id(pipeline.steps[0])
        step2_id = id(pipeline.steps[1])
        pos_step_id = id(pipeline.steps[2])

        path_overrides = {
            f"{step1_id}_output_dir": Path("/override/output1"),
            f"{step2_id}_input_dir": Path("/override/input2"),
            f"{step2_id}_output_dir": Path("/override/output2"),
            f"{pos_step_id}_output_dir": Path("/override/positions")
        }

        # Create context with resolved paths and overrides
        context = orchestrator.create_context(pipeline, path_overrides=path_overrides)

        # Verify overrides are applied
        assert context.get_step_output_dir(pipeline.steps[0]) == Path("/override/output1")
        assert context.get_step_input_dir(pipeline.steps[1]) == Path("/override/input2")
        assert context.get_step_output_dir(pipeline.steps[1]) == Path("/override/output2")
        assert context.get_step_output_dir(pipeline.steps[2]) == Path("/override/positions")

        # Verify that overrides affect downstream steps (chaining is preserved)
        # The ImageStitchingStep should use the custom positions directory as input
        assert context.get_step_input_dir(pipeline.steps[3]) == Path("/override/positions")

    def test_override_precedence(self):
        """Test that override precedence is correctly applied."""
        # Create a mock orchestrator
        orchestrator = PipelineOrchestrator(plate_path="test_plate")

        # Create a pipeline with steps that have inline path overrides
        pipeline = Pipeline(
            steps=[
                Step(name="Step 1", func=lambda x: x, output_dir=Path("/inline/output1")),
                Step(name="Step 2", func=lambda x: x, input_dir=Path("/inline/input2"), output_dir=Path("/inline/output2")),
            ]
        )

        # Create path overrides that should take precedence over inline overrides
        step1_id = id(pipeline.steps[0])
        step2_id = id(pipeline.steps[1])

        # Start with the pipeline's path_overrides
        path_overrides = pipeline.path_overrides.copy()

        # Add explicit overrides that should take precedence
        path_overrides.update({
            f"{step1_id}_output_dir": Path("/override/output1"),
            f"{step2_id}_input_dir": Path("/override/input2"),
        })

        # Create context with resolved paths and overrides
        context = orchestrator.create_context(pipeline, path_overrides=path_overrides)

        # Verify that explicit overrides take precedence over inline overrides
        assert context.get_step_output_dir(pipeline.steps[0]) == Path("/override/output1")
        assert context.get_step_input_dir(pipeline.steps[1]) == Path("/override/input2")

        # Verify that inline overrides that weren't explicitly overridden are still applied
        assert context.get_step_output_dir(pipeline.steps[1]) == Path("/inline/output2")
