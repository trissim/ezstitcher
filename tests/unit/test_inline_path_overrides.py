"""
Unit tests for inline path overrides in step constructors.
"""

import pytest
from pathlib import Path
from ezstitcher.core.pipeline import Pipeline, ProcessingContext, StepExecutionPlan
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep, NormStep, ZFlatStep
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.config import PipelineConfig


class TestInlinePathOverrides:
    """Tests for inline path overrides in step constructors."""

    def test_inline_path_extraction(self):
        """Test that inline path overrides are extracted from step constructors."""
        # Create custom paths
        custom_input = Path("/custom/input")
        custom_output = Path("/custom/output")

        # Create steps with inline path overrides
        step1 = Step(name="Step 1", func=lambda x: x)
        step2 = Step(name="Step 2", func=lambda x: x)

        # Set attributes dynamically
        step1.input_dir = custom_input
        step2.output_dir = custom_output

        # Create a pipeline with these steps
        pipeline = Pipeline(steps=[step1, step2])

        # Verify that path_overrides contains the extracted paths
        assert len(pipeline.path_overrides) == 2

        # Get step IDs
        step1_id = id(pipeline.steps[0])
        step2_id = id(pipeline.steps[1])

        # Verify that the path_overrides dictionary contains the correct keys and values
        assert f"{step1_id}_input_dir" in pipeline.path_overrides
        assert f"{step2_id}_output_dir" in pipeline.path_overrides
        assert pipeline.path_overrides[f"{step1_id}_input_dir"] == custom_input
        assert pipeline.path_overrides[f"{step2_id}_output_dir"] == custom_output

        # Verify that the attributes were removed from the step instances
        assert not hasattr(pipeline.steps[0], "input_dir")
        assert not hasattr(pipeline.steps[1], "output_dir")

    def test_string_path_conversion(self):
        """Test that string paths are correctly converted to Path objects."""
        # Create steps with string path overrides
        step1 = Step(name="Step 1", func=lambda x: x)
        step2 = Step(name="Step 2", func=lambda x: x)

        # Set attributes dynamically
        step1.input_dir = "input/dir"
        step2.output_dir = "output/dir"

        # Create a pipeline with these steps
        pipeline = Pipeline(steps=[step1, step2])

        # Get step IDs
        step1_id = id(pipeline.steps[0])
        step2_id = id(pipeline.steps[1])

        # Verify that the string paths were converted to Path objects
        assert isinstance(pipeline.path_overrides[f"{step1_id}_input_dir"], Path)
        assert isinstance(pipeline.path_overrides[f"{step2_id}_output_dir"], Path)
        assert pipeline.path_overrides[f"{step1_id}_input_dir"] == Path("input/dir")
        assert pipeline.path_overrides[f"{step2_id}_output_dir"] == Path("output/dir")

    def test_specialized_steps(self):
        """Test that inline path overrides work with specialized step classes."""
        # Create specialized steps
        norm_step = NormStep()
        zflat_step = ZFlatStep()
        pos_step = PositionGenerationStep()
        stitch_step = ImageStitchingStep()

        # Set attributes dynamically
        norm_step.output_dir = "intermediate/norm"
        zflat_step.output_dir = "intermediate/zflat"
        pos_step.output_dir = "positions"
        stitch_step.input_dir = "positions"
        stitch_step.output_dir = "stitched"

        # Create a pipeline with these steps
        pipeline = Pipeline(steps=[norm_step, zflat_step, pos_step, stitch_step])

        # Verify that path_overrides contains the extracted paths
        assert len(pipeline.path_overrides) == 5  # 5 overrides for 4 steps

        # Get step IDs
        norm_step_id = id(pipeline.steps[0])
        zflat_step_id = id(pipeline.steps[1])
        pos_step_id = id(pipeline.steps[2])
        stitch_step_id = id(pipeline.steps[3])

        # Verify that the path_overrides dictionary contains the correct keys and values
        assert f"{norm_step_id}_output_dir" in pipeline.path_overrides
        assert f"{zflat_step_id}_output_dir" in pipeline.path_overrides
        assert f"{pos_step_id}_output_dir" in pipeline.path_overrides
        assert f"{stitch_step_id}_input_dir" in pipeline.path_overrides
        assert f"{stitch_step_id}_output_dir" in pipeline.path_overrides

        # Verify that the attributes were removed from the step instances
        assert not hasattr(pipeline.steps[0], "output_dir")
        assert not hasattr(pipeline.steps[1], "output_dir")
        assert not hasattr(pipeline.steps[2], "output_dir")
        assert not hasattr(pipeline.steps[3], "input_dir")
        assert not hasattr(pipeline.steps[3], "output_dir")

    def test_orchestrator_integration(self):
        """Test that inline path overrides are correctly passed to the orchestrator."""
        # Create a mock orchestrator
        orchestrator = PipelineOrchestrator(plate_path="test_plate")

        # Create specialized steps
        norm_step = NormStep()
        zflat_step = ZFlatStep()
        pos_step = PositionGenerationStep()
        stitch_step = ImageStitchingStep()

        # Set attributes dynamically
        norm_step.output_dir = Path("intermediate/norm")
        zflat_step.output_dir = Path("intermediate/zflat")
        pos_step.output_dir = Path("positions")
        stitch_step.output_dir = Path("stitched")

        # Create a pipeline with these steps
        pipeline = Pipeline(steps=[norm_step, zflat_step, pos_step, stitch_step])

        # Create context with resolved paths
        context = orchestrator.create_context(pipeline, path_overrides=pipeline.path_overrides)

        # Verify that the inline overrides are correctly applied
        assert context.get_step_output_dir(pipeline.steps[0]) == Path("intermediate/norm")
        assert context.get_step_output_dir(pipeline.steps[1]) == Path("intermediate/zflat")
        assert context.get_step_output_dir(pipeline.steps[2]) == Path("positions")
        assert context.get_step_output_dir(pipeline.steps[3]) == Path("stitched")

        # Verify that chaining is preserved where appropriate
        assert context.get_step_input_dir(pipeline.steps[1]) == context.get_step_output_dir(pipeline.steps[0])
        assert context.get_step_input_dir(pipeline.steps[2]) == context.get_step_output_dir(pipeline.steps[1])
        assert context.get_step_input_dir(pipeline.steps[3]) == context.get_step_output_dir(pipeline.steps[2])

    def test_direct_inline_declaration(self):
        """Test the direct inline declaration syntax that the user wants to support."""
        # Create a mock orchestrator
        orchestrator = PipelineOrchestrator(plate_path="test_plate")

        # Create steps with inline path declarations
        norm_step = NormStep()
        norm_step.output_dir = "intermediate/norm"

        zflat_step = ZFlatStep()
        zflat_step.output_dir = "intermediate/zflat"

        pos_step = PositionGenerationStep()
        pos_step.output_dir = "positions"

        stitch_step = ImageStitchingStep()
        stitch_step.output_dir = "stitched"

        # Create a pipeline with these steps
        pipeline = Pipeline([
            norm_step,
            zflat_step,
            pos_step,
            stitch_step
        ])

        # Create context with resolved paths
        context = orchestrator.create_context(pipeline, path_overrides=pipeline.path_overrides)

        # Verify that the inline overrides are correctly applied
        assert context.get_step_output_dir(pipeline.steps[0]) == Path("intermediate/norm")
        assert context.get_step_output_dir(pipeline.steps[1]) == Path("intermediate/zflat")
        assert context.get_step_output_dir(pipeline.steps[2]) == Path("positions")
        assert context.get_step_output_dir(pipeline.steps[3]) == Path("stitched")

        # Verify that chaining is preserved where appropriate
        assert context.get_step_input_dir(pipeline.steps[1]) == context.get_step_output_dir(pipeline.steps[0])
        assert context.get_step_input_dir(pipeline.steps[2]) == context.get_step_output_dir(pipeline.steps[1])
        assert context.get_step_input_dir(pipeline.steps[3]) == context.get_step_output_dir(pipeline.steps[2])

    def test_inline_path_before_pipeline_construction(self):
        """Test that inline paths must be set before pipeline construction."""
        # Create a mock orchestrator
        orchestrator = PipelineOrchestrator(plate_path="test_plate")

        # Create steps with inline path declarations BEFORE pipeline construction
        norm_step = NormStep()
        norm_step.output_dir = "intermediate/norm"

        zflat_step = ZFlatStep()
        zflat_step.output_dir = "intermediate/zflat"

        pos_step = PositionGenerationStep()
        pos_step.output_dir = "positions"

        stitch_step = ImageStitchingStep()
        stitch_step.output_dir = "stitched"

        # Create a pipeline with these steps
        pipeline = Pipeline([
            norm_step,
            zflat_step,
            pos_step,
            stitch_step
        ])

        # Create context with resolved paths
        context = orchestrator.create_context(pipeline, path_overrides=pipeline.path_overrides)

        # Verify that the inline overrides are correctly applied
        assert context.get_step_output_dir(pipeline.steps[0]) == Path("intermediate/norm")
        assert context.get_step_output_dir(pipeline.steps[1]) == Path("intermediate/zflat")
        assert context.get_step_output_dir(pipeline.steps[2]) == Path("positions")
        assert context.get_step_output_dir(pipeline.steps[3]) == Path("stitched")

        # Verify that chaining is preserved where appropriate
        assert context.get_step_input_dir(pipeline.steps[1]) == context.get_step_output_dir(pipeline.steps[0])
        assert context.get_step_input_dir(pipeline.steps[2]) == context.get_step_output_dir(pipeline.steps[1])
        assert context.get_step_input_dir(pipeline.steps[3]) == context.get_step_output_dir(pipeline.steps[2])

    def test_post_hoc_assignment_has_no_effect(self):
        """Test that post-hoc assignment of input_dir/output_dir has no effect."""
        # Create a mock orchestrator
        orchestrator = PipelineOrchestrator(plate_path="test_plate")

        # Create a pipeline with steps
        pipeline = Pipeline([
            NormStep(),
            ZFlatStep(),
            PositionGenerationStep(),
            ImageStitchingStep()
        ])

        # Try to set attributes after pipeline creation - this should have no effect
        # because the steps are already added to the pipeline
        try:
            pipeline.steps[0].output_dir = "intermediate/norm"
            pipeline.steps[1].output_dir = "intermediate/zflat"
            pipeline.steps[2].output_dir = "positions"
            pipeline.steps[3].output_dir = "stitched"
        except AttributeError:
            # This is expected - setting attributes after pipeline creation is not supported
            pass

        # Verify that the path_overrides dictionary is empty
        # because post-hoc assignment should have no effect
        assert len(pipeline.path_overrides) == 0

        # Create context with resolved paths
        context = orchestrator.create_context(pipeline, path_overrides=pipeline.path_overrides)

        # Verify that the default paths are used, not the post-hoc assignments
        # The actual values will depend on the default path resolution logic
        # We just verify that they're not the values we tried to set post-hoc
        assert context.get_step_output_dir(pipeline.steps[0]) != Path("intermediate/norm")
        assert context.get_step_output_dir(pipeline.steps[1]) != Path("intermediate/zflat")
        assert context.get_step_output_dir(pipeline.steps[2]) != Path("positions")
        assert context.get_step_output_dir(pipeline.steps[3]) != Path("stitched")
