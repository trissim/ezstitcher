"""
Unit tests for Pipeline input_dir and output_dir arguments.
"""

import pytest
from pathlib import Path
from ezstitcher.core.pipeline import Pipeline, ProcessingContext, StepExecutionPlan
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep, NormStep, ZFlatStep
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.config import PipelineConfig


class TestPipelineInputOutputDirs:
    """Tests for Pipeline input_dir and output_dir arguments."""

    def test_input_output_dir_first_last_step(self):
        """Test that input_dir applies to first step and output_dir applies to last step."""
        # Create a mock orchestrator
        orchestrator = PipelineOrchestrator(plate_path="test_plate")
        
        # Create custom paths
        input_dir = Path("/custom/input")
        output_dir = Path("/custom/output")
        
        # Create a pipeline with input_dir and output_dir
        pipeline = Pipeline(
            steps=[
                Step(name="Step 1", func=lambda x: x),
                Step(name="Step 2", func=lambda x: x),
                Step(name="Step 3", func=lambda x: x)
            ],
            input_dir=input_dir,
            output_dir=output_dir
        )
        
        # Create context with resolved paths
        context = orchestrator.create_context(pipeline, path_overrides=pipeline.path_overrides)
        
        # Verify that input_dir applies to first step
        assert context.get_step_input_dir(pipeline.steps[0]) == input_dir
        
        # Verify that output_dir applies to last step
        assert context.get_step_output_dir(pipeline.steps[2]) == output_dir
        
        # Verify that middle steps use default chaining
        assert context.get_step_input_dir(pipeline.steps[1]) == context.get_step_output_dir(pipeline.steps[0])
        assert context.get_step_input_dir(pipeline.steps[2]) == context.get_step_output_dir(pipeline.steps[1])
    
    def test_step_override_precedence(self):
        """Test that step-level overrides take precedence over pipeline-level input_dir and output_dir."""
        # Create a mock orchestrator
        orchestrator = PipelineOrchestrator(plate_path="test_plate")
        
        # Create custom paths
        pipeline_input_dir = Path("/pipeline/input")
        pipeline_output_dir = Path("/pipeline/output")
        step_input_dir = Path("/step/input")
        step_output_dir = Path("/step/output")
        
        # Create steps with inline path overrides
        step1 = Step(name="Step 1", func=lambda x: x)
        step1.input_dir = step_input_dir
        
        step3 = Step(name="Step 3", func=lambda x: x)
        step3.output_dir = step_output_dir
        
        # Create a pipeline with input_dir and output_dir
        pipeline = Pipeline(
            steps=[
                step1,
                Step(name="Step 2", func=lambda x: x),
                step3
            ],
            input_dir=pipeline_input_dir,
            output_dir=pipeline_output_dir
        )
        
        # Create context with resolved paths
        context = orchestrator.create_context(pipeline, path_overrides=pipeline.path_overrides)
        
        # Verify that step-level overrides take precedence
        assert context.get_step_input_dir(pipeline.steps[0]) == step_input_dir
        assert context.get_step_output_dir(pipeline.steps[2]) == step_output_dir
        
        # Verify that middle step uses default chaining
        assert context.get_step_input_dir(pipeline.steps[1]) == context.get_step_output_dir(pipeline.steps[0])
        assert context.get_step_input_dir(pipeline.steps[2]) == context.get_step_output_dir(pipeline.steps[1])
    
    def test_example_usage(self):
        """Test the example usage from the requirements."""
        # Create a mock orchestrator
        orchestrator = PipelineOrchestrator(plate_path="test_plate")
        
        # Create a pipeline with the example usage
        pipeline = Pipeline(
            steps=[
                NormStep(),
                ZFlatStep(output_dir="tmp/zflat"),
                Step(name="SaveStep", func=lambda x: x)
            ],
            input_dir="raw",
            output_dir="final"
        )
        
        # Create context with resolved paths
        context = orchestrator.create_context(pipeline, path_overrides=pipeline.path_overrides)
        
        # Verify that NormStep gets input=raw, output=final
        assert context.get_step_input_dir(pipeline.steps[0]) == Path("raw")
        
        # Verify that ZFlatStep keeps its explicit output of tmp/zflat, but input should be raw
        assert context.get_step_output_dir(pipeline.steps[1]) == Path("tmp/zflat")
        assert context.get_step_input_dir(pipeline.steps[1]) == context.get_step_output_dir(pipeline.steps[0])
        
        # Verify that SaveStep inherits both input and output from the pipeline
        assert context.get_step_input_dir(pipeline.steps[2]) == context.get_step_output_dir(pipeline.steps[1])
        assert context.get_step_output_dir(pipeline.steps[2]) == Path("final")
