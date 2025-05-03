"""Tests for directory resolution logic."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from ezstitcher.core.pipeline import Pipeline, ProcessingContext, StepExecutionPlan
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.config import PipelineConfig
from ezstitcher.io.filemanager import FileManager


class TestDirectoryResolution:
    """Test the directory resolution logic."""

    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.plate_path = self.temp_dir / "plate"
        self.plate_path.mkdir(exist_ok=True)

        # Create a file manager for testing
        self.file_manager = FileManager(backend="disk")

        # Create a mock orchestrator
        self.orchestrator = MagicMock()
        self.orchestrator.file_manager = self.file_manager
        self.orchestrator.plate_path = self.plate_path

        # Create a config with custom suffixes
        self.config = PipelineConfig(
            out_dir_suffix="_out",
            positions_dir_suffix="_positions",
            stitched_dir_suffix="_stitched"
        )
        self.orchestrator.config = self.config

    def teardown_method(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_get_step_output_dir_creates_directory(self):
        """Test that get_step_output_dir creates the directory if it doesn't exist."""
        # Create a context
        context = ProcessingContext(orchestrator=self.orchestrator)

        # Create a step
        step = Step(func=lambda x: x, name="Test Step")

        # Create a plan with a non-existent output directory
        output_dir = self.temp_dir / "nonexistent_dir"
        plan = StepExecutionPlan(
            step_id=id(step),
            step_name=step.name,
            step_type=type(step).__name__,
            input_dir=self.plate_path,
            output_dir=output_dir
        )

        # Add the plan to the context
        context.add_step_plan(step, plan)

        # Get the output directory
        result = context.get_step_output_dir(step)

        # Check that the directory was created
        assert result == output_dir
        assert output_dir.exists()

    def test_position_generation_step_uses_positions_suffix(self):
        """Test that PositionGenerationStep uses the positions_dir_suffix."""
        # Create a pipeline with a PositionGenerationStep
        step = PositionGenerationStep()
        pipeline = Pipeline(steps=[step])

        # Create a plan with the correct suffix
        expected_output_dir = self.plate_path.parent / f"{self.plate_path.name}{self.config.positions_dir_suffix}"
        plan = StepExecutionPlan(
            step_id=id(step),
            step_name=step.name,
            step_type=type(step).__name__,
            input_dir=self.plate_path,
            output_dir=expected_output_dir
        )

        # Create a context and add the plan
        context = ProcessingContext(orchestrator=self.orchestrator)
        context.add_step_plan(step, plan)

        # Get the output directory
        result = context.get_step_output_dir(step)

        # Check that the directory was created with the correct suffix
        assert result == expected_output_dir
        assert expected_output_dir.exists()
        assert expected_output_dir.name == f"{self.plate_path.name}{self.config.positions_dir_suffix}"

    def test_image_stitching_step_uses_stitched_suffix(self):
        """Test that ImageStitchingStep uses the stitched_dir_suffix."""
        # Create a pipeline with an ImageStitchingStep
        step = ImageStitchingStep()
        pipeline = Pipeline(steps=[step])

        # Create a plan with the correct suffix
        expected_output_dir = self.plate_path.parent / f"{self.plate_path.name}{self.config.stitched_dir_suffix}"
        plan = StepExecutionPlan(
            step_id=id(step),
            step_name=step.name,
            step_type=type(step).__name__,
            input_dir=self.plate_path,
            output_dir=expected_output_dir
        )

        # Create a context and add the plan
        context = ProcessingContext(orchestrator=self.orchestrator)
        context.add_step_plan(step, plan)

        # Get the output directory
        result = context.get_step_output_dir(step)

        # Check that the directory was created with the correct suffix
        assert result == expected_output_dir
        assert expected_output_dir.exists()
        assert expected_output_dir.name == f"{self.plate_path.name}{self.config.stitched_dir_suffix}"

    def test_normal_step_uses_out_suffix(self):
        """Test that a normal step uses the out_dir_suffix."""
        # Create a pipeline with a normal step
        step = Step(func=lambda x: x, name="Normal Step")
        pipeline = Pipeline(steps=[step])

        # Create a plan with the correct suffix
        expected_output_dir = self.plate_path.parent / f"{self.plate_path.name}{self.config.out_dir_suffix}"
        plan = StepExecutionPlan(
            step_id=id(step),
            step_name=step.name,
            step_type=type(step).__name__,
            input_dir=self.plate_path,
            output_dir=expected_output_dir
        )

        # Create a context and add the plan
        context = ProcessingContext(orchestrator=self.orchestrator)
        context.add_step_plan(step, plan)

        # Get the output directory
        result = context.get_step_output_dir(step)

        # Check that the directory was created with the correct suffix
        assert result == expected_output_dir
        assert expected_output_dir.exists()
        assert expected_output_dir.name == f"{self.plate_path.name}{self.config.out_dir_suffix}"
