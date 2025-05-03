"""
Unit tests for AutoPipelineFactory.

This module tests the AutoPipelineFactory class that creates pre-configured
pipelines for common workflows.
"""

import unittest
from pathlib import Path

from ezstitcher.core import AutoPipelineFactory
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep, ZFlatStep, CompositeStep
from ezstitcher.core.image_processor import ImageProcessor as IP


class TestAutoPipelineFactory(unittest.TestCase):
    """Test the AutoPipelineFactory class."""

    def test_basic_pipeline(self):
        """Test basic pipeline with minimal configuration."""
        input_dir = "path/to/input"
        factory = AutoPipelineFactory(
            input_dir=input_dir,
            normalize=True
        )
        pipelines = factory.create_pipelines()

        # Check that we have two pipelines
        self.assertEqual(len(pipelines), 2)

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        # Verify that path_overrides exists and is a dictionary
        self.assertIsInstance(position_pipeline.path_overrides, dict)
        # ZFlat + Normalization + Composite + PositionGeneration
        self.assertEqual(len(position_pipeline.steps), 4)
        self.assertIsInstance(position_pipeline.steps[0], ZFlatStep)
        self.assertIsInstance(position_pipeline.steps[1], Step)  # Normalization
        self.assertIsInstance(position_pipeline.steps[2], CompositeStep)
        self.assertIsInstance(position_pipeline.steps[3], PositionGenerationStep)

        # Check stitching pipeline
        stitching_pipeline = pipelines[1]
        # Verify that path_overrides exists and is a dictionary
        self.assertIsInstance(stitching_pipeline.path_overrides, dict)
        self.assertEqual(len(stitching_pipeline.steps), 2)  # Normalization + ImageStitching
        self.assertIsInstance(stitching_pipeline.steps[0], Step)  # Normalization
        self.assertIsInstance(stitching_pipeline.steps[1], ImageStitchingStep)

    def test_no_normalization(self):
        """Test without normalization."""
        input_dir = "path/to/input"
        factory = AutoPipelineFactory(
            input_dir=input_dir,
            normalize=False
        )
        pipelines = factory.create_pipelines()

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        self.assertEqual(len(position_pipeline.steps), 3)  # ZFlat + Composite + PositionGeneration
        self.assertIsInstance(position_pipeline.steps[0], ZFlatStep)
        self.assertIsInstance(position_pipeline.steps[1], CompositeStep)
        self.assertIsInstance(position_pipeline.steps[2], PositionGenerationStep)

        # Check stitching pipeline
        stitching_pipeline = pipelines[1]
        self.assertEqual(len(stitching_pipeline.steps), 1)  # Only ImageStitching
        self.assertIsInstance(stitching_pipeline.steps[0], ImageStitchingStep)

    def test_output_dir(self):
        """Test with custom output directory."""
        input_dir = "path/to/input"
        output_dir = "path/to/output"
        factory = AutoPipelineFactory(
            input_dir=input_dir,
            output_dir=output_dir
        )
        pipelines = factory.create_pipelines()

        # Check that pipelines are created correctly
        position_pipeline = pipelines[0]
        stitching_pipeline = pipelines[1]

        # Verify that path_overrides exists and is a dictionary
        self.assertIsInstance(position_pipeline.path_overrides, dict)
        self.assertIsInstance(stitching_pipeline.path_overrides, dict)

        # Verify the correct steps are in the pipelines
        self.assertIsInstance(position_pipeline.steps[-1], PositionGenerationStep)

        # Check that the output directory is correctly set in the path_overrides
        # for the last step of the stitching pipeline
        last_step_id = id(stitching_pipeline.steps[-1])
        output_key = f"{last_step_id}_output_dir"
        self.assertIn(output_key, stitching_pipeline.path_overrides)
        self.assertEqual(str(stitching_pipeline.path_overrides[output_key]), output_dir)

    def test_well_filter(self):
        """Test with well filter."""
        input_dir = "path/to/input"
        well_filter = ["A1", "A2", "A3"]
        factory = AutoPipelineFactory(
            input_dir=input_dir,
            well_filter=well_filter
        )
        pipelines = factory.create_pipelines()

        # Check that pipelines are created correctly
        position_pipeline = pipelines[0]
        stitching_pipeline = pipelines[1]

        # Verify that path_overrides exists and is a dictionary
        self.assertIsInstance(position_pipeline.path_overrides, dict)
        self.assertIsInstance(stitching_pipeline.path_overrides, dict)

        # Verify the correct steps are in the pipelines
        self.assertEqual(len(position_pipeline.steps), 4)  # ZFlat + Normalization + Composite + PositionGeneration
        self.assertIsInstance(position_pipeline.steps[0], ZFlatStep)
        self.assertIsInstance(position_pipeline.steps[1], Step)  # Normalization
        self.assertIsInstance(position_pipeline.steps[2], CompositeStep)
        self.assertIsInstance(position_pipeline.steps[3], PositionGenerationStep)

    def test_multichannel_pipeline(self):
        """Test multichannel pipeline with weights."""
        input_dir = "path/to/input"
        channel_weights = [0.7, 0.3]
        factory = AutoPipelineFactory(
            input_dir=input_dir,
            channel_weights=channel_weights
        )
        pipelines = factory.create_pipelines()

        # Check composite step
        position_pipeline = pipelines[0]
        composite_step = position_pipeline.steps[2]  # After normalization
        self.assertIsInstance(composite_step, CompositeStep)

    def test_zstack_pipeline(self):
        """Test z-stack pipeline with max projection."""
        input_dir = "path/to/input"
        factory = AutoPipelineFactory(
            input_dir=input_dir,
            flatten_z=True,
            z_method="max"
        )
        pipelines = factory.create_pipelines()

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        # ZFlat + Normalization + Composite + PositionGeneration
        self.assertEqual(len(position_pipeline.steps), 4)
        self.assertIsInstance(position_pipeline.steps[0], ZFlatStep)
        self.assertIsInstance(position_pipeline.steps[1], Step)  # Normalization
        self.assertIsInstance(position_pipeline.steps[2], CompositeStep)
        self.assertIsInstance(position_pipeline.steps[3], PositionGenerationStep)

        # Check stitching pipeline
        stitching_pipeline = pipelines[1]
        self.assertEqual(len(stitching_pipeline.steps), 3)  # Normalization + ZFlat + ImageStitching
        self.assertIsInstance(stitching_pipeline.steps[0], Step)  # Normalization
        self.assertIsInstance(stitching_pipeline.steps[1], ZFlatStep)
        self.assertIsInstance(stitching_pipeline.steps[2], ImageStitchingStep)

    def test_zstack_multichannel_pipeline(self):
        """Test z-stack pipeline with max projection and channel weights."""
        input_dir = "path/to/input"
        channel_weights = [0.7, 0.3]
        factory = AutoPipelineFactory(
            input_dir=input_dir,
            flatten_z=True,
            z_method="max",
            channel_weights=channel_weights
        )
        pipelines = factory.create_pipelines()

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        # ZFlat + Normalization + Composite + PositionGeneration
        self.assertEqual(len(position_pipeline.steps), 4)
        self.assertIsInstance(position_pipeline.steps[0], ZFlatStep)
        self.assertIsInstance(position_pipeline.steps[1], Step)  # Normalization
        self.assertIsInstance(position_pipeline.steps[2], CompositeStep)
        self.assertIsInstance(position_pipeline.steps[3], PositionGenerationStep)

        # Check composite step
        composite_step = position_pipeline.steps[2]
        self.assertIsInstance(composite_step, CompositeStep)


if __name__ == '__main__':
    unittest.main()
