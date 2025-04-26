"""
Unit tests for pipeline factories.

This module tests the pipeline factory functions that create pre-configured
pipelines for common workflows.
"""

import unittest
from pathlib import Path

# Import directly from the module
from ezstitcher.core.pipeline_factories import (
    create_basic_pipeline,
    create_multichannel_pipeline,
    create_zstack_pipeline,
    create_focus_pipeline
)
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.step_factories import ZFlatStep, FocusStep, CompositeStep
from ezstitcher.core.image_processor import ImageProcessor as IP


class TestBasicPipeline(unittest.TestCase):
    """Test the create_basic_pipeline function."""

    def test_default_parameters(self):
        """Test with default parameters."""
        input_dir = "path/to/input"
        pipelines = create_basic_pipeline(input_dir)

        # Check that we have two pipelines
        self.assertEqual(len(pipelines), 2)

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        self.assertEqual(str(position_pipeline.input_dir), input_dir)
        self.assertEqual(len(position_pipeline.steps), 2)  # Normalization + PositionGeneration
        self.assertIsInstance(position_pipeline.steps[0], Step)
        self.assertIsInstance(position_pipeline.steps[1], PositionGenerationStep)

        # Check stitching pipeline
        stitching_pipeline = pipelines[1]
        self.assertEqual(str(stitching_pipeline.input_dir), input_dir)
        self.assertEqual(len(stitching_pipeline.steps), 1)
        self.assertIsInstance(stitching_pipeline.steps[0], ImageStitchingStep)

    def test_no_normalization(self):
        """Test without normalization."""
        input_dir = "path/to/input"
        pipelines = create_basic_pipeline(
            input_dir,
            normalize=False
        )

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        self.assertEqual(len(position_pipeline.steps), 1)  # Only PositionGeneration
        self.assertIsInstance(position_pipeline.steps[0], PositionGenerationStep)

    def test_output_dir(self):
        """Test with custom output directory."""
        input_dir = "path/to/input"
        output_dir = "path/to/output"
        pipelines = create_basic_pipeline(
            input_dir,
            output_dir=output_dir
        )

        # Check output directories
        position_pipeline = pipelines[0]
        stitching_pipeline = pipelines[1]
        self.assertEqual(str(position_pipeline.output_dir), output_dir)
        self.assertEqual(str(stitching_pipeline.output_dir), output_dir)

    def test_well_filter(self):
        """Test with well filter."""
        input_dir = "path/to/input"
        well_filter = ["A1", "A2", "A3"]
        pipelines = create_basic_pipeline(
            input_dir,
            well_filter=well_filter
        )

        # Check well filter in pipelines
        position_pipeline = pipelines[0]
        stitching_pipeline = pipelines[1]
        self.assertEqual(position_pipeline.well_filter, well_filter)
        self.assertEqual(stitching_pipeline.well_filter, well_filter)


class TestMultiChannelPipeline(unittest.TestCase):
    """Test the create_multichannel_pipeline function."""

    def test_default_parameters(self):
        """Test with default parameters."""
        input_dir = "path/to/input"
        pipelines = create_multichannel_pipeline(input_dir)

        # Check that we have two pipelines
        self.assertEqual(len(pipelines), 2)

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        self.assertEqual(str(position_pipeline.input_dir), input_dir)
        self.assertEqual(len(position_pipeline.steps), 3)  # Normalization + Composite + PositionGeneration
        self.assertIsInstance(position_pipeline.steps[0], Step)  # Normalization
        self.assertIsInstance(position_pipeline.steps[1], CompositeStep)
        self.assertIsInstance(position_pipeline.steps[2], PositionGenerationStep)

        # Check stitching pipeline
        stitching_pipeline = pipelines[1]
        self.assertEqual(str(stitching_pipeline.input_dir), input_dir)
        self.assertEqual(len(stitching_pipeline.steps), 1)
        self.assertIsInstance(stitching_pipeline.steps[0], ImageStitchingStep)

    def test_custom_weights(self):
        """Test with custom composite weights."""
        input_dir = "path/to/input"
        weights = [0.7, 0.3]
        pipelines = create_multichannel_pipeline(
            input_dir,
            weights=weights
        )

        # Check composite step
        position_pipeline = pipelines[0]
        composite_step = position_pipeline.steps[1]
        self.assertIsInstance(composite_step, CompositeStep)

    def test_no_normalization(self):
        """Test without normalization."""
        input_dir = "path/to/input"
        pipelines = create_multichannel_pipeline(
            input_dir,
            normalize=False
        )

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        self.assertEqual(len(position_pipeline.steps), 2)  # Only Composite + PositionGeneration
        self.assertIsInstance(position_pipeline.steps[0], CompositeStep)
        self.assertIsInstance(position_pipeline.steps[1], PositionGenerationStep)

    def test_stitch_channels_separately(self):
        """Test with stitch_channels_separately=True."""
        input_dir = "path/to/input"
        pipelines = create_multichannel_pipeline(
            input_dir,
            stitch_channels_separately=True
        )

        # Check stitching pipeline
        stitching_pipeline = pipelines[1]
        self.assertEqual(len(stitching_pipeline.steps), 2)  # Channel selector + ImageStitching
        self.assertEqual(stitching_pipeline.steps[0].variable_components, ['channel'])

    def test_well_filter(self):
        """Test with well filter."""
        input_dir = "path/to/input"
        well_filter = ["A1", "A2", "A3"]
        pipelines = create_multichannel_pipeline(
            input_dir,
            well_filter=well_filter
        )

        # Check well filter in pipelines
        position_pipeline = pipelines[0]
        stitching_pipeline = pipelines[1]
        self.assertEqual(position_pipeline.well_filter, well_filter)
        self.assertEqual(stitching_pipeline.well_filter, well_filter)


class TestZStackPipeline(unittest.TestCase):
    """Test the create_zstack_pipeline function."""

    def test_default_parameters(self):
        """Test with default parameters."""
        input_dir = "path/to/input"
        pipelines = create_zstack_pipeline(input_dir)

        # Check that we have two pipelines
        self.assertEqual(len(pipelines), 2)

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        self.assertEqual(str(position_pipeline.input_dir), input_dir)
        self.assertEqual(len(position_pipeline.steps), 3)  # ZFlat + Normalization + PositionGeneration
        self.assertIsInstance(position_pipeline.steps[0], ZFlatStep)
        self.assertIsInstance(position_pipeline.steps[1], Step)  # Normalization
        self.assertIsInstance(position_pipeline.steps[2], PositionGenerationStep)

        # Check stitching pipeline
        stitching_pipeline = pipelines[1]
        self.assertEqual(str(stitching_pipeline.input_dir), input_dir)
        self.assertEqual(len(stitching_pipeline.steps), 1)
        self.assertIsInstance(stitching_pipeline.steps[0], ImageStitchingStep)

    def test_custom_method(self):
        """Test with custom projection method."""
        input_dir = "path/to/input"
        pipelines = create_zstack_pipeline(
            input_dir,
            method="mean"
        )

        # Check ZFlat step
        position_pipeline = pipelines[0]
        zflat_step = position_pipeline.steps[0]
        self.assertIsInstance(zflat_step, ZFlatStep)

    def test_no_normalization(self):
        """Test without normalization."""
        input_dir = "path/to/input"
        pipelines = create_zstack_pipeline(
            input_dir,
            normalize=False
        )

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        self.assertEqual(len(position_pipeline.steps), 2)  # Only ZFlat + PositionGeneration
        self.assertIsInstance(position_pipeline.steps[0], ZFlatStep)
        self.assertIsInstance(position_pipeline.steps[1], PositionGenerationStep)

    def test_stitch_original(self):
        """Test with stitch_original=True."""
        input_dir = "path/to/input"
        pipelines = create_zstack_pipeline(
            input_dir,
            stitch_original=True
        )

        # Check stitching pipeline
        stitching_pipeline = pipelines[1]
        self.assertEqual(len(stitching_pipeline.steps), 2)  # Z-index selector + ImageStitching
        self.assertEqual(stitching_pipeline.steps[0].variable_components, ['z_index'])

    def test_well_filter(self):
        """Test with well filter."""
        input_dir = "path/to/input"
        well_filter = ["A1", "A2", "A3"]
        pipelines = create_zstack_pipeline(
            input_dir,
            well_filter=well_filter
        )

        # Check well filter in pipelines
        position_pipeline = pipelines[0]
        stitching_pipeline = pipelines[1]
        self.assertEqual(position_pipeline.well_filter, well_filter)
        self.assertEqual(stitching_pipeline.well_filter, well_filter)


class TestFocusPipeline(unittest.TestCase):
    """Test the create_focus_pipeline function."""

    def test_default_parameters(self):
        """Test with default parameters."""
        input_dir = "path/to/input"
        pipelines = create_focus_pipeline(input_dir)

        # Check that we have two pipelines
        self.assertEqual(len(pipelines), 2)

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        self.assertEqual(str(position_pipeline.input_dir), input_dir)
        self.assertEqual(len(position_pipeline.steps), 3)  # Focus + Normalization + PositionGeneration
        self.assertIsInstance(position_pipeline.steps[0], FocusStep)
        self.assertIsInstance(position_pipeline.steps[1], Step)  # Normalization
        self.assertIsInstance(position_pipeline.steps[2], PositionGenerationStep)

        # Check stitching pipeline
        stitching_pipeline = pipelines[1]
        self.assertEqual(str(stitching_pipeline.input_dir), input_dir)
        self.assertEqual(len(stitching_pipeline.steps), 1)
        self.assertIsInstance(stitching_pipeline.steps[0], ImageStitchingStep)

    def test_custom_metric(self):
        """Test with custom focus metric."""
        input_dir = "path/to/input"
        pipelines = create_focus_pipeline(
            input_dir,
            metric="laplacian"
        )

        # Check Focus step
        position_pipeline = pipelines[0]
        focus_step = position_pipeline.steps[0]
        self.assertIsInstance(focus_step, FocusStep)

    def test_no_normalization(self):
        """Test without normalization."""
        input_dir = "path/to/input"
        pipelines = create_focus_pipeline(
            input_dir,
            normalize=False
        )

        # Check position generation pipeline
        position_pipeline = pipelines[0]
        self.assertEqual(len(position_pipeline.steps), 2)  # Only Focus + PositionGeneration
        self.assertIsInstance(position_pipeline.steps[0], FocusStep)
        self.assertIsInstance(position_pipeline.steps[1], PositionGenerationStep)

    def test_well_filter(self):
        """Test with well filter."""
        input_dir = "path/to/input"
        well_filter = ["A1", "A2", "A3"]
        pipelines = create_focus_pipeline(
            input_dir,
            well_filter=well_filter
        )

        # Check well filter in pipelines
        position_pipeline = pipelines[0]
        stitching_pipeline = pipelines[1]
        self.assertEqual(position_pipeline.well_filter, well_filter)
        self.assertEqual(stitching_pipeline.well_filter, well_filter)


if __name__ == '__main__':
    unittest.main()
