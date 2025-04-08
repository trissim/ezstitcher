"""
Unit tests for the legacy dataclass-based configuration classes.

Note: This file tests the original dataclass-based configuration system.
For tests of the newer Pydantic-based configuration system, see test_pydantic_config.py.
This file may be deprecated in the future as the codebase transitions to Pydantic models.
"""

import unittest
from pathlib import Path

from ezstitcher.core.config import (
    StitcherConfig,
    FocusAnalyzerConfig,
    ImagePreprocessorConfig,
    ZStackProcessorConfig,
    PlateProcessorConfig
)


class TestConfig(unittest.TestCase):
    """Test the configuration classes."""

    def test_stitcher_config(self):
        """Test the StitcherConfig class."""
        # Test default values
        config = StitcherConfig()
        self.assertEqual(config.tile_overlap, 6.5)
        self.assertIsNone(config.tile_overlap_x)
        self.assertIsNone(config.tile_overlap_y)
        self.assertEqual(config.max_shift, 50)
        self.assertEqual(config.margin_ratio, 0.1)
        self.assertEqual(config.pixel_size, 1.0)

        # Test custom values
        config = StitcherConfig(
            tile_overlap=10.0,
            tile_overlap_x=12.0,
            tile_overlap_y=8.0,
            max_shift=100,
            margin_ratio=0.2,
            pixel_size=0.5
        )
        self.assertEqual(config.tile_overlap, 10.0)
        self.assertEqual(config.tile_overlap_x, 12.0)
        self.assertEqual(config.tile_overlap_y, 8.0)
        self.assertEqual(config.max_shift, 100)
        self.assertEqual(config.margin_ratio, 0.2)
        self.assertEqual(config.pixel_size, 0.5)

    def test_focus_analyzer_config(self):
        """Test the FocusAnalyzerConfig class."""
        # Test default values
        config = FocusAnalyzerConfig()
        self.assertEqual(config.method, "combined")
        self.assertIsNone(config.roi)
        self.assertIsNone(config.weights)

        # Test custom values
        config = FocusAnalyzerConfig(
            method="laplacian",
            roi=(10, 10, 100, 100),
            weights={"nvar": 0.5, "lap": 0.5}
        )
        self.assertEqual(config.method, "laplacian")
        self.assertEqual(config.roi, (10, 10, 100, 100))
        self.assertEqual(config.weights, {"nvar": 0.5, "lap": 0.5})

    def test_image_preprocessor_config(self):
        """Test the ImagePreprocessorConfig class."""
        # Test default values
        config = ImagePreprocessorConfig()
        self.assertEqual(config.preprocessing_funcs, {})
        self.assertIsNone(config.composite_weights)

        # Test custom values
        def dummy_func(img):
            return img

        config = ImagePreprocessorConfig(
            preprocessing_funcs={"1": dummy_func},
            composite_weights={"1": 0.7, "2": 0.3}
        )
        self.assertEqual(len(config.preprocessing_funcs), 1)
        self.assertIn("1", config.preprocessing_funcs)
        self.assertEqual(config.composite_weights, {"1": 0.7, "2": 0.3})

    def test_zstack_processor_config(self):
        """Test the ZStackProcessorConfig class."""
        # Test default values
        config = ZStackProcessorConfig()
        self.assertFalse(config.focus_detect)
        self.assertEqual(config.focus_method, "combined")
        self.assertFalse(config.create_projections)
        self.assertEqual(config.stitch_z_reference, "max")
        self.assertTrue(config.save_projections)
        self.assertFalse(config.stitch_all_z_planes)
        self.assertEqual(config.projection_types, ["max"])

        # Test custom values
        config = ZStackProcessorConfig(
            focus_detect=True,
            focus_method="laplacian",
            create_projections=True,
            stitch_z_reference="max",
            save_projections=False,
            stitch_all_z_planes=True,
            projection_types=["max", "mean"]
        )
        self.assertTrue(config.focus_detect)
        self.assertEqual(config.focus_method, "laplacian")
        self.assertTrue(config.create_projections)
        self.assertEqual(config.stitch_z_reference, "max")
        self.assertFalse(config.save_projections)
        self.assertTrue(config.stitch_all_z_planes)
        self.assertEqual(config.projection_types, ["max", "mean"])

    def test_plate_processor_config(self):
        """Test the PlateProcessorConfig class."""
        # Test default values
        config = PlateProcessorConfig()
        self.assertEqual(config.reference_channels, ["1"])
        self.assertIsNone(config.well_filter)
        self.assertFalse(config.use_reference_positions)
        self.assertEqual(config.output_dir_suffix, "_processed")
        self.assertEqual(config.positions_dir_suffix, "_positions")
        self.assertEqual(config.stitched_dir_suffix, "_stitched")
        self.assertEqual(config.best_focus_dir_suffix, "_best_focus")
        self.assertEqual(config.projections_dir_suffix, "_Projections")
        self.assertEqual(config.timepoint_dir_name, "TimePoint_1")
        self.assertIsNone(config.preprocessing_funcs)
        self.assertIsNone(config.composite_weights)
        self.assertIsInstance(config.stitcher, StitcherConfig)
        self.assertIsInstance(config.focus_analyzer, FocusAnalyzerConfig)
        self.assertIsInstance(config.image_preprocessor, ImagePreprocessorConfig)
        self.assertIsInstance(config.z_stack_processor, ZStackProcessorConfig)

        # Test custom values
        stitcher_config = StitcherConfig(tile_overlap=10.0)
        focus_config = FocusAnalyzerConfig(method="laplacian")
        image_config = ImagePreprocessorConfig(composite_weights={"1": 0.7, "2": 0.3})
        zstack_config = ZStackProcessorConfig(focus_detect=True)

        config = PlateProcessorConfig(
            reference_channels=["1", "2"],
            well_filter=["A01", "A02"],
            use_reference_positions=True,
            output_dir_suffix="_custom_processed",
            positions_dir_suffix="_custom_positions",
            stitched_dir_suffix="_custom_stitched",
            best_focus_dir_suffix="_custom_best_focus",
            projections_dir_suffix="_custom_projections",
            timepoint_dir_name="CustomTimePoint",
            preprocessing_funcs={"1": lambda x: x},
            composite_weights={"1": 0.7, "2": 0.3},
            stitcher=stitcher_config,
            focus_analyzer=focus_config,
            image_preprocessor=image_config,
            z_stack_processor=zstack_config
        )

        self.assertEqual(config.reference_channels, ["1", "2"])
        self.assertEqual(config.well_filter, ["A01", "A02"])
        self.assertTrue(config.use_reference_positions)
        self.assertEqual(config.output_dir_suffix, "_custom_processed")
        self.assertEqual(config.positions_dir_suffix, "_custom_positions")
        self.assertEqual(config.stitched_dir_suffix, "_custom_stitched")
        self.assertEqual(config.best_focus_dir_suffix, "_custom_best_focus")
        self.assertEqual(config.projections_dir_suffix, "_custom_projections")
        self.assertEqual(config.timepoint_dir_name, "CustomTimePoint")
        self.assertIsNotNone(config.preprocessing_funcs)
        self.assertEqual(config.composite_weights, {"1": 0.7, "2": 0.3})
        self.assertEqual(config.stitcher.tile_overlap, 10.0)
        self.assertEqual(config.focus_analyzer.method, "laplacian")
        self.assertEqual(config.image_preprocessor.composite_weights, {"1": 0.7, "2": 0.3})
        self.assertTrue(config.z_stack_processor.focus_detect)


if __name__ == "__main__":
    unittest.main()
