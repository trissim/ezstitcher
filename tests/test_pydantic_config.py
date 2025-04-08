"""
Unit tests for the Pydantic configuration models.
"""

import unittest
import tempfile
import os
from pathlib import Path

from ezstitcher.core.pydantic_config import (
    StitcherConfig,
    FocusAnalyzerConfig,
    ImagePreprocessorConfig,
    ZStackProcessorConfig,
    PlateProcessorConfig,
    ConfigPresets
)


class TestPydanticConfig(unittest.TestCase):
    """Test the Pydantic configuration models."""

    def test_stitcher_config(self):
        """Test the StitcherConfig model."""
        # Test default values
        config = StitcherConfig()
        self.assertEqual(config.tile_overlap, 6.5)
        self.assertEqual(config.max_shift, 50)
        self.assertEqual(config.margin_ratio, 0.1)
        
        # Test custom values
        config = StitcherConfig(tile_overlap=10.0, max_shift=100, margin_ratio=0.2)
        self.assertEqual(config.tile_overlap, 10.0)
        self.assertEqual(config.max_shift, 100)
        self.assertEqual(config.margin_ratio, 0.2)
        
        # Test validation
        with self.assertRaises(ValueError):
            StitcherConfig(margin_ratio=1.5)  # Should be between 0 and 1
        
        with self.assertRaises(ValueError):
            StitcherConfig(tile_overlap=60)  # Should be between 0 and 50

    def test_focus_analyzer_config(self):
        """Test the FocusAnalyzerConfig model."""
        # Test default values
        config = FocusAnalyzerConfig()
        self.assertEqual(config.method, "combined")
        self.assertIsNone(config.roi)
        
        # Test custom values
        config = FocusAnalyzerConfig(method="laplacian", roi=(10, 10, 100, 100))
        self.assertEqual(config.method, "laplacian")
        self.assertEqual(config.roi, (10, 10, 100, 100))
        
        # Test validation
        with self.assertRaises(ValueError):
            FocusAnalyzerConfig(method="invalid_method")

    def test_image_preprocessor_config(self):
        """Test the ImagePreprocessorConfig model."""
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
        self.assertEqual(config.preprocessing_funcs["1"], dummy_func)
        self.assertEqual(config.composite_weights, {"1": 0.7, "2": 0.3})

    def test_zstack_processor_config(self):
        """Test the ZStackProcessorConfig model."""
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
            stitch_z_reference="best_focus",
            save_projections=False,
            stitch_all_z_planes=True,
            projection_types=["max", "mean"]
        )
        self.assertTrue(config.focus_detect)
        self.assertEqual(config.focus_method, "laplacian")
        self.assertTrue(config.create_projections)
        self.assertEqual(config.stitch_z_reference, "best_focus")
        self.assertFalse(config.save_projections)
        self.assertTrue(config.stitch_all_z_planes)
        self.assertEqual(config.projection_types, ["max", "mean"])
        
        # Test validation
        with self.assertRaises(ValueError):
            ZStackProcessorConfig(focus_method="invalid_method")
        
        with self.assertRaises(ValueError):
            ZStackProcessorConfig(projection_types=["invalid_type"])
        
        with self.assertRaises(ValueError):
            ZStackProcessorConfig(stitch_z_reference="invalid_reference")
        
        # Test custom function
        def dummy_func(img_stack):
            return img_stack[0]
        
        config = ZStackProcessorConfig(stitch_z_reference=dummy_func)
        self.assertEqual(config.stitch_z_reference, dummy_func)

    def test_plate_processor_config(self):
        """Test the PlateProcessorConfig model."""
        # Test default values
        config = PlateProcessorConfig()
        self.assertEqual(config.reference_channels, ["1"])
        self.assertIsNone(config.well_filter)
        self.assertFalse(config.use_reference_positions)
        self.assertEqual(config.output_dir_suffix, "_processed")
        
        # Test nested configurations
        self.assertIsInstance(config.stitcher, StitcherConfig)
        self.assertIsInstance(config.focus_analyzer, FocusAnalyzerConfig)
        self.assertIsInstance(config.image_preprocessor, ImagePreprocessorConfig)
        self.assertIsInstance(config.z_stack_processor, ZStackProcessorConfig)
        
        # Test custom values
        config = PlateProcessorConfig(
            reference_channels=["1", "2"],
            well_filter=["A01", "A02"],
            use_reference_positions=True,
            output_dir_suffix="_custom",
            stitcher=StitcherConfig(tile_overlap=15.0),
            focus_analyzer=FocusAnalyzerConfig(method="laplacian"),
            z_stack_processor=ZStackProcessorConfig(focus_detect=True)
        )
        self.assertEqual(config.reference_channels, ["1", "2"])
        self.assertEqual(config.well_filter, ["A01", "A02"])
        self.assertTrue(config.use_reference_positions)
        self.assertEqual(config.output_dir_suffix, "_custom")
        self.assertEqual(config.stitcher.tile_overlap, 15.0)
        self.assertEqual(config.focus_analyzer.method, "laplacian")
        self.assertTrue(config.z_stack_processor.focus_detect)
        
        # Test validation
        with self.assertRaises(ValueError):
            PlateProcessorConfig(reference_channels=[])

    def test_config_serialization(self):
        """Test configuration serialization to JSON and YAML."""
        config = ConfigPresets.z_stack_best_focus()
        
        # Test JSON serialization
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "config.json")
            config.to_json(json_path)
            
            # Load the config back
            loaded_config = PlateProcessorConfig.from_json(json_path)
            
            # Check that the loaded config matches the original
            self.assertEqual(loaded_config.z_stack_processor.focus_detect, True)
            self.assertEqual(loaded_config.z_stack_processor.focus_method, "combined")
            self.assertEqual(loaded_config.z_stack_processor.stitch_z_reference, "best_focus")
        
        # Test YAML serialization
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = os.path.join(temp_dir, "config.yaml")
            config.to_yaml(yaml_path)
            
            # Load the config back
            loaded_config = PlateProcessorConfig.from_yaml(yaml_path)
            
            # Check that the loaded config matches the original
            self.assertEqual(loaded_config.z_stack_processor.focus_detect, True)
            self.assertEqual(loaded_config.z_stack_processor.focus_method, "combined")
            self.assertEqual(loaded_config.z_stack_processor.stitch_z_reference, "best_focus")

    def test_config_presets(self):
        """Test the configuration presets."""
        # Test default preset
        default_config = ConfigPresets.default()
        self.assertIsInstance(default_config, PlateProcessorConfig)
        
        # Test Z-stack best focus preset
        z_stack_config = ConfigPresets.z_stack_best_focus()
        self.assertTrue(z_stack_config.z_stack_processor.focus_detect)
        self.assertEqual(z_stack_config.z_stack_processor.stitch_z_reference, "best_focus")
        
        # Test Z-stack per-plane preset
        per_plane_config = ConfigPresets.z_stack_per_plane()
        self.assertTrue(per_plane_config.z_stack_processor.stitch_all_z_planes)
        self.assertEqual(per_plane_config.z_stack_processor.stitch_z_reference, "max")
        
        # Test high-resolution preset
        high_res_config = ConfigPresets.high_resolution()
        self.assertEqual(high_res_config.stitcher.tile_overlap, 10.0)
        self.assertEqual(high_res_config.stitcher.max_shift, 100)


if __name__ == "__main__":
    unittest.main()
