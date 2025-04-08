"""
Integration tests for the configuration system.
"""

import unittest
import tempfile
import os
from pathlib import Path
import json
import yaml

from ezstitcher.core.pydantic_config import (
    PlateProcessorConfig,
    ConfigPresets
)
from ezstitcher.core.main import process_plate_folder_with_config


class TestConfigIntegration(unittest.TestCase):
    """Test the integration of the configuration system with the processing pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a mock plate folder structure
        self.plate_folder = self.temp_path / "test_plate"
        self.plate_folder.mkdir()
        
        # Create TimePoint_1 directory
        self.timepoint_dir = self.plate_folder / "TimePoint_1"
        self.timepoint_dir.mkdir()
        
        # Create some dummy image files
        (self.timepoint_dir / "A01_s001_w1.tif").touch()
        (self.timepoint_dir / "A01_s001_w2.tif").touch()
        (self.timepoint_dir / "A01_s002_w1.tif").touch()
        (self.timepoint_dir / "A01_s002_w2.tif").touch()
        
        # Create a JSON configuration file
        self.json_config = self.temp_path / "config.json"
        config = ConfigPresets.default()
        config.reference_channels = ["1", "2"]
        config.well_filter = ["A01"]
        config.to_json(self.json_config)
        
        # Create a YAML configuration file
        self.yaml_config = self.temp_path / "config.yaml"
        config.to_yaml(self.yaml_config)

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_process_with_json_config(self):
        """Test processing with a JSON configuration file."""
        # This is a mock test that doesn't actually process images
        # In a real test, you would check for the existence of output files
        try:
            process_plate_folder_with_config(
                self.plate_folder,
                config_file=self.json_config
            )
            # If we get here without an exception, consider it a success
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"process_plate_folder_with_config raised {type(e).__name__} unexpectedly: {e}")

    def test_process_with_yaml_config(self):
        """Test processing with a YAML configuration file."""
        try:
            process_plate_folder_with_config(
                self.plate_folder,
                config_file=self.yaml_config
            )
            # If we get here without an exception, consider it a success
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"process_plate_folder_with_config raised {type(e).__name__} unexpectedly: {e}")

    def test_process_with_preset(self):
        """Test processing with a configuration preset."""
        try:
            process_plate_folder_with_config(
                self.plate_folder,
                config_preset="z_stack_best_focus"
            )
            # If we get here without an exception, consider it a success
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"process_plate_folder_with_config raised {type(e).__name__} unexpectedly: {e}")

    def test_process_with_kwargs_override(self):
        """Test processing with kwargs overriding configuration values."""
        try:
            process_plate_folder_with_config(
                self.plate_folder,
                config_preset="default",
                reference_channels=["2"],
                well_filter=["A02"]
            )
            # If we get here without an exception, consider it a success
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"process_plate_folder_with_config raised {type(e).__name__} unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main()
