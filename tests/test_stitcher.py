"""
Unit tests for the Stitcher class.
"""

import os
import tempfile
import shutil
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.config import StitcherConfig
from ezstitcher.core.file_system_manager import FileSystemManager


class TestStitcher(unittest.TestCase):
    """Test the Stitcher class."""

    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = StitcherConfig(
            tile_overlap=10.0,
            tile_overlap_x=None,
            tile_overlap_y=None,
            max_shift=50,
            margin_ratio=0.1,
            pixel_size=1.0
        )
        self.stitcher = Stitcher(self.config)
        self.fs_manager = FileSystemManager()

    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test the initialization of the Stitcher class."""
        self.assertEqual(self.stitcher.config.tile_overlap, 10.0)
        self.assertIsNone(self.stitcher.config.tile_overlap_x)
        self.assertIsNone(self.stitcher.config.tile_overlap_y)
        self.assertEqual(self.stitcher.config.max_shift, 50)
        self.assertEqual(self.stitcher.config.margin_ratio, 0.1)
        self.assertEqual(self.stitcher.config.pixel_size, 1.0)
        self.assertIsInstance(self.stitcher.fs_manager, FileSystemManager)

    def test_compute_stitched_name(self):
        """Test the compute_stitched_name method."""
        # Test with well and wavelength pattern
        pattern = "A01_s{iii}_w1.tif"
        result = self.stitcher.compute_stitched_name(pattern)
        self.assertEqual(result, "A01_w1.tif")

        # Test with well, wavelength, and z-stack pattern
        pattern = "A01_s{iii}_w1_z001.tif"
        result = self.stitcher.compute_stitched_name(pattern)
        self.assertEqual(result, "A01_w1_z001.tif")

        # Test with composite pattern
        pattern = "composite_A01_s{iii}_w1.tif"
        result = self.stitcher.compute_stitched_name(pattern)
        self.assertEqual(result, "composite_A01_w1.tif")

    def test_generate_positions_df(self):
        """Test the generate_positions_df method."""
        # Create test directory
        test_dir = Path(self.temp_dir) / "test_positions"
        test_dir.mkdir()

        # Create test images
        image_files = [
            test_dir / "A01_s001_w1.tif",
            test_dir / "A01_s002_w1.tif",
            test_dir / "A01_s003_w1.tif",
            test_dir / "A01_s004_w1.tif"
        ]

        # Create all files
        for file_path in image_files:
            # Create a small test image
            img = np.zeros((10, 10), dtype=np.uint8)
            self.fs_manager.save_image(file_path, img)

        # Define positions
        positions = [(0, 0), (100, 0), (0, 100), (100, 100)]

        # Generate positions DataFrame
        pattern = "A01_s{iii}_w1.tif"
        df = self.stitcher.generate_positions_df(test_dir, pattern, positions, 2, 2)

        # Check DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)
        
        # Check file names
        file_names = [row.split(": ")[1] for row in df["file"].values]
        for file_name in ["A01_s001_w1.tif", "A01_s002_w1.tif", "A01_s003_w1.tif", "A01_s004_w1.tif"]:
            self.assertIn(file_name, file_names)

    def test_auto_detect_patterns(self):
        """Test the auto_detect_patterns method."""
        # Create test directory
        test_dir = Path(self.temp_dir) / "test_patterns"
        test_dir.mkdir()

        # Create test images for different wells and wavelengths
        image_files = [
            # Well A01, wavelength 1
            test_dir / "A01_s001_w1.tif",
            test_dir / "A01_s002_w1.tif",
            # Well A01, wavelength 2
            test_dir / "A01_s001_w2.tif",
            test_dir / "A01_s002_w2.tif",
            # Well B02, wavelength 1
            test_dir / "B02_s001_w1.tif",
            test_dir / "B02_s002_w1.tif"
        ]

        # Create all files
        for file_path in image_files:
            # Create a small test image
            img = np.zeros((10, 10), dtype=np.uint8)
            self.fs_manager.save_image(file_path, img)

        # Auto-detect patterns
        patterns = self.stitcher.auto_detect_patterns(test_dir)

        # Check patterns
        self.assertIn("A01", patterns)
        self.assertIn("B02", patterns)
        self.assertEqual(len(patterns["A01"]), 2)  # Two wavelengths
        self.assertEqual(len(patterns["B02"]), 1)  # One wavelength
        self.assertIn("1", patterns["A01"])
        self.assertIn("2", patterns["A01"])
        self.assertIn("1", patterns["B02"])
        self.assertEqual(patterns["A01"]["1"], "A01_s{iii}_w1.tif")
        self.assertEqual(patterns["A01"]["2"], "A01_s{iii}_w2.tif")
        self.assertEqual(patterns["B02"]["1"], "B02_s{iii}_w1.tif")

        # Test with well filter
        patterns = self.stitcher.auto_detect_patterns(test_dir, ["A01"])
        self.assertIn("A01", patterns)
        self.assertNotIn("B02", patterns)

    def test_prepare_reference_channel(self):
        """Test the prepare_reference_channel method."""
        # Create test directory structure
        input_dir = Path(self.temp_dir) / "input"
        processed_dir = Path(self.temp_dir) / "processed"
        positions_dir = Path(self.temp_dir) / "positions"
        stitched_dir = Path(self.temp_dir) / "stitched"

        input_dir.mkdir()
        processed_dir.mkdir()
        positions_dir.mkdir()
        stitched_dir.mkdir()

        dirs = {
            'input': input_dir,
            'processed': processed_dir,
            'positions': positions_dir,
            'stitched': stitched_dir
        }

        # Create test images for different wavelengths
        image_files = [
            # Wavelength 1
            input_dir / "A01_s001_w1.tif",
            input_dir / "A01_s002_w1.tif",
            # Wavelength 2
            input_dir / "A01_s001_w2.tif",
            input_dir / "A01_s002_w2.tif"
        ]

        # Create all files
        for file_path in image_files:
            # Create a small test image
            img = np.zeros((10, 10), dtype=np.uint8)
            self.fs_manager.save_image(file_path, img)

        # Define wavelength patterns
        wavelength_patterns = {
            "1": "A01_s{iii}_w1.tif",
            "2": "A01_s{iii}_w2.tif"
        }

        # Test with single reference channel
        ref_channel, ref_pattern, ref_dir, updated_patterns = self.stitcher.prepare_reference_channel(
            "A01", wavelength_patterns, dirs, ["1"], None, None
        )

        self.assertEqual(ref_channel, "1")
        self.assertEqual(ref_pattern, "A01_s{iii}_w1.tif")
        self.assertEqual(ref_dir, input_dir)
        self.assertEqual(updated_patterns, wavelength_patterns)

        # Test with multiple reference channels (would create composite)
        # This is a more complex test that would require mocking the image_preprocessor
        # For simplicity, we'll just check that the method doesn't crash
        try:
            ref_channel, ref_pattern, ref_dir, updated_patterns = self.stitcher.prepare_reference_channel(
                "A01", wavelength_patterns, dirs, ["1", "2"], None, None
            )
            # If we get here, the method didn't crash
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"prepare_reference_channel raised exception {e}")


if __name__ == "__main__":
    unittest.main()
