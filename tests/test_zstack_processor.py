"""
Unit tests for the ZStackProcessor class.
"""

import os
import tempfile
import shutil
from pathlib import Path
import unittest
import numpy as np

from ezstitcher.core.zstack_processor import ZStackProcessor
from ezstitcher.core.config import ZStackProcessorConfig
from ezstitcher.core.file_system_manager import FileSystemManager


class TestZStackProcessor(unittest.TestCase):
    """Test the ZStackProcessor class."""

    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ZStackProcessorConfig(
            focus_detect=True,
            focus_method="combined",
            create_projections=True,
            stitch_z_reference="best_focus",
            save_projections=True,
            stitch_all_z_planes=False,
            projection_types=["max", "mean"]
        )
        self.zstack_processor = ZStackProcessor(self.config)
        self.fs_manager = FileSystemManager()

    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test the initialization of the ZStackProcessor class."""
        self.assertEqual(self.zstack_processor.config.focus_detect, True)
        self.assertEqual(self.zstack_processor.config.focus_method, "combined")
        self.assertEqual(self.zstack_processor.config.create_projections, True)
        self.assertEqual(self.zstack_processor.config.stitch_z_reference, "best_focus")
        self.assertEqual(self.zstack_processor.config.save_projections, True)
        self.assertEqual(self.zstack_processor.config.stitch_all_z_planes, False)
        self.assertEqual(self.zstack_processor.config.projection_types, ["max", "mean"])
        self.assertIsInstance(self.zstack_processor.fs_manager, FileSystemManager)

    def test_detect_z_stacks(self):
        """Test the detect_z_stacks method."""
        # Create test directory structure
        plate_dir = Path(self.temp_dir) / "plate"
        timepoint_dir = plate_dir / "TimePoint_1"
        timepoint_dir.mkdir(parents=True)

        # Create Z-stack folders
        zstack_folders = [
            timepoint_dir / "ZStep_1",
            timepoint_dir / "ZStep_2",
            timepoint_dir / "ZStep_3"
        ]

        for folder in zstack_folders:
            folder.mkdir()

        # Test detection of Z-stack folders
        has_zstack = self.zstack_processor.detect_z_stacks(plate_dir)
        self.assertTrue(has_zstack)

        # Test with no Z-stack folders
        shutil.rmtree(timepoint_dir)
        timepoint_dir.mkdir()
        has_zstack = self.zstack_processor.detect_z_stacks(plate_dir)
        self.assertFalse(has_zstack)

    def test_detect_zstack_images(self):
        """Test the detect_zstack_images method."""
        # Create test directory
        test_dir = Path(self.temp_dir) / "test_zstack_images"
        test_dir.mkdir()

        # Create Z-stack images
        zstack_images = [
            test_dir / "A01_s001_w1_z001.tif",
            test_dir / "A01_s001_w1_z002.tif",
            test_dir / "A01_s001_w1_z003.tif",
            test_dir / "A01_s002_w1_z001.tif",
            test_dir / "A01_s002_w1_z002.tif",
            test_dir / "A01_s002_w1_z003.tif"
        ]

        # Create non-Z-stack images
        non_zstack_images = [
            test_dir / "A01_s001_w2.tif",
            test_dir / "A01_s002_w2.tif"
        ]

        # Create all files
        for file_path in zstack_images + non_zstack_images:
            # Create a small test image
            img = np.zeros((10, 10), dtype=np.uint8)
            self.fs_manager.save_image(file_path, img)

        # Test detection of Z-stack images
        has_zstack, z_indices_map = self.zstack_processor.detect_zstack_images(test_dir)
        self.assertTrue(has_zstack)
        self.assertEqual(len(z_indices_map), 2)  # Two base names (A01_s001_w1 and A01_s002_w1)
        self.assertIn("A01_s001_w1", z_indices_map)
        self.assertIn("A01_s002_w1", z_indices_map)
        self.assertEqual(len(z_indices_map["A01_s001_w1"]), 3)  # Three Z-planes
        self.assertEqual(len(z_indices_map["A01_s002_w1"]), 3)  # Three Z-planes
        self.assertEqual(z_indices_map["A01_s001_w1"], [1, 2, 3])
        self.assertEqual(z_indices_map["A01_s002_w1"], [1, 2, 3])

        # Test with no Z-stack images
        shutil.rmtree(test_dir)
        test_dir.mkdir()
        for file_path in non_zstack_images:
            # Create a small test image
            img = np.zeros((10, 10), dtype=np.uint8)
            self.fs_manager.save_image(file_path, img)

        has_zstack, z_indices_map = self.zstack_processor.detect_zstack_images(test_dir)
        self.assertFalse(has_zstack)
        self.assertEqual(len(z_indices_map), 0)

    def test_pad_site_number(self):
        """Test the pad_site_number method."""
        # Test with single-digit site number
        filename = "A01_s1_w1.tif"
        result = self.zstack_processor.pad_site_number(filename)
        self.assertEqual(result, "A01_s001_w1.tif")

        # Test with double-digit site number
        filename = "A01_s12_w1.tif"
        result = self.zstack_processor.pad_site_number(filename)
        self.assertEqual(result, "A01_s012_w1.tif")

        # Test with triple-digit site number
        filename = "A01_s123_w1.tif"
        result = self.zstack_processor.pad_site_number(filename)
        self.assertEqual(result, "A01_s123_w1.tif")

        # Test with already padded site number
        filename = "A01_s001_w1.tif"
        result = self.zstack_processor.pad_site_number(filename)
        self.assertEqual(result, "A01_s001_w1.tif")

        # Test with non-matching filename
        filename = "other_file.tif"
        result = self.zstack_processor.pad_site_number(filename)
        self.assertEqual(result, "other_file.tif")


if __name__ == "__main__":
    unittest.main()
