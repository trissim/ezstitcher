"""
Unit tests for the StitcherManager class.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
import os
import shutil

from ezstitcher.core.stitcher_manager import StitcherManager
from ezstitcher.core.utils import save_image

class TestStitcherManager(unittest.TestCase):
    """Test the StitcherManager class."""

    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.plate_dir = os.path.join(self.temp_dir, "plate")
        self.timepoint_dir = os.path.join(self.plate_dir, "TimePoint_1")
        self.metadata_dir = os.path.join(self.plate_dir, "MetaData")

        os.makedirs(self.timepoint_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        # Create test images
        self.test_image = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)

        # Create images for different wells, sites, and wavelengths
        for well in ['A01', 'B02']:
            for site in range(1, 10):  # 3x3 grid
                for wavelength in [1, 2]:
                    filename = f"{well}_s{site:03d}_w{wavelength}.tif"
                    filepath = os.path.join(self.timepoint_dir, filename)

                    # Add some variation to each site
                    site_image = self.test_image * (0.8 + 0.05 * site)
                    site_image = np.clip(site_image, 0, 65535).astype(np.uint16)

                    save_image(filepath, site_image)

        # Create HTD file
        htd_content = """
        [HTD]
        SiteColumns=3
        SiteRows=3
        """

        htd_path = os.path.join(self.metadata_dir, "plate.HTD")
        with open(htd_path, 'w') as f:
            f.write(htd_content)

        # Create positions CSV
        positions_dir = os.path.join(self.temp_dir, "plate_positions")
        os.makedirs(positions_dir, exist_ok=True)

        positions_content = """
        file: A01_s001_w1.tif; grid: (0, 0); position: (0.0, 0.0)
        file: A01_s002_w1.tif; grid: (1, 0); position: (100.0, 0.0)
        file: A01_s003_w1.tif; grid: (2, 0); position: (200.0, 0.0)
        file: A01_s004_w1.tif; grid: (0, 1); position: (0.0, 100.0)
        file: A01_s005_w1.tif; grid: (1, 1); position: (100.0, 100.0)
        file: A01_s006_w1.tif; grid: (2, 1); position: (200.0, 100.0)
        file: A01_s007_w1.tif; grid: (0, 2); position: (0.0, 200.0)
        file: A01_s008_w1.tif; grid: (1, 2); position: (100.0, 200.0)
        file: A01_s009_w1.tif; grid: (2, 2); position: (200.0, 200.0)
        """

        positions_path = os.path.join(positions_dir, "A01_w1.csv")
        with open(positions_path, 'w') as f:
            f.write(positions_content)

        self.positions_dir = positions_dir

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_find_HTD_file(self):
        """Test the find_HTD_file method."""
        # Find HTD file
        htd_file = StitcherManager.find_HTD_file(self.plate_dir)

        # Check that HTD file was found
        self.assertIsNotNone(htd_file)
        self.assertTrue(os.path.exists(htd_file))
        self.assertEqual(os.path.basename(htd_file), "plate.HTD")

    def test_parse_HTD_file(self):
        """Test the parse_HTD_file method."""
        # Find HTD file
        htd_file = StitcherManager.find_HTD_file(self.plate_dir)

        # Parse HTD file
        grid_size_x, grid_size_y = StitcherManager.parse_HTD_file(htd_file)

        # Check that grid dimensions were parsed correctly
        self.assertEqual(grid_size_x, 3)
        self.assertEqual(grid_size_y, 3)

    def test_auto_detect_patterns(self):
        """Test the auto_detect_patterns method."""
        # Detect patterns
        patterns = StitcherManager.auto_detect_patterns(self.timepoint_dir)

        # Check that patterns were detected for all wells
        self.assertEqual(len(patterns), 2)
        self.assertIn('A01', patterns)
        self.assertIn('B02', patterns)

        # Check that patterns were detected for all wavelengths
        for well in ['A01', 'B02']:
            self.assertEqual(len(patterns[well]), 2)
            self.assertIn('1', patterns[well])
            self.assertIn('2', patterns[well])

        # Check with well filter
        patterns_filtered = StitcherManager.auto_detect_patterns(
            self.timepoint_dir,
            well_filter=['A01']
        )

        self.assertEqual(len(patterns_filtered), 1)
        self.assertIn('A01', patterns_filtered)

    def test_compute_stitched_name(self):
        """Test the compute_stitched_name method."""
        # Test with site placeholder
        pattern = "A01_s{iii}_w1.tif"
        stitched_name = StitcherManager.compute_stitched_name(pattern)
        self.assertEqual(stitched_name, "A01_stitched_w1.tif")

        # Test without site placeholder
        pattern = "A01_w1.tif"
        stitched_name = StitcherManager.compute_stitched_name(pattern)
        self.assertEqual(stitched_name, "A01_w1_stitched.tif")

    def test_prepare_reference_channel(self):
        """Test the prepare_reference_channel method."""
        # Create directories
        dirs = {
            'input': self.timepoint_dir,
            'processed': os.path.join(self.temp_dir, "processed"),
            'positions': os.path.join(self.temp_dir, "positions"),
            'stitched': os.path.join(self.temp_dir, "stitched")
        }

        # Create processed directory
        os.makedirs(dirs['processed'], exist_ok=True)

        # Get patterns for a well
        patterns = StitcherManager.auto_detect_patterns(self.timepoint_dir)
        well = 'A01'
        wavelength_patterns = patterns[well]

        # Test with single reference channel
        ref_channel, ref_pattern, ref_dir, updated_patterns = StitcherManager.prepare_reference_channel(
            well,
            wavelength_patterns,
            dirs,
            reference_channels=['1']
        )

        self.assertEqual(ref_channel, '1')
        self.assertEqual(ref_pattern, wavelength_patterns['1'])
        self.assertEqual(ref_dir, dirs['input'])
        self.assertEqual(updated_patterns, wavelength_patterns)

        # Test with multiple reference channels
        ref_channel, ref_pattern, ref_dir, updated_patterns = StitcherManager.prepare_reference_channel(
            well,
            wavelength_patterns,
            dirs,
            reference_channels=['1', '2']
        )

        self.assertEqual(ref_channel, 'composite')
        self.assertIn('composite', updated_patterns)
        self.assertEqual(str(ref_dir), dirs['processed'])

    def test_process_imgs_from_pattern(self):
        """Test the process_imgs_from_pattern method."""
        # Create output directory
        out_dir = os.path.join(self.temp_dir, "processed")
        os.makedirs(out_dir, exist_ok=True)

        # Define a simple processing function
        def process_func(images):
            return [img // 2 for img in images]

        # Process images
        pattern = "A01_s{iii}_w1.tif"
        count = StitcherManager.process_imgs_from_pattern(
            self.timepoint_dir,
            pattern,
            process_func,
            out_dir
        )

        # Check that all images were processed
        self.assertEqual(count, 9)

        # Check that output files exist
        for site in range(1, 10):
            filename = f"A01_s{site:03d}_w1.tif"
            filepath = os.path.join(out_dir, filename)
            self.assertTrue(os.path.exists(filepath))


if __name__ == "__main__":
    unittest.main()
