"""
Tests for Z-stack handling functionality in ezstitcher.core.z_stack_handler.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
import tifffile

from ezstitcher.core.z_stack_handler import (
    detect_zstack_images, organize_zstack_folders,
    preprocess_plate_folder, create_zstack_projections,
    create_3d_projections
)


class TestZStackHandler(unittest.TestCase):
    def setUp(self):
        """Set up test environment with temporary directory."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.timepoint_dir = os.path.join(self.test_dir, "TimePoint_1")
        os.makedirs(self.timepoint_dir, exist_ok=True)
        
        # Create sample Z-stack images
        self.z_levels = 3
        self.test_image = np.ones((100, 100), dtype=np.uint8) * 100
        
        # Create sample Z-stack files
        for z in range(1, self.z_levels + 1):
            # Create files for well A01, site 1, wavelength 1, different Z levels
            filename = f"A01_s001_w1_z{z:03d}.tif"
            file_path = os.path.join(self.timepoint_dir, filename)
            tifffile.imwrite(file_path, self.test_image * z // 3, compression=None)  # Different intensity for each Z level
            
            # Create files for well A01, site 1, wavelength 2, different Z levels
            filename = f"A01_s001_w2_z{z:03d}.tif"
            file_path = os.path.join(self.timepoint_dir, filename)
            tifffile.imwrite(file_path, self.test_image * z // 3, compression=None)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_detect_zstack_images(self):
        """Test detection of Z-stack pattern in filenames."""
        # Test with Z-stack pattern
        has_zstack, z_indices_map = detect_zstack_images(self.timepoint_dir)
        self.assertTrue(has_zstack)
        self.assertTrue(len(z_indices_map) > 0)
        
        # Test with non-Z-stack pattern
        # Create a new directory without Z-stack files
        no_z_dir = os.path.join(self.test_dir, "No_Z")
        os.makedirs(no_z_dir, exist_ok=True)
        
        # Create regular files (no Z-stack)
        filename = "A01_s001_w1.tif"
        file_path = os.path.join(no_z_dir, filename)
        tifffile.imwrite(file_path, self.test_image)
        
        has_zstack, z_indices_map = detect_zstack_images(no_z_dir)
        self.assertFalse(has_zstack)
        self.assertEqual(len(z_indices_map), 0)

    def test_preprocess_plate_folder(self):
        """Test preprocessing of plate folder with Z-stacks."""
        # Run preprocessing
        has_zstack, z_info = preprocess_plate_folder(self.test_dir)
        
        # Check detection
        self.assertTrue(has_zstack)
        self.assertIsNotNone(z_info)
        
        # Check Z-info structure
        self.assertTrue('z_indices_map' in z_info)
        
        # Check that z_indices_map contains our files
        z_indices_map = z_info['z_indices_map']
        # We should have entries for both A01_s001_w1 and A01_s001_w2
        self.assertEqual(len(z_indices_map), 2)
        # Each entry should have 3 z-indices
        for base_name, indices in z_indices_map.items():
            self.assertEqual(len(indices), self.z_levels)
            # Check that indices are sorted
            self.assertEqual(indices, sorted(indices))

    def test_create_3d_projections(self):
        """Test creation of 3D projections."""
        # Create the output directory
        output_dir = os.path.join(self.test_dir, "projections")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create projections
        projection_types = ['max', 'mean']
        num_projections = create_3d_projections(
            self.timepoint_dir, 
            output_dir, 
            projection_types=projection_types
        )
        
        # Check if the right number of projections were created
        # We should have 2 projections types × 2 wavelengths = 4 projection files
        self.assertEqual(num_projections, 4)
        
        # Check if projection files were created with correct naming
        for proj_type in projection_types:
            for wavelength in ['1', '2']:
                proj_file = f"A01_s001_w{wavelength}_{proj_type}proj.tif"
                proj_path = os.path.join(output_dir, proj_file)
                self.assertTrue(os.path.exists(proj_path), f"Projection file {proj_file} not created")
                
                # Load projection and check dimensions
                proj_img = tifffile.imread(proj_path)
                # Check only height and width, not channels which might vary
                self.assertEqual(proj_img.shape[:2], self.test_image.shape[:2])

    def test_create_zstack_projections(self):
        """Test the wrapper function for creating projections in a plate folder."""
        # Create projections using the high-level function
        projection_types = ['max', 'mean']
        success, proj_dir = create_zstack_projections(
            self.test_dir, 
            projection_types=projection_types
        )
        
        # Check if the operation was successful
        self.assertTrue(success)
        self.assertIsNotNone(proj_dir)
        
        # Check if the expected projection directory exists
        expected_proj_dir = os.path.join(self.test_dir, "TimePoint_1_Projections")
        self.assertEqual(Path(proj_dir), Path(expected_proj_dir))
        
        # Check if projection files were created in the expected directory
        for proj_type in projection_types:
            for wavelength in ['1', '2']:
                proj_file = f"A01_s001_w{wavelength}_{proj_type}proj.tif"
                proj_path = os.path.join(expected_proj_dir, proj_file)
                self.assertTrue(os.path.exists(proj_path), f"Projection file {proj_file} not created")


if __name__ == '__main__':
    unittest.main()
