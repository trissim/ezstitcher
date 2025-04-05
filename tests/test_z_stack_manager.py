"""
Unit tests for the ZStackManager class.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
import os
import shutil
import cv2

from ezstitcher.core.z_stack_manager import ZStackManager
from ezstitcher.core.utils import save_image

class TestZStackManager(unittest.TestCase):
    """Test the ZStackManager class."""

    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.timepoint_dir = os.path.join(self.temp_dir, "TimePoint_1")
        os.makedirs(self.timepoint_dir, exist_ok=True)

        # Create test images
        self.test_image = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)

        # Create Z-stack images
        for z in range(1, 6):
            # Create images for different wells, sites, and wavelengths
            for well in ['A01', 'B02']:
                for site in [1, 2]:
                    for wavelength in [1, 2]:
                        filename = f"{well}_s{site:03d}_w{wavelength}_z{z:03d}.tif"
                        filepath = os.path.join(self.timepoint_dir, filename)

                        # Add some variation to each Z-plane
                        z_image = self.test_image * (0.8 + 0.1 * z)
                        z_image = np.clip(z_image, 0, 65535).astype(np.uint16)

                        save_image(filepath, z_image)

        # Create Z-stack folders
        for z in range(1, 4):
            z_folder = os.path.join(self.timepoint_dir, f"ZStep_{z}")
            os.makedirs(z_folder, exist_ok=True)

            # Create images in Z-stack folders
            for well in ['C03', 'D04']:
                for site in [1, 2]:
                    for wavelength in [1, 2]:
                        filename = f"{well}_s{site:03d}_w{wavelength}.tif"
                        filepath = os.path.join(z_folder, filename)

                        # Add some variation to each Z-plane
                        z_image = self.test_image * (0.8 + 0.1 * z)
                        z_image = np.clip(z_image, 0, 65535).astype(np.uint16)

                        save_image(filepath, z_image)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_detect_zstack_folders(self):
        """Test the detect_zstack_folders method."""
        # Detect Z-stack folders
        has_zstack, z_folders = ZStackManager.detect_zstack_folders(self.temp_dir)

        # Check that Z-stack folders were detected
        self.assertTrue(has_zstack)
        self.assertEqual(len(z_folders), 3)

        # Check that Z-indices are correct
        z_indices = [z for z, _ in z_folders]
        self.assertEqual(z_indices, [1, 2, 3])

    def test_organize_zstack_folders(self):
        """Test the organize_zstack_folders method."""
        # Organize Z-stack folders
        result = ZStackManager.organize_zstack_folders(self.temp_dir)

        # Check that the function returned True
        self.assertTrue(result)

        # Check that files were moved and renamed
        for well in ['C03', 'D04']:
            for site in [1, 2]:
                for wavelength in [1, 2]:
                    for z in range(1, 4):
                        filename = f"{well}_s{site:03d}_w{wavelength}_z{z:03d}.tif"
                        filepath = os.path.join(self.timepoint_dir, filename)
                        self.assertTrue(os.path.exists(filepath))

    def test_detect_zstack_images(self):
        """Test the detect_zstack_images method."""
        # Detect Z-stack images
        has_zstack, z_indices_map = ZStackManager.detect_zstack_images(self.timepoint_dir)

        # Check that Z-stack images were detected
        self.assertTrue(has_zstack)

        # Check that all base names are in the map
        for well in ['A01', 'B02']:
            for site in [1, 2]:
                for wavelength in [1, 2]:
                    base_name = f"{well}_s{site:03d}_w{wavelength}"
                    self.assertIn(base_name, z_indices_map)

                    # Check that all Z-indices are present
                    self.assertEqual(z_indices_map[base_name], [1, 2, 3, 4, 5])

    def test_load_image_stack(self):
        """Test the load_image_stack method."""
        # Load an image stack
        base_name = "A01_s001_w1"
        z_indices = [1, 2, 3, 4, 5]

        image_stack = ZStackManager.load_image_stack(
            self.timepoint_dir,
            base_name,
            z_indices,
            file_ext=".tif"
        )

        # Check that all images were loaded
        self.assertEqual(len(image_stack), 5)

        # Check that Z-indices are correct
        loaded_z_indices = [z for z, _ in image_stack]
        self.assertEqual(loaded_z_indices, z_indices)

        # Check image shapes and dtypes
        for _, img in image_stack:
            self.assertEqual(img.shape, (100, 100))
            self.assertEqual(img.dtype, np.uint16)

        # Test auto-detection of file extension
        image_stack_auto = ZStackManager.load_image_stack(
            self.timepoint_dir,
            base_name,
            z_indices
        )

        self.assertEqual(len(image_stack_auto), 5)

    def test_create_projection(self):
        """Test the create_projection method."""
        # Create a stack of test images
        images = [
            np.ones((10, 10), dtype=np.uint16) * 1000,
            np.ones((10, 10), dtype=np.uint16) * 2000,
            np.ones((10, 10), dtype=np.uint16) * 3000
        ]

        # Test different projection types
        proj_max = ZStackManager.create_projection(images, 'max')
        self.assertEqual(proj_max.shape, (10, 10))
        self.assertEqual(proj_max.dtype, np.uint16)
        self.assertEqual(np.max(proj_max), 3000)

        proj_min = ZStackManager.create_projection(images, 'min')
        self.assertEqual(np.min(proj_min), 1000)

        proj_mean = ZStackManager.create_projection(images, 'mean')
        self.assertEqual(np.mean(proj_mean), 2000)

        proj_std = ZStackManager.create_projection(images, 'std')
        self.assertGreater(np.mean(proj_std), 0)

        proj_sum = ZStackManager.create_projection(images, 'sum')
        self.assertEqual(np.mean(proj_sum), 6000)

        # Test invalid projection type
        proj_invalid = ZStackManager.create_projection(images, 'invalid')
        self.assertIsNone(proj_invalid)

    def test_create_zstack_projections(self):
        """Test the create_zstack_projections method."""
        # Create output directory
        output_dir = os.path.join(self.temp_dir, "projections")

        # Create projections
        projections = ZStackManager.create_zstack_projections(
            self.timepoint_dir,
            output_dir,
            projection_types=['max', 'mean']
        )

        # Check that projections were created
        self.assertGreater(len(projections), 0)

        # Check that output files exist
        for base_name, proj_list in projections.items():
            for proj_type, proj_path in proj_list:
                self.assertTrue(os.path.exists(proj_path))

    def test_find_best_focus_in_stack(self):
        """Test the find_best_focus_in_stack method."""
        # Create a stack of test images with varying focus
        image_stack = []
        for z in range(1, 6):
            # Create base image
            img = np.zeros((100, 100), dtype=np.uint8)

            # Add features with varying sharpness
            if z == 3:  # Best focus at z=3
                # Sharp features
                cv2.rectangle(img, (20, 20), (80, 80), 255, 2)
                cv2.circle(img, (50, 50), 20, 128, -1)
            else:
                # Blurred features
                base = np.zeros((100, 100), dtype=np.uint8)
                cv2.rectangle(base, (20, 20), (80, 80), 255, 2)
                cv2.circle(base, (50, 50), 20, 128, -1)
                img = cv2.GaussianBlur(base, (0, 0), abs(z - 3) * 2)

            image_stack.append((z, img))

        # Find best focus
        best_z, best_img, scores = ZStackManager.find_best_focus_in_stack(
            image_stack,
            method='combined'
        )

        # Check that the best focus is at z=3
        self.assertEqual(best_z, 3)

        # Check with ROI
        best_z_roi, _, _ = ZStackManager.find_best_focus_in_stack(
            image_stack,
            method='combined',
            roi=(25, 25, 50, 50)
        )

        self.assertEqual(best_z_roi, 3)

    def test_preprocess_plate_folder(self):
        """Test the preprocess_plate_folder method."""
        # Preprocess plate folder
        has_zstack, z_info = ZStackManager.preprocess_plate_folder(self.temp_dir)

        # Check that Z-stack was detected
        self.assertTrue(has_zstack)

        # Check that Z-info contains the expected keys
        self.assertIn('has_zstack_folders', z_info)
        self.assertIn('z_folders', z_info)
        self.assertIn('has_zstack_images', z_info)
        self.assertIn('z_indices_map', z_info)

        # Check that Z-stack folders were detected
        self.assertTrue(z_info['has_zstack_folders'])

        # Check that Z-stack images were detected
        self.assertTrue(z_info['has_zstack_images'])


if __name__ == "__main__":
    unittest.main()
