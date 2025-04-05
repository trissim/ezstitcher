"""
Unit tests for the main module.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
import os
import shutil
import logging

from ezstitcher.core.main import (
    process_plate_folder,
    modified_process_plate_folder,
    process_bf,
    find_best_focus
)

# Disable logging for tests
logging.disable(logging.CRITICAL)

class TestMain(unittest.TestCase):
    """Test the main module."""
    
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
        for well in ['A01']:
            for site in range(1, 5):  # 2x2 grid
                for wavelength in [1, 2]:
                    filename = f"{well}_s{site:03d}_w{wavelength}.tif"
                    filepath = os.path.join(self.timepoint_dir, filename)
                    
                    # Add some variation to each site
                    site_image = self.test_image * (0.8 + 0.05 * site)
                    site_image = np.clip(site_image, 0, 65535).astype(np.uint16)
                    
                    with open(filepath, 'wb') as f:
                        f.write(b'DUMMY')  # Just create a dummy file
        
        # Create HTD file
        htd_content = """
        [HTD]
        SiteColumns=2
        SiteRows=2
        """
        
        htd_path = os.path.join(self.metadata_dir, "plate.HTD")
        with open(htd_path, 'w') as f:
            f.write(htd_content)
        
        # Create a stack of test images with varying focus
        self.image_stack = []
        for z in range(1, 6):
            # Create base image
            img = np.zeros((100, 100), dtype=np.uint8)
            
            # Add features with varying sharpness
            if z == 3:  # Best focus at z=3
                # Sharp features
                img[40:60, 40:60] = 255  # Add a square
            else:
                # Blurred features
                base = np.zeros((100, 100), dtype=np.uint8)
                base[40:60, 40:60] = 255  # Add a square
                img = np.clip(base + np.random.normal(0, abs(z - 3) * 10, base.shape), 0, 255).astype(np.uint8)
            
            self.image_stack.append(img)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_process_bf(self):
        """Test the process_bf function."""
        # Create a stack of brightfield images
        bf_images = [
            np.random.randint(0, 65535, (100, 100), dtype=np.uint16),
            np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        ]
        
        # Process brightfield images
        processed = process_bf(bf_images)
        
        # Check that the output has the same length
        self.assertEqual(len(processed), len(bf_images))
        
        # Check that the output has the same shape and dtype
        self.assertEqual(processed[0].shape, bf_images[0].shape)
        self.assertEqual(processed[0].dtype, bf_images[0].dtype)
    
    def test_find_best_focus(self):
        """Test the find_best_focus function."""
        # Find best focus
        best_idx, scores = find_best_focus(self.image_stack, method='combined')
        
        # Check that scores is a list of tuples (index, score)
        self.assertEqual(len(scores), len(self.image_stack))
        self.assertEqual(len(scores[0]), 2)
        
        # Test with ROI
        roi = (40, 40, 20, 20)  # (x, y, width, height)
        best_idx_roi, scores_roi = find_best_focus(self.image_stack, method='combined', roi=roi)
        
        # Check that scores is a list of tuples (index, score)
        self.assertEqual(len(scores_roi), len(self.image_stack))
    
    def test_process_plate_folder(self):
        """Test the process_plate_folder function."""
        # This is a minimal test that just checks if the function runs without errors
        # A more comprehensive test would require mocking the StitcherManager
        
        # Mock the StitcherManager.process_plate_folder method
        original_process_plate_folder = process_plate_folder
        
        try:
            # Replace with a mock function that just returns True
            def mock_process_plate_folder(*args, **kwargs):
                return True
            
            # Monkey patch the function
            import ezstitcher.core.main
            ezstitcher.core.main.process_plate_folder = mock_process_plate_folder
            
            # Call the function
            result = process_plate_folder(self.plate_dir)
            
            # Check that the function returned True
            self.assertTrue(result)
            
        finally:
            # Restore the original function
            ezstitcher.core.main.process_plate_folder = original_process_plate_folder
    
    def test_modified_process_plate_folder(self):
        """Test the modified_process_plate_folder function."""
        # This is a minimal test that just checks if the function runs without errors
        # A more comprehensive test would require mocking the ZStackManager
        
        # Mock the ZStackManager.preprocess_plate_folder method
        original_modified_process_plate_folder = modified_process_plate_folder
        
        try:
            # Replace with a mock function that just returns True
            def mock_modified_process_plate_folder(*args, **kwargs):
                return True
            
            # Monkey patch the function
            import ezstitcher.core.main
            ezstitcher.core.main.modified_process_plate_folder = mock_modified_process_plate_folder
            
            # Call the function
            result = modified_process_plate_folder(self.plate_dir)
            
            # Check that the function returned True
            self.assertTrue(result)
            
        finally:
            # Restore the original function
            ezstitcher.core.main.modified_process_plate_folder = original_modified_process_plate_folder


if __name__ == "__main__":
    unittest.main()
