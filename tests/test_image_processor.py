"""
Unit tests for the ImageProcessor class.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
import os
import shutil

from ezstitcher.core.image_processor import ImageProcessor
from ezstitcher.core.utils import save_image

class TestImageProcessor(unittest.TestCase):
    """Test the ImageProcessor class."""
    
    def setUp(self):
        """Set up test images."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test images
        self.test_image_8bit = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.test_image_16bit = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        
        # Create a stack of test images
        self.test_stack = [
            self.test_image_16bit,
            np.clip(self.test_image_16bit * 0.8, 0, 65535).astype(np.uint16),
            np.clip(self.test_image_16bit * 1.2, 0, 65535).astype(np.uint16)
        ]
        
        # Save test images to disk
        self.test_image_path = os.path.join(self.temp_dir, "test_image.tif")
        save_image(self.test_image_path, self.test_image_16bit)
        
        # Create test positions CSV
        self.positions_path = os.path.join(self.temp_dir, "positions.csv")
        with open(self.positions_path, 'w') as f:
            f.write("file: test_image.tif; grid: (0, 0); position: (0.0, 0.0)\n")
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_blur(self):
        """Test the blur method."""
        # Apply blur
        blurred = ImageProcessor.blur(self.test_image_16bit, sigma=1)
        
        # Check that the output has the same shape and dtype
        self.assertEqual(blurred.shape, self.test_image_16bit.shape)
        self.assertEqual(blurred.dtype, self.test_image_16bit.dtype)
        
        # Check that the blur reduced the variance
        self.assertLess(np.var(blurred), np.var(self.test_image_16bit))
    
    def test_find_edge(self):
        """Test the find_edge method."""
        # Apply edge detection
        edges = ImageProcessor.find_edge(self.test_image_16bit)
        
        # Check that the output has the same shape and dtype
        self.assertEqual(edges.shape, self.test_image_16bit.shape)
        self.assertEqual(edges.dtype, self.test_image_16bit.dtype)
    
    def test_tophat(self):
        """Test the tophat method."""
        # Apply tophat
        tophat = ImageProcessor.tophat(self.test_image_16bit, selem_radius=5, downsample_factor=2)
        
        # Check that the output has the same shape and dtype
        self.assertEqual(tophat.shape, self.test_image_16bit.shape)
        self.assertEqual(tophat.dtype, self.test_image_16bit.dtype)
    
    def test_create_weighted_composite(self):
        """Test the create_weighted_composite method."""
        # Create test images
        images_dict = {
            "1": self.test_image_16bit,
            "2": np.clip(self.test_image_16bit * 0.8, 0, 65535).astype(np.uint16)
        }
        
        weights_dict = {
            "1": 0.7,
            "2": 0.3
        }
        
        # Create composite
        composite = ImageProcessor.create_weighted_composite(images_dict, weights_dict)
        
        # Check that the output has the same shape and dtype
        self.assertEqual(composite.shape, self.test_image_16bit.shape)
        self.assertEqual(composite.dtype, self.test_image_16bit.dtype)
        
        # Check with default weights
        composite_default = ImageProcessor.create_weighted_composite(images_dict)
        self.assertEqual(composite_default.shape, self.test_image_16bit.shape)
    
    def test_normalize_16bit_global(self):
        """Test the normalize_16bit_global method."""
        # Normalize images
        normalized = ImageProcessor.normalize_16bit_global(
            self.test_stack, 
            lower_percentile=1, 
            upper_percentile=99
        )
        
        # Check that the output has the same length, shape, and dtype
        self.assertEqual(len(normalized), len(self.test_stack))
        self.assertEqual(normalized[0].shape, self.test_stack[0].shape)
        self.assertEqual(normalized[0].dtype, self.test_stack[0].dtype)
        
        # Check that the normalization worked
        for img in normalized:
            self.assertGreaterEqual(np.min(img), 0)
            self.assertLessEqual(np.max(img), 65535)
    
    def test_hist_match_stack(self):
        """Test the hist_match_stack method."""
        # Match histograms
        matched = ImageProcessor.hist_match_stack(self.test_stack)
        
        # Check that the output has the same length, shape, and dtype
        self.assertEqual(len(matched), len(self.test_stack))
        self.assertEqual(matched[0].shape, self.test_stack[0].shape)
        self.assertEqual(matched[0].dtype, self.test_stack[0].dtype)
        
        # Check with reference image
        reference = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        matched_ref = ImageProcessor.hist_match_stack(self.test_stack, reference)
        self.assertEqual(len(matched_ref), len(self.test_stack))
    
    def test_process_bf(self):
        """Test the process_bf method."""
        # Process brightfield images
        processed = ImageProcessor.process_bf(self.test_stack)
        
        # Check that the output has the same length, shape, and dtype
        self.assertEqual(len(processed), len(self.test_stack))
        self.assertEqual(processed[0].shape, self.test_stack[0].shape)
        self.assertEqual(processed[0].dtype, self.test_stack[0].dtype)
    
    def test_assemble_image_subpixel(self):
        """Test the assemble_image_subpixel method."""
        # Create output path
        output_path = os.path.join(self.temp_dir, "stitched.tif")
        
        # Assemble image
        result = ImageProcessor.assemble_image_subpixel(
            positions_path=self.positions_path,
            images_dir=self.temp_dir,
            output_path=output_path,
            margin_ratio=0.1
        )
        
        # Check that the function returned True
        self.assertTrue(result)
        
        # Check that the output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Test with override_names
        output_path2 = os.path.join(self.temp_dir, "stitched2.tif")
        result2 = ImageProcessor.assemble_image_subpixel(
            positions_path=self.positions_path,
            images_dir=self.temp_dir,
            output_path=output_path2,
            margin_ratio=0.1,
            override_names=["test_image.tif"]
        )
        
        # Check that the function returned True
        self.assertTrue(result2)
        
        # Check that the output file exists
        self.assertTrue(os.path.exists(output_path2))


if __name__ == "__main__":
    unittest.main()
