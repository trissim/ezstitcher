"""
Unit tests for the FocusDetector class.
"""

import unittest
import numpy as np
import cv2

from ezstitcher.core.focus_detector import FocusDetector

class TestFocusDetector(unittest.TestCase):
    """Test the FocusDetector class."""
    
    def setUp(self):
        """Set up test images."""
        # Create a base image (100x100)
        self.base_image = np.zeros((100, 100), dtype=np.uint8)
        
        # Add some features to the base image
        cv2.rectangle(self.base_image, (20, 20), (80, 80), 255, 2)
        cv2.circle(self.base_image, (50, 50), 20, 128, -1)
        
        # Create a stack of images with varying blur levels
        self.image_stack = []
        for sigma in [0, 1, 2, 3, 4]:
            if sigma == 0:
                # Sharp image (no blur)
                self.image_stack.append(self.base_image.copy())
            else:
                # Blurred image
                blurred = cv2.GaussianBlur(self.base_image, (0, 0), sigma)
                self.image_stack.append(blurred)
    
    def test_original_fft_focus(self):
        """Test the original_fft_focus method."""
        # Calculate focus scores for all images
        scores = [FocusDetector.original_fft_focus(img) for img in self.image_stack]
        
        # The first image (no blur) should have the highest score
        self.assertEqual(np.argmax(scores), 0)
    
    def test_adaptive_fft_focus(self):
        """Test the adaptive_fft_focus method."""
        # Calculate focus scores for all images
        scores = [FocusDetector.adaptive_fft_focus(img) for img in self.image_stack]
        
        # The first image (no blur) should have the highest score
        self.assertEqual(np.argmax(scores), 0)
    
    def test_normalized_variance(self):
        """Test the normalized_variance method."""
        # Calculate focus scores for all images
        scores = [FocusDetector.normalized_variance(img) for img in self.image_stack]
        
        # The first image (no blur) should have the highest score
        self.assertEqual(np.argmax(scores), 0)
    
    def test_laplacian_energy(self):
        """Test the laplacian_energy method."""
        # Calculate focus scores for all images
        scores = [FocusDetector.laplacian_energy(img) for img in self.image_stack]
        
        # The first image (no blur) should have the highest score
        self.assertEqual(np.argmax(scores), 0)
    
    def test_tenengrad_variance(self):
        """Test the tenengrad_variance method."""
        # Calculate focus scores for all images
        scores = [FocusDetector.tenengrad_variance(img) for img in self.image_stack]
        
        # The first image (no blur) should have the highest score
        self.assertEqual(np.argmax(scores), 0)
    
    def test_combined_focus_measure(self):
        """Test the combined_focus_measure method."""
        # Calculate focus scores for all images
        scores = [FocusDetector.combined_focus_measure(img) for img in self.image_stack]
        
        # The first image (no blur) should have the highest score
        self.assertEqual(np.argmax(scores), 0)
        
        # Test with custom weights
        weights = {
            'nvar': 0.5,
            'lap': 0.2,
            'ten': 0.2,
            'fft': 0.1
        }
        scores_custom = [FocusDetector.combined_focus_measure(img, weights) for img in self.image_stack]
        
        # The first image (no blur) should still have the highest score
        self.assertEqual(np.argmax(scores_custom), 0)
    
    def test_find_best_focus(self):
        """Test the find_best_focus method."""
        # Find best focus using different methods
        for method in ['combined', 'nvar', 'lap', 'ten', 'fft', 'adaptive_fft']:
            best_idx, scores = FocusDetector.find_best_focus(self.image_stack, method)
            
            # The first image (no blur) should be the best focused
            self.assertEqual(best_idx, 0)
            
            # Check that scores is a list of tuples (index, score)
            self.assertEqual(len(scores), len(self.image_stack))
            self.assertEqual(len(scores[0]), 2)
        
        # Test with ROI
        roi = (25, 25, 50, 50)  # (x, y, width, height)
        best_idx_roi, scores_roi = FocusDetector.find_best_focus(self.image_stack, 'combined', roi)
        
        # The first image should still be the best focused
        self.assertEqual(best_idx_roi, 0)
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            FocusDetector.find_best_focus(self.image_stack, 'invalid_method')


if __name__ == "__main__":
    unittest.main()
