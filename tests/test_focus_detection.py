"""
Tests for focus detection algorithms in ezstitcher.core.focus_detect.
"""

import unittest
import numpy as np
import cv2
from skimage import data, filters

from ezstitcher.core.focus_detect import (
    normalized_variance, laplacian_energy, tenengrad_variance,
    original_fft_focus, adaptive_fft_focus, combined_focus_measure,
    find_best_focus
)


class TestFocusDetection(unittest.TestCase):
    def setUp(self):
        """Set up test images with varying focus levels."""
        # Create a base image
        base_img = data.camera()
        
        # Create a stack of images with varying blur levels to simulate focus
        self.focused_img = base_img.copy()
        
        # Create increasingly blurred versions to simulate out-of-focus images
        self.slightly_blurred = cv2.GaussianBlur(base_img, (5, 5), 1)
        self.very_blurred = cv2.GaussianBlur(base_img, (15, 15), 3)
        
        # Create image stack from sharp to blurry
        self.test_stack = [
            self.focused_img,
            self.slightly_blurred,
            self.very_blurred
        ]

    def test_normalized_variance(self):
        """Test normalized variance focus measure."""
        # Calculate focus scores
        focused_score = normalized_variance(self.focused_img)
        slight_blur_score = normalized_variance(self.slightly_blurred)
        very_blur_score = normalized_variance(self.very_blurred)
        
        # Focused image should have higher score than blurred ones
        self.assertGreater(focused_score, slight_blur_score)
        self.assertGreater(slight_blur_score, very_blur_score)

    def test_laplacian_energy(self):
        """Test Laplacian energy focus measure."""
        # Calculate focus scores
        focused_score = laplacian_energy(self.focused_img)
        slight_blur_score = laplacian_energy(self.slightly_blurred)
        very_blur_score = laplacian_energy(self.very_blurred)
        
        # Focused image should have higher score than blurred ones
        self.assertGreater(focused_score, slight_blur_score)
        self.assertGreater(slight_blur_score, very_blur_score)

    def test_tenengrad_variance(self):
        """Test Tenengrad variance focus measure."""
        # Calculate focus scores
        focused_score = tenengrad_variance(self.focused_img)
        slight_blur_score = tenengrad_variance(self.slightly_blurred)
        very_blur_score = tenengrad_variance(self.very_blurred)
        
        # Focused image should have higher score than blurred ones
        self.assertGreater(focused_score, slight_blur_score)
        self.assertGreater(slight_blur_score, very_blur_score)

    def test_original_fft_focus(self):
        """Test original FFT-based focus measure."""
        # Calculate focus scores
        focused_score = original_fft_focus(self.focused_img)
        slight_blur_score = original_fft_focus(self.slightly_blurred)
        very_blur_score = original_fft_focus(self.very_blurred)
        
        # Focused image should have higher score than blurred ones
        self.assertGreater(focused_score, slight_blur_score)
        self.assertGreater(slight_blur_score, very_blur_score)

    def test_adaptive_fft_focus(self):
        """Test adaptive FFT-based focus measure."""
        # Calculate focus scores
        focused_score = adaptive_fft_focus(self.focused_img)
        slight_blur_score = adaptive_fft_focus(self.slightly_blurred)
        very_blur_score = adaptive_fft_focus(self.very_blurred)
        
        # Focused image should have higher score than blurred ones
        self.assertGreater(focused_score, slight_blur_score)
        self.assertGreater(slight_blur_score, very_blur_score)

    def test_combined_focus_measure(self):
        """Test combined focus measure."""
        # Calculate focus scores
        focused_score = combined_focus_measure(self.focused_img)
        slight_blur_score = combined_focus_measure(self.slightly_blurred)
        very_blur_score = combined_focus_measure(self.very_blurred)
        
        # Focused image should have higher score than blurred ones
        self.assertGreater(focused_score, slight_blur_score)
        self.assertGreater(slight_blur_score, very_blur_score)

    def test_find_best_focus(self):
        """Test find_best_focus function with different methods."""
        # Test with different focus methods
        methods = ['combined', 'nvar', 'lap', 'ten', 'fft', 'adaptive_fft']
        
        for method in methods:
            best_idx, scores = find_best_focus(self.test_stack, method=method)
            
            # Best focus should be the first image (index 0)
            self.assertEqual(best_idx, 0, f"Method {method} failed to identify best focus")
            
            # Extract the scores from the list of tuples to compare them
            # The format changed from a simple list to a list of (index, score) tuples
            score_values = [score[1] for score in scores]
            
            # Scores should be in descending order
            self.assertGreaterEqual(score_values[0], score_values[1], 
                                   f"Method {method} scores not in expected order: {scores}")
            self.assertGreaterEqual(score_values[1], score_values[2], 
                                   f"Method {method} scores not in expected order: {scores}")


if __name__ == '__main__':
    unittest.main()
