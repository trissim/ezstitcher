import numpy as np
import cv2
import logging
from ezstitcher.core.config import FocusAnalyzerConfig
from ezstitcher.core.file_system_manager import FileSystemManager

logger = logging.getLogger(__name__)

class FocusAnalyzer:
    """
    Provides focus metrics and best focus selection.

    This class implements various focus measure algorithms and methods to find
    the best focused image in a Z-stack. It uses the FileSystemManager for
    image handling to avoid code duplication.
    """
    def __init__(self, config: FocusAnalyzerConfig):
        self.config = config
        self.fs_manager = FileSystemManager()

    def normalized_variance(self, img):
        """
        Normalized variance focus measure.
        Robust to illumination changes.

        Args:
            img (numpy.ndarray): Input grayscale image

        Returns:
            float: Focus quality score
        """
        # FileSystemManager.load_image already ensures grayscale
        # No need for additional conversion

        mean_val = np.mean(img)
        if mean_val == 0:  # Avoid division by zero
            return 0

        return np.var(img) / mean_val

    def laplacian_energy(self, img, ksize=3):
        """
        Laplacian energy focus measure.
        Sensitive to edges and high-frequency content.

        Args:
            img (numpy.ndarray): Input grayscale image
            ksize (int): Kernel size for Laplacian

        Returns:
            float: Focus quality score
        """
        # FileSystemManager.load_image already ensures grayscale
        # No need for additional conversion

        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
        return np.mean(np.square(lap))

    def tenengrad_variance(self, img, ksize=3, threshold=0):
        """
        Tenengrad variance focus measure.
        Based on gradient magnitude.

        Args:
            img (numpy.ndarray): Input grayscale image
            ksize (int): Kernel size for Sobel operator
            threshold (float): Threshold for gradient magnitude

        Returns:
            float: Focus quality score
        """
        # FileSystemManager.load_image already ensures grayscale
        # No need for additional conversion

        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        fm = gx**2 + gy**2
        fm[fm < threshold] = 0  # Thresholding to reduce noise impact

        return np.mean(fm)

    def adaptive_fft_focus(self, img):
        """
        Adaptive FFT focus measure optimized for low-contrast microscopy images.
        Uses image statistics to set threshold adaptively.

        Args:
            img (numpy.ndarray): Input grayscale image

        Returns:
            float: Focus quality score
        """
        # FileSystemManager.load_image already ensures grayscale
        # No need for additional conversion

        # Apply FFT
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Calculate image statistics for adaptive thresholding
        img_mean = np.mean(img)
        img_std = np.std(img)

        # Adaptive threshold based on image statistics
        threshold_factor = max(0.1, min(1.0, img_std / 50.0))
        threshold = np.max(magnitude) * threshold_factor

        # Count frequency components above threshold
        high_freq_count = np.sum(magnitude > threshold)

        # Normalize by image size
        score = high_freq_count / (img.shape[0] * img.shape[1])

        return score

    def combined_focus_measure(self, img, weights=None):
        """
        Combined focus measure using multiple metrics.
        Optimized for microscopy images, especially low-contrast specimens.

        Args:
            img (numpy.ndarray): Input grayscale image
            weights (dict): Optional dictionary with weights for each metric

        Returns:
            float: Combined focus quality score
        """
        # Default weights if none provided
        if weights is None:
            weights = {
                'nvar': 0.3,
                'lap': 0.3,
                'ten': 0.2,
                'fft': 0.2
            }

        # FileSystemManager.load_image already ensures grayscale
        # No need for additional conversion

        # Calculate individual metrics
        nvar = self.normalized_variance(img)
        lap = self.laplacian_energy(img)
        ten = self.tenengrad_variance(img)
        fft = self.adaptive_fft_focus(img)

        # Weighted combination
        score = (
            weights['nvar'] * nvar +
            weights['lap'] * lap +
            weights['ten'] * ten +
            weights['fft'] * fft
        )

        return score

    def _get_focus_function(self, method):
        """
        Get the appropriate focus measure function based on method name.

        This helper method centralizes the logic for selecting the focus measure function,
        avoiding code duplication in find_best_focus and compute_focus_metrics methods.

        Args:
            method (str): Focus detection method name

        Returns:
            callable: The focus measure function

        Raises:
            ValueError: If the method is unknown
        """
        if method == 'combined':
            return self.combined_focus_measure
        elif method == 'nvar' or method == 'normalized_variance':
            return self.normalized_variance
        elif method == 'lap' or method == 'laplacian':
            return self.laplacian_energy
        elif method == 'ten' or method == 'tenengrad':
            return self.tenengrad_variance
        elif method == 'fft':
            return self.adaptive_fft_focus
        else:
            raise ValueError(f"Unknown focus method: {method}")

    def find_best_focus(self, image_stack, method='combined', roi=None):
        """
        Find the best focused image in a stack using specified method.

        Args:
            image_stack (list): List of images
            method (str): Focus detection method
            roi (tuple): Optional region of interest as (x, y, width, height)

        Returns:
            tuple: (best_focus_index, focus_scores)
        """
        focus_scores = []

        # Get the appropriate focus measure function
        focus_func = self._get_focus_function(method)

        # Process each image in stack
        for i, img in enumerate(image_stack):
            # Extract ROI if specified
            if roi is not None:
                x, y, w, h = roi
                img_roi = img[y:y+h, x:x+w]
            else:
                img_roi = img

            # Calculate focus score
            score = focus_func(img_roi)
            focus_scores.append((i, score))

        # Find index with maximum focus score
        best_focus_idx = max(focus_scores, key=lambda x: x[1])[0]

        return best_focus_idx, focus_scores

    def select_best_focus(self, image_stack, method='combined', roi=None):
        """
        Select the best focus plane from a stack of images.

        Args:
            image_stack (list): List of images
            method (str): Focus detection method
            roi (tuple): Optional region of interest as (x, y, width, height)

        Returns:
            tuple: (best_focus_image, best_focus_index, focus_scores)
        """
        best_idx, scores = self.find_best_focus(image_stack, method, roi)
        return image_stack[best_idx], best_idx, scores

    def compute_focus_metrics(self, image_stack, method='combined', roi=None):
        """
        Compute focus metrics for a stack of images.

        Args:
            image_stack (list): List of images
            method (str): Focus detection method
            roi (tuple): Optional region of interest as (x, y, width, height)

        Returns:
            list: List of focus scores for each image
        """
        focus_scores = []

        # Get the appropriate focus measure function
        focus_func = self._get_focus_function(method)

        # Process each image in stack
        for img in image_stack:
            # Extract ROI if specified
            if roi is not None:
                x, y, w, h = roi
                img_roi = img[y:y+h, x:x+w]
            else:
                img_roi = img

            # Calculate focus score
            score = focus_func(img_roi)
            focus_scores.append(score)

        return focus_scores
