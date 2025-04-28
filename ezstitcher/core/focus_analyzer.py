from typing import Dict, List, Tuple, Union, Optional
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class FocusAnalyzer:
    """
    Provides focus metrics and best focus selection.

    This class implements various focus measure algorithms and methods to find
    the best focused image in a Z-stack. All methods are static and do not require
    an instance.
    """

    # Default weights for combined focus measure
    DEFAULT_WEIGHTS = {
        'nvar': 0.3,  # Normalized variance
        'lap': 0.3,   # Laplacian energy
        'ten': 0.2,   # Tenengrad variance
        'fft': 0.2    # FFT-based focus
    }

    @staticmethod
    def normalized_variance(img: np.ndarray) -> float:
        """
        Normalized variance focus measure.
        Robust to illumination changes.

        Args:
            img: Input grayscale image

        Returns:
            Focus quality score
        """
        mean_val = np.mean(img)
        if mean_val == 0:  # Avoid division by zero
            return 0

        return np.var(img) / mean_val

    @staticmethod
    def laplacian_energy(img: np.ndarray, ksize: int = 3) -> float:
        """
        Laplacian energy focus measure.
        Sensitive to edges and high-frequency content.

        Args:
            img: Input grayscale image
            ksize: Kernel size for Laplacian

        Returns:
            Focus quality score
        """
        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
        return np.mean(np.square(lap))

    @staticmethod
    def tenengrad_variance(img: np.ndarray, ksize: int = 3, threshold: float = 0) -> float:
        """
        Tenengrad variance focus measure.
        Based on gradient magnitude.

        Args:
            img: Input grayscale image
            ksize: Kernel size for Sobel operator
            threshold: Threshold for gradient magnitude

        Returns:
            Focus quality score
        """
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        fm = gx**2 + gy**2
        fm[fm < threshold] = 0  # Thresholding to reduce noise impact

        return np.mean(fm)

    @staticmethod
    def adaptive_fft_focus(img: np.ndarray) -> float:
        """
        Adaptive FFT focus measure optimized for low-contrast microscopy images.
        Uses image statistics to set threshold adaptively.

        Args:
            img: Input grayscale image

        Returns:
            Focus quality score
        """
        # Apply FFT
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Calculate image statistics for adaptive thresholding
        # Only img_std is used for thresholding
        img_std = np.std(img)

        # Adaptive threshold based on image statistics
        threshold_factor = max(0.1, min(1.0, img_std / 50.0))
        threshold = np.max(magnitude) * threshold_factor

        # Count frequency components above threshold
        high_freq_count = np.sum(magnitude > threshold)

        # Normalize by image size
        score = high_freq_count / (img.shape[0] * img.shape[1])

        return score

    @staticmethod
    def combined_focus_measure(
        img: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Combined focus measure using multiple metrics.
        Optimized for microscopy images, especially low-contrast specimens.

        Args:
            img: Input grayscale image
            weights: Weights for each metric. If None, uses default weights.

        Returns:
            Combined focus quality score
        """
        # Use provided weights or defaults
        if weights is None:
            weights = FocusAnalyzer.DEFAULT_WEIGHTS

        # Calculate individual metrics
        nvar = FocusAnalyzer.normalized_variance(img)
        lap = FocusAnalyzer.laplacian_energy(img)
        ten = FocusAnalyzer.tenengrad_variance(img)
        fft = FocusAnalyzer.adaptive_fft_focus(img)

        # Weighted combination
        score = (
            weights.get('nvar', 0.3) * nvar +
            weights.get('lap', 0.3) * lap +
            weights.get('ten', 0.2) * ten +
            weights.get('fft', 0.2) * fft
        )

        return score

    @staticmethod
    def _get_focus_function(metric: Union[str, Dict[str, float]]):
        """
        Get the appropriate focus measure function based on metric.

        Args:
            metric: Focus detection method name or weights dictionary
                   If string: "combined", "normalized_variance", "laplacian", "tenengrad", "fft"
                   If dict: Weights for combined focus measure

        Returns:
            callable: The focus measure function and any additional arguments

        Raises:
            ValueError: If the method is unknown
        """
        # If metric is a dictionary, use it as weights for combined focus measure
        if isinstance(metric, dict):
            return lambda img: FocusAnalyzer.combined_focus_measure(img, metric)

        # Otherwise, treat it as a string method name
        if metric == 'combined':
            return FocusAnalyzer.combined_focus_measure
        if metric in ('nvar', 'normalized_variance'):
            return FocusAnalyzer.normalized_variance
        if metric in ('lap', 'laplacian'):
            return FocusAnalyzer.laplacian_energy
        if metric in ('ten', 'tenengrad'):
            return FocusAnalyzer.tenengrad_variance
        if metric == 'fft':
            return FocusAnalyzer.adaptive_fft_focus

        # If we get here, the metric is unknown
        raise ValueError(f"Unknown focus method: {metric}")

    @staticmethod
    def find_best_focus(
        image_stack: List[np.ndarray],
        metric: Union[str, Dict[str, float]] = "combined"
    ) -> Tuple[int, List[Tuple[int, float]]]:
        """
        Find the best focused image in a stack using specified method.

        Args:
            image_stack: List of images
            metric: Focus detection method or weights dictionary
                   If string: "combined", "normalized_variance", "laplacian", "tenengrad", "fft"
                   If dict: Weights for combined focus measure

        Returns:
            Tuple of (best_focus_index, focus_scores)
        """
        focus_scores = []

        # Get the appropriate focus measure function
        focus_func = FocusAnalyzer._get_focus_function(metric)

        # Process each image in stack
        for i, img in enumerate(image_stack):
            # Calculate focus score
            score = focus_func(img)
            focus_scores.append((i, score))

        # Find index with maximum focus score
        best_focus_idx = max(focus_scores, key=lambda x: x[1])[0]

        return best_focus_idx, focus_scores

    @staticmethod
    def select_best_focus(
        image_stack: List[np.ndarray],
        metric: Union[str, Dict[str, float]] = "combined"
    ) -> Tuple[np.ndarray, int, List[Tuple[int, float]]]:
        """
        Select the best focus plane from a stack of images.

        Args:
            image_stack: List of images
            metric: Focus detection method or weights dictionary
                   If string: "combined", "normalized_variance", "laplacian", "tenengrad", "fft"
                   If dict: Weights for combined focus measure

        Returns:
            Tuple of (best_focus_image, best_focus_index, focus_scores)
        """
        best_idx, scores = FocusAnalyzer.find_best_focus(image_stack, metric)
        return image_stack[best_idx], best_idx, scores

    @staticmethod
    def compute_focus_metrics(image_stack: List[np.ndarray],
                             metric: Union[str, Dict[str, float]] = "combined") -> List[float]:
        """
        Compute focus metrics for a stack of images.

        Args:
            image_stack: List of images
            metric: Focus detection method or weights dictionary
                   If string: "combined", "normalized_variance", "laplacian", "tenengrad", "fft"
                   If dict: Weights for combined focus measure

        Returns:
            List of focus scores for each image
        """
        focus_scores = []

        # Get the appropriate focus measure function
        focus_func = FocusAnalyzer._get_focus_function(metric)

        # Process each image in stack
        for img in image_stack:
            # Calculate focus score
            score = focus_func(img)
            focus_scores.append(score)

        return focus_scores
