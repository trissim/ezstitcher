Focus Analyzer
==============

.. module:: ezstitcher.core.focus_analyzer

This module contains the FocusAnalyzer class for analyzing focus quality in microscopy images.

FocusAnalyzer
------------

.. py:class:: FocusAnalyzer(metric="combined", roi=None, weights=None)

   Provides focus metrics and best focus selection.

   This class implements various focus measure algorithms and methods to find
   the best focused image in a Z-stack. It uses the FileSystemManager for
   image handling to avoid code duplication.

   :param metric: Focus detection method. Options: "combined", "normalized_variance", "laplacian", "tenengrad", "fft".
   :type metric: str
   :param roi: Optional region of interest as (x, y, width, height).
   :type roi: tuple, optional
   :param weights: Optional dictionary with weights for each metric in combined focus measure.
   :type weights: dict, optional

   .. py:method:: normalized_variance(img)

      Normalized variance focus measure.
      Robust to illumination changes.

      :param img: Input grayscale image
      :type img: numpy.ndarray
      :return: Focus quality score
      :rtype: float

   .. py:method:: laplacian_energy(img, ksize=3)

      Laplacian energy focus measure.
      Sensitive to edges and high-frequency content.

      :param img: Input grayscale image
      :type img: numpy.ndarray
      :param ksize: Kernel size for Laplacian
      :type ksize: int
      :return: Focus quality score
      :rtype: float

   .. py:method:: tenengrad_variance(img, ksize=3, threshold=0)

      Tenengrad variance focus measure.
      Based on gradient magnitude.

      :param img: Input grayscale image
      :type img: numpy.ndarray
      :param ksize: Kernel size for Sobel operator
      :type ksize: int
      :param threshold: Threshold for gradient magnitude
      :type threshold: float
      :return: Focus quality score
      :rtype: float

   .. py:method:: adaptive_fft_focus(img)

      Adaptive FFT focus measure optimized for low-contrast microscopy images.
      Uses image statistics to set threshold adaptively.

      :param img: Input grayscale image
      :type img: numpy.ndarray
      :return: Focus quality score
      :rtype: float

   .. py:method:: combined_focus_measure(img, weights=None)

      Combined focus measure using multiple metrics.
      Optimized for microscopy images, especially low-contrast specimens.

      :param img: Input grayscale image
      :type img: numpy.ndarray
      :param weights: Weights for each metric. If None, uses the weights specified in the constructor or defaults.
      :type weights: dict, optional
      :return: Combined focus quality score
      :rtype: float

   .. py:method:: find_best_focus(image_stack, method=None, roi=None)

      Find the best focused image in a stack using specified method.

      :param image_stack: List of images
      :type image_stack: list
      :param method: Focus detection method. If None, uses the method specified in the constructor.
      :type method: str, optional
      :param roi: Optional region of interest as (x, y, width, height). If None, uses the ROI specified in the constructor.
      :type roi: tuple, optional
      :return: Tuple of (best_focus_index, focus_scores)
      :rtype: tuple

   .. py:method:: select_best_focus(image_stack, method=None, roi=None)

      Select the best focus plane from a stack of images.

      :param image_stack: List of images
      :type image_stack: list
      :param method: Focus detection method. If None, uses the method specified in the constructor.
      :type method: str, optional
      :param roi: Optional region of interest as (x, y, width, height). If None, uses the ROI specified in the constructor.
      :type roi: tuple, optional
      :return: Tuple of (best_focus_image, best_focus_index, focus_scores)
      :rtype: tuple

   .. py:method:: compute_focus_metrics(image_stack, method=None, roi=None)

      Compute focus metrics for a stack of images.

      :param image_stack: List of images
      :type image_stack: list
      :param method: Focus detection method. If None, uses the method specified in the constructor.
      :type method: str, optional
      :param roi: Optional region of interest as (x, y, width, height). If None, uses the ROI specified in the constructor.
      :type roi: tuple, optional
      :return: List of focus scores for each image
      :rtype: list


