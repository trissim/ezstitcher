Focus Analyzer
==============

.. module:: ezstitcher.core.focus_analyzer

This module contains the FocusAnalyzer class for analyzing focus quality in microscopy images.

FocusAnalyzer
------------

.. py:class:: FocusAnalyzer(config)

   Provides focus metrics and best focus selection.

   This class implements various focus measure algorithms and methods to find
   the best focused image in a Z-stack. It uses the FileSystemManager for
   image handling to avoid code duplication.

   :param config: Configuration for focus analysis
   :type config: :class:`~ezstitcher.core.config.FocusAnalyzerConfig`

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
      :param weights: Optional dictionary with weights for each metric
      :type weights: dict, optional
      :return: Combined focus quality score
      :rtype: float

   .. py:method:: find_best_focus(image_stack, method='combined', roi=None)

      Find the best focused image in a stack using specified method.

      :param image_stack: List of images
      :type image_stack: list
      :param method: Focus detection method
      :type method: str
      :param roi: Optional region of interest as (x, y, width, height)
      :type roi: tuple, optional
      :return: Tuple of (best_focus_index, focus_scores)
      :rtype: tuple

   .. py:method:: select_best_focus(image_stack, method='combined', roi=None)

      Select the best focus plane from a stack of images.

      :param image_stack: List of images
      :type image_stack: list
      :param method: Focus detection method
      :type method: str
      :param roi: Optional region of interest as (x, y, width, height)
      :type roi: tuple, optional
      :return: Tuple of (best_focus_image, best_focus_index, focus_scores)
      :rtype: tuple

   .. py:method:: compute_focus_metrics(image_stack, method='combined', roi=None)

      Compute focus metrics for a stack of images.

      :param image_stack: List of images
      :type image_stack: list
      :param method: Focus detection method
      :type method: str
      :param roi: Optional region of interest as (x, y, width, height)
      :type roi: tuple, optional
      :return: List of focus scores for each image
      :rtype: list

FocusAnalyzerConfig
-----------------

.. py:class:: FocusAnalyzerConfig

   Configuration for the FocusAnalyzer class.

   .. py:attribute:: method
      :type: str
      :value: "combined"

      Focus detection method. Options: "combined", "normalized_variance", "laplacian", "tenengrad", "fft".

   .. py:attribute:: roi
      :type: tuple or None
      :value: None

      Optional region of interest as (x, y, width, height).

   .. py:attribute:: weights
      :type: dict or None
      :value: None

      Optional dictionary with weights for each metric in combined focus measure.
