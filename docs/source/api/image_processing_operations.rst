.. _image-processing-operations:

========================
Image Processing Operations
========================

.. module:: ezstitcher.core.image_processor

This module provides a comprehensive set of image processing operations for microscopy images. These operations can be used in pipeline steps to process images in various ways.

.. note::
   This guide focuses on practical usage of image processing operations, with examples and common use cases.
   For the complete API reference of the ImageProcessor class, including all method signatures and parameters,
   see :doc:`image_processor`.

   The documentation here is organized by operation type and includes examples of how to use these operations
   in pipeline steps, while :doc:`image_processor` provides the formal API documentation.

Basic Operations
--------------

.. _operation-normalize:

Normalization
^^^^^^^^^^^

.. py:function:: stack_percentile_normalize(images, low_percentile=1.0, high_percentile=99.0)

   Normalize a stack of images using percentile-based normalization.

   :param images: List of input images
   :type images: list of numpy.ndarray
   :param low_percentile: Lower percentile for normalization (default: 1.0)
   :type low_percentile: float
   :param high_percentile: Upper percentile for normalization (default: 99.0)
   :type high_percentile: float
   :return: List of normalized images
   :rtype: list of numpy.ndarray

.. py:function:: normalize(image, low=None, high=None)

   Normalize a single image to the range [0, 1].

   :param image: Input image
   :type image: numpy.ndarray
   :param low: Lower bound for normalization (default: min value in image)
   :type low: float or None
   :param high: Upper bound for normalization (default: max value in image)
   :type high: float or None
   :return: Normalized image
   :rtype: numpy.ndarray

.. _operation-filtering:

Filtering
^^^^^^^

.. py:function:: gaussian_blur(image, sigma=1.0)

   Apply Gaussian blur to an image.

   :param image: Input image
   :type image: numpy.ndarray
   :param sigma: Standard deviation for Gaussian kernel
   :type sigma: float
   :return: Blurred image
   :rtype: numpy.ndarray

.. py:function:: median_filter(image, size=3)

   Apply median filter to an image.

   :param image: Input image
   :type image: numpy.ndarray
   :param size: Size of the median filter window
   :type size: int
   :return: Filtered image
   :rtype: numpy.ndarray

.. py:function:: tophat(image, size=15)

   Apply white tophat filter to an image to remove background.

   :param image: Input image
   :type image: numpy.ndarray
   :param size: Size of the structuring element
   :type size: int
   :return: Filtered image
   :rtype: numpy.ndarray

.. _operation-enhancement:

Enhancement
^^^^^^^^^

.. py:function:: sharpen(image, sigma=1.0, amount=1.5)

   Sharpen an image using unsharp masking.

   :param image: Input image
   :type image: numpy.ndarray
   :param sigma: Standard deviation for Gaussian kernel
   :type sigma: float
   :param amount: Sharpening amount
   :type amount: float
   :return: Sharpened image
   :rtype: numpy.ndarray

.. py:function:: contrast_stretch(image, low_percentile=1.0, high_percentile=99.0)

   Stretch the contrast of an image using percentile-based normalization.

   :param image: Input image
   :type image: numpy.ndarray
   :param low_percentile: Lower percentile for contrast stretching
   :type low_percentile: float
   :param high_percentile: Upper percentile for contrast stretching
   :type high_percentile: float
   :return: Contrast-stretched image
   :rtype: numpy.ndarray

Z-Stack Operations
----------------

.. _operation-z-projection:

Z-Stack Projection
^^^^^^^^^^^^^^^

.. py:function:: create_projection(images, method='max_projection', focus_analyzer=None)

   Create a projection from a Z-stack of images.

   :param images: List of Z-stack images
   :type images: list of numpy.ndarray
   :param method: Projection method ('max_projection', 'mean_projection', or 'best_focus')
   :type method: str
   :param focus_analyzer: Focus analyzer for 'best_focus' method
   :type focus_analyzer: FocusAnalyzer or None
   :return: Projected image
   :rtype: numpy.ndarray

.. py:function:: max_projection(images)

   Create a maximum intensity projection from a Z-stack of images.

   :param images: List of Z-stack images
   :type images: list of numpy.ndarray
   :return: Maximum intensity projection
   :rtype: numpy.ndarray

.. py:function:: mean_projection(images)

   Create a mean intensity projection from a Z-stack of images.

   :param images: List of Z-stack images
   :type images: list of numpy.ndarray
   :return: Mean intensity projection
   :rtype: numpy.ndarray

.. py:function:: best_focus_projection(images, focus_analyzer)

   Create a projection by selecting the best focused slice for each pixel.

   :param images: List of Z-stack images
   :type images: list of numpy.ndarray
   :param focus_analyzer: Focus analyzer for determining focus quality
   :type focus_analyzer: FocusAnalyzer
   :return: Best focus projection
   :rtype: numpy.ndarray

Multi-Channel Operations
---------------------

.. _operation-composite:

Channel Compositing
^^^^^^^^^^^^^^^

.. py:function:: create_composite(images, weights=None)

   Create a composite image from multiple channel images.

   :param images: List of channel images
   :type images: list of numpy.ndarray
   :param weights: List of weights for each channel (default: equal weights)
   :type weights: list of float or None
   :return: Composite image
   :rtype: numpy.ndarray

Using Operations in Pipelines
--------------------------

These operations can be used in pipeline steps in various ways:

.. code-block:: python

    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_processor import ImageProcessor as IP
    from ezstitcher.core.utils import stack

    # Single operation
    Step(
        func=IP.stack_percentile_normalize,
        input_dir=orchestrator.workspace_path
    )

    # Operation with arguments
    Step(
        func=(IP.stack_percentile_normalize, {
            'low_percentile': 1.0,
            'high_percentile': 99.0
        }),
        input_dir=orchestrator.workspace_path
    )

    # Multiple operations in sequence
    Step(
        func=[
            (stack(IP.tophat), {'size': 15}),
            (stack(IP.sharpen), {'sigma': 1.0, 'amount': 1.5}),
            IP.stack_percentile_normalize
        ],
        input_dir=orchestrator.workspace_path
    )

    # Channel-specific operations
    Step(
        func={
            "1": (stack(IP.tophat), {'size': 15}),
            "2": (stack(IP.sharpen), {'sigma': 1.0, 'amount': 1.5})
        },
        group_by='channel',
        input_dir=orchestrator.workspace_path
    )

For more information on function handling patterns, see :ref:`function-handling`.

Common Use Cases
-------------

Here are some common use cases for these operations:

1. **Basic Image Enhancement**:

   .. code-block:: python

       # Enhance image contrast
       Step(
           func=IP.stack_percentile_normalize,
           input_dir=orchestrator.workspace_path
       )

2. **Background Removal**:

   .. code-block:: python

       # Remove background using tophat filter
       Step(
           func=(stack(IP.tophat), {'size': 15}),
           input_dir=orchestrator.workspace_path
       )

3. **Z-Stack Flattening**:

   .. code-block:: python

       # Flatten Z-stack using maximum intensity projection
       Step(
           func=(IP.create_projection, {'method': 'max_projection'}),
           variable_components=['z_index'],
           input_dir=orchestrator.workspace_path
       )

4. **Multi-Channel Composite**:

   .. code-block:: python

       # Create composite image from multiple channels
       Step(
           func=(IP.create_composite, {'weights': [0.7, 0.3]}),
           variable_components=['channel'],
           input_dir=orchestrator.workspace_path
       )

5. **Complete Image Processing Workflow**:

   .. code-block:: python

       # Complete workflow: background removal, sharpening, normalization
       Step(
           func=[
               (stack(IP.tophat), {'size': 15}),
               (stack(IP.sharpen), {'sigma': 1.0, 'amount': 1.5}),
               IP.stack_percentile_normalize
           ],
           input_dir=orchestrator.workspace_path
       )

For more examples and best practices, see :ref:`best-practices-function-handling` in the :doc:`../user_guide/best_practices` guide.
