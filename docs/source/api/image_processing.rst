.. _image-processing-operations:

========================
Image Processing Operations
========================

.. module:: ezstitcher.core.image_processor

This module provides a comprehensive set of image processing operations for microscopy images through the ImageProcessor class. All methods are static and do not require an instance of the class.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
-------

The ImageProcessor class provides operations for:

* Image normalization and contrast enhancement
* Filtering and background removal
* Image sharpening and enhancement
* Z-stack processing (max/mean projections, best focus)
* Multi-channel operations (compositing)

API Reference
-----------

.. py:class:: ImageProcessor

   Handles image normalization, filtering, and compositing.
   All methods are static and do not require an instance.

   .. py:staticmethod:: preprocess(image, channel, preprocessing_funcs=None)

      Apply preprocessing to a single image for a given channel.

      :param image: Input image
      :type image: numpy.ndarray
      :param channel: Channel identifier
      :type channel: str
      :param preprocessing_funcs: Dictionary mapping channels to preprocessing functions
      :type preprocessing_funcs: dict, optional
      :return: Processed image
      :rtype: numpy.ndarray

   .. py:staticmethod:: blur(image, sigma=1)

      Apply Gaussian blur to an image.

      :param image: Input image
      :type image: numpy.ndarray
      :param sigma: Standard deviation for Gaussian kernel
      :type sigma: float
      :return: Blurred image
      :rtype: numpy.ndarray

   .. py:staticmethod:: sharpen(image, radius=1, amount=1.0)

      Sharpen an image using unsharp masking.

      :param image: Input image
      :type image: numpy.ndarray
      :param radius: Radius of Gaussian blur
      :type radius: float
      :param amount: Sharpening strength
      :type amount: float
      :return: Sharpened image
      :rtype: numpy.ndarray

   .. py:staticmethod:: normalize(image, target_min=0, target_max=65535)

      Normalize image to specified range.

      :param image: Input image
      :type image: numpy.ndarray
      :param target_min: Target minimum value
      :type target_min: int
      :param target_max: Target maximum value
      :type target_max: int
      :return: Normalized image
      :rtype: numpy.ndarray

   .. py:staticmethod:: percentile_normalize(image, low_percentile=1, high_percentile=99, target_min=0, target_max=65535)

      Normalize image using percentile-based contrast stretching.

      :param image: Input image
      :type image: numpy.ndarray
      :param low_percentile: Lower percentile (0-100)
      :type low_percentile: float
      :param high_percentile: Upper percentile (0-100)
      :type high_percentile: float
      :param target_min: Target minimum value
      :type target_min: int
      :param target_max: Target maximum value
      :type target_max: int
      :return: Normalized image
      :rtype: numpy.ndarray

   .. py:staticmethod:: stack_percentile_normalize(stack, low_percentile=1, high_percentile=99, target_min=0, target_max=65535)

      Normalize a stack of images using global percentile-based contrast stretching.
      This ensures consistent normalization across all images in the stack.

      :param stack: Stack of images
      :type stack: list or numpy.ndarray
      :param low_percentile: Lower percentile (0-100)
      :type low_percentile: float
      :param high_percentile: Upper percentile (0-100)
      :type high_percentile: float
      :param target_min: Target minimum value
      :type target_min: int
      :param target_max: Target maximum value
      :type target_max: int
      :return: Normalized stack of images
      :rtype: numpy.ndarray

   .. py:staticmethod:: create_composite(images, weights=None)

      Create a grayscale composite image from multiple channels.

      :param images: Dictionary mapping channel names to images or list of images
      :type images: dict or list
      :param weights: Optional dictionary with weights for each channel or list of weights
      :type weights: dict or list, optional
      :return: Grayscale composite image (16-bit)
      :rtype: numpy.ndarray

   .. py:staticmethod:: apply_mask(image, mask)

      Apply a mask to an image.

      :param image: Input image
      :type image: numpy.ndarray
      :param mask: Mask image (same shape as input)
      :type mask: numpy.ndarray
      :return: Masked image
      :rtype: numpy.ndarray

   .. py:staticmethod:: create_weight_mask(shape, margin_ratio=0.1)

      Create a weight mask for blending images.

      :param shape: Shape of the mask (height, width)
      :type shape: tuple
      :param margin_ratio: Ratio of image size to use as margin
      :type margin_ratio: float
      :return: Weight mask
      :rtype: numpy.ndarray

   .. py:staticmethod:: max_projection(stack)

      Create a maximum intensity projection from a Z-stack.

      :param stack: Stack of images
      :type stack: list or numpy.ndarray
      :return: Maximum intensity projection
      :rtype: numpy.ndarray

   .. py:staticmethod:: mean_projection(stack)

      Create a mean intensity projection from a Z-stack.

      :param stack: Stack of images
      :type stack: list or numpy.ndarray
      :return: Mean intensity projection
      :rtype: numpy.ndarray

   .. py:staticmethod:: stack_equalize_histogram(stack, bins=65536, range_min=0, range_max=65535)

      Apply true histogram equalization to an entire stack of images.
      This ensures consistent contrast enhancement across all images in the stack.

      :param stack: Stack of images
      :type stack: list or numpy.ndarray
      :param bins: Number of bins for histogram computation
      :type bins: int
      :param range_min: Minimum value for histogram range
      :type range_min: int
      :param range_max: Maximum value for histogram range
      :type range_max: int
      :return: Histogram-equalized stack of images
      :rtype: numpy.ndarray

   .. py:staticmethod:: create_projection(stack, method="max_projection", focus_analyzer=None)

      Create a projection from a stack using the specified method.

      :param stack: List of images
      :type stack: list
      :param method: Projection method (max_projection, mean_projection, best_focus)
      :type method: str
      :param focus_analyzer: Focus analyzer for best_focus method
      :type focus_analyzer: FocusAnalyzer, optional
      :return: Projected image
      :rtype: numpy.ndarray

   .. py:staticmethod:: tophat(image, selem_radius=50, downsample_factor=4)

      Apply white top-hat transform to an image.

      :param image: Input image
      :type image: numpy.ndarray
      :param selem_radius: Radius of structuring element
      :type selem_radius: int
      :param downsample_factor: Factor to downsample image for faster processing
      :type downsample_factor: int
      :return: Top-hat transformed image
      :rtype: numpy.ndarray

Operations by Category
-------------------

.. _operation-normalize:

Normalization
^^^^^^^^^^^

Normalization operations adjust the intensity range of images to improve contrast and consistency.

**stack_percentile_normalize**

The most commonly used normalization function is ``stack_percentile_normalize``, which normalizes a stack of images using percentile-based contrast stretching:

.. code-block:: python

    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Basic normalization
    step = Step(
        func=IP.stack_percentile_normalize,
        input_dir=orchestrator.workspace_path
    )

    # Normalization with custom percentiles
    step = Step(
        func=(IP.stack_percentile_normalize, {
            'low_percentile': 1.0,  # Bottom 1% becomes black
            'high_percentile': 99.0  # Top 1% becomes white
        }),
        input_dir=orchestrator.workspace_path
    )

.. _operation-filtering:

Filtering
^^^^^^^

Filtering operations remove noise or background from images.

**tophat**

The ``tophat`` filter is particularly useful for removing background in fluorescence microscopy:

.. code-block:: python

    from ezstitcher.core.utils import stack

    # Apply tophat filter to remove background
    step = Step(
        func=(stack(IP.tophat), {'selem_radius': 50}),
        input_dir=orchestrator.workspace_path
    )

**blur**

Gaussian blur can be used to reduce noise:

.. code-block:: python

    # Apply Gaussian blur
    step = Step(
        func=(stack(IP.blur), {'sigma': 1.5}),
        input_dir=orchestrator.workspace_path
    )

.. _operation-enhancement:

Enhancement
^^^^^^^^^

Enhancement operations improve image details and contrast.

**sharpen**

Sharpening enhances edges and fine details:

.. code-block:: python

    # Sharpen images
    step = Step(
        func=(stack(IP.sharpen), {
            'radius': 1.0,  # Gaussian blur radius
            'amount': 1.5   # Sharpening amount
        }),
        input_dir=orchestrator.workspace_path
    )

.. _operation-z-projection:

Z-Stack Projection
^^^^^^^^^^^^^^^

Z-stack projection operations combine multiple Z-slices into a single 2D image.

**create_projection**

The ``create_projection`` function supports multiple projection methods:

.. code-block:: python

    # Maximum intensity projection
    step = Step(
        func=(IP.create_projection, {'method': 'max_projection'}),
        variable_components=['z_index'],
        input_dir=orchestrator.workspace_path
    )

    # Mean intensity projection
    step = Step(
        func=(IP.create_projection, {'method': 'mean_projection'}),
        variable_components=['z_index'],
        input_dir=orchestrator.workspace_path
    )

    # Best focus projection (requires a focus analyzer)
    from ezstitcher.core.focus_analyzer import FocusAnalyzer

    focus_analyzer = FocusAnalyzer(metric='variance_of_laplacian')
    step = Step(
        func=(IP.create_projection, {
            'method': 'best_focus',
            'focus_analyzer': focus_analyzer
        }),
        variable_components=['z_index'],
        input_dir=orchestrator.workspace_path
    )

.. important::
   When using Z-stack projection operations, always set ``variable_components=['z_index']`` to ensure
   that images with the same site and channel but different z-indices are grouped together.

.. _operation-composite:

Channel Compositing
^^^^^^^^^^^^^^^

Channel compositing operations combine multiple channels into a single image.

**create_composite**

The ``create_composite`` function combines multiple channel images:

.. code-block:: python

    # Create composite with equal weights
    step = Step(
        func=IP.create_composite,
        variable_components=['channel'],
        input_dir=orchestrator.workspace_path
    )

    # Create composite with custom weights (70% channel 1, 30% channel 2)
    step = Step(
        func=(IP.create_composite, {'weights': [0.7, 0.3]}),
        variable_components=['channel'],
        input_dir=orchestrator.workspace_path
    )

.. important::
   When using channel compositing operations, always set ``variable_components=['channel']`` to ensure
   that images with the same site and z-index but different channels are grouped together.

Using Operations in Pipelines
--------------------------

These operations can be used in pipeline steps in various ways:

**Single Operation**

.. code-block:: python

    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_processor import ImageProcessor as IP
    from ezstitcher.core.utils import stack

    # Single operation
    Step(
        func=IP.stack_percentile_normalize,
        input_dir=orchestrator.workspace_path
    )

**Operation with Arguments**

.. code-block:: python

    # Operation with arguments
    Step(
        func=(IP.stack_percentile_normalize, {
            'low_percentile': 1.0,
            'high_percentile': 99.0
        }),
        input_dir=orchestrator.workspace_path
    )

**Multiple Operations in Sequence**

.. code-block:: python

    # Multiple operations in sequence
    Step(
        func=[
            (stack(IP.tophat), {'selem_radius': 50}),
            (stack(IP.sharpen), {'radius': 1.0, 'amount': 1.5}),
            IP.stack_percentile_normalize
        ],
        input_dir=orchestrator.workspace_path
    )

**Channel-Specific Operations**

.. code-block:: python

    # Channel-specific operations
    Step(
        func={
            "1": (stack(IP.tophat), {'selem_radius': 50}),
            "2": (stack(IP.sharpen), {'radius': 1.0, 'amount': 1.5})
        },
        group_by='channel',
        input_dir=orchestrator.workspace_path
    )

For more information on function handling patterns, see :ref:`function-handling`.

Common Use Cases
-------------

Here are some common use cases for these operations:

**Basic Image Enhancement**

.. code-block:: python

    # Enhance image contrast
    Step(
        func=IP.stack_percentile_normalize,
        input_dir=orchestrator.workspace_path
    )

**Background Removal**

.. code-block:: python

    # Remove background using tophat filter
    Step(
        func=(stack(IP.tophat), {'selem_radius': 50}),
        input_dir=orchestrator.workspace_path
    )

**Z-Stack Flattening**

.. code-block:: python

    # Flatten Z-stack using maximum intensity projection
    Step(
        func=(IP.create_projection, {'method': 'max_projection'}),
        variable_components=['z_index'],
        input_dir=orchestrator.workspace_path
    )

**Multi-Channel Composite**

.. code-block:: python

    # Create composite image from multiple channels
    Step(
        func=(IP.create_composite, {'weights': [0.7, 0.3]}),
        variable_components=['channel'],
        input_dir=orchestrator.workspace_path
    )

**Complete Image Processing Workflow**

.. code-block:: python

    # Complete workflow: background removal, sharpening, normalization
    Step(
        func=[
            (stack(IP.tophat), {'selem_radius': 50}),
            (stack(IP.sharpen), {'radius': 1.0, 'amount': 1.5}),
            IP.stack_percentile_normalize
        ],
        input_dir=orchestrator.workspace_path
    )

Best Practices
-----------

When using image processing operations, follow these best practices:

1. **Use stack() for Single-Image Functions**:
   - Use the ``stack()`` utility to apply single-image functions to stacks
   - Example: ``(stack(IP.tophat), {'selem_radius': 50})``

2. **Set variable_components Appropriately**:
   - For Z-stack operations: ``variable_components=['z_index']``
   - For channel operations: ``variable_components=['channel']``
   - For most other operations: Default ``['site']`` is appropriate

3. **Use Tuples for Function Arguments**:
   - Always use the tuple pattern ``(func, kwargs)`` for passing arguments
   - Example: ``(IP.stack_percentile_normalize, {'low_percentile': 1.0, 'high_percentile': 99.0})``

4. **Process Images in the Right Order**:
   - Background removal → Enhancement → Normalization
   - Z-stack flattening → Channel-specific processing → Compositing

5. **Balance Performance and Quality**:
   - For large images, consider using smaller filter sizes or downsampling
   - For tophat filtering, adjust ``selem_radius`` and ``downsample_factor`` based on image size

For more examples and best practices, see :ref:`best-practices-function-handling` in the :doc:`../user_guide/best_practices` guide.
