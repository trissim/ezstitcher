Z-Stack Handling
==============

This page explains how EZStitcher handles Z-stacks (3D image stacks).

Z-Stack Concepts
--------------

Z-stacks are 3D image stacks captured at different focal planes. Each Z-plane represents a different depth in the sample.

EZStitcher provides several options for handling Z-stacks:

- **Projection**: Create a 2D representation of the 3D stack
- **Best Focus Selection**: Select the best focused plane
- **Per-Plane Stitching**: Stitch each Z-plane separately

Z-Stack Organization
-----------------

EZStitcher supports different Z-stack organizations depending on the microscope type. For detailed information about microscope-specific Z-stack formats, see the :doc:`../appendices/microscope_formats` appendix.

EZStitcher automatically detects Z-stacks and organizes them for processing.

Z-Stack Processing Options
----------------------

EZStitcher provides several configuration options for Z-stack processing:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Configure Z-stack processing
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",  # Method for reference channel
        stitch_flatten="best_focus"          # Method for final stitching
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

The key configuration parameters for Z-stack processing are:

- **reference_flatten**: Method used for position generation ("max_projection", "mean_projection", "best_focus", etc.)
- **stitch_flatten**: Method used for final stitching ("max_projection", "mean_projection", "best_focus", None for per-plane stitching)
- **focus_method**: Method used for focus detection ("combined", "nvar", "lap", "ten", "fft")
- **additional_projections**: List of additional projections to create (["max", "mean", "std"])

Projection Methods
---------------

EZStitcher provides several projection methods for Z-stacks:

Maximum Intensity Projection
~~~~~~~~~~~~~~~~~~~~~~~~~

Creates a 2D image where each pixel is the maximum value across all Z-planes:

.. code-block:: python

    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",  # Use max projection for reference
        stitch_flatten="max_projection"      # Use max projection for final stitching
    )

Mean Projection
~~~~~~~~~~~~

Creates a 2D image where each pixel is the average value across all Z-planes:

.. code-block:: python

    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="mean_projection",  # Use mean projection for reference
        stitch_flatten="mean_projection"      # Use mean projection for final stitching
    )

Standard Deviation Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates a 2D image where each pixel is the standard deviation across all Z-planes:

.. code-block:: python

    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="std_projection",  # Use std projection for reference
        stitch_flatten="std_projection"      # Use std projection for final stitching
    )

Best Focus Detection
-----------------

EZStitcher can detect the best focused plane in each Z-stack:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",  # Use max projection for reference
        stitch_flatten="best_focus",         # Use best focus for final images
        focus_config=FocusAnalyzerConfig(
            method="combined",               # Combined focus metrics
            roi=(100, 100, 200, 200)         # Optional ROI for focus detection
        )
    )

Focus Detection Methods
~~~~~~~~~~~~~~~~~~~~

EZStitcher provides several focus detection methods:

- **Normalized Variance (nvar)**: Based on image variance
- **Laplacian (lap)**: Based on edge detection
- **Tenengrad (ten)**: Based on gradient magnitude
- **Fourier (fft)**: Based on frequency domain analysis
- **Combined**: Weighted combination of multiple methods

You can customize the weights for the combined method:

.. code-block:: python

    focus_config=FocusAnalyzerConfig(
        method="combined",
        weights={
            "nvar": 0.4,
            "lap": 0.3,
            "ten": 0.2,
            "fft": 0.1
        }
    )

Per-Plane Stitching
----------------

EZStitcher can stitch each Z-plane separately:

.. code-block:: python

    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",  # Use max projection for reference
        stitch_flatten=None                  # Stitch each Z-plane separately
    )

This creates a separate stitched image for each Z-plane, preserving the 3D structure.

Advanced Z-Stack Processing
------------------------

Custom Z-Stack Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can define custom functions for processing Z-stacks:

.. code-block:: python

    import numpy as np
    from skimage import filters

    def preprocess_zstack(stack):
        """Apply custom preprocessing to each plane in a Z-stack."""
        processed_stack = []
        for plane in stack:
            # Apply histogram equalization
            from ezstitcher.core.image_preprocessor import ImagePreprocessor
            processed_plane = ImagePreprocessor.equalize_histogram(plane)
            # Apply denoising
            processed_plane = filters.gaussian(processed_plane, sigma=1)
            processed_stack.append(processed_plane)
        return processed_stack

    config = PipelineConfig(
        reference_channels=["1"],
        reference_processing=preprocess_zstack,  # Custom preprocessing function
        reference_flatten="max_projection",
        stitch_flatten="best_focus"
    )

Custom Projection Functions
~~~~~~~~~~~~~~~~~~~~~~~~

You can define custom projection functions:

.. code-block:: python

    import numpy as np

    def median_projection(stack):
        """Create a median projection from a Z-stack."""
        return np.median(stack, axis=0)

    def weighted_projection(stack, weights=None):
        """Create a weighted projection from a Z-stack."""
        if weights is None:
            # Default: emphasize middle planes
            num_planes = len(stack)
            weights = np.ones(num_planes)
            middle = num_planes // 2
            for i in range(num_planes):
                weights[i] = 1.0 - 0.5 * abs(i - middle) / middle

        # Apply weights
        weighted_stack = np.array([stack[i] * weights[i] for i in range(len(stack))])
        return np.sum(weighted_stack, axis=0) / np.sum(weights)

Multiple Projections
-----------------

EZStitcher can create multiple projections from the same Z-stack:

.. code-block:: python

    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",     # Use max projection for reference
        stitch_flatten="best_focus",            # Use best focus for final images
        additional_projections=["max", "mean"]  # Create additional projections
    )

This creates additional projections in the output directory, allowing you to compare different methods.

Best Practices
-----------

- **For position generation**: Use max projection (fastest and most reliable)
- **For final stitching**: Use best focus for most applications
- **For 3D analysis**: Use per-plane stitching
- **For noisy images**: Use mean projection to reduce noise
- **For focus detection**: Use combined method for best results

For practical examples of Z-stack processing, see the :doc:`../examples/zstack_processing` guide.
