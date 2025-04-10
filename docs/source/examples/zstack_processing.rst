Z-Stack Processing
=================

This example demonstrates how to process Z-stack microscopy images.

Best Focus Detection
------------------

.. code-block:: python

    from ezstitcher.core.main import process_plate_auto

    # Process Z-stack data with best focus detection
    process_plate_auto(
        'path/to/plate_folder',
        **{
            "reference_channels": ["1"],
            "z_stack_processor.focus_detect": True,
            "z_stack_processor.focus_method": "combined"
        }
    )

Z-Stack Projections
-----------------

.. code-block:: python

    from ezstitcher.core.main import process_plate_auto

    # Process Z-stack data with projections
    process_plate_auto(
        'path/to/plate_folder',
        **{
            "reference_channels": ["1"],
            "z_stack_processor.create_projections": True,
            "z_stack_processor.projection_types": ["max", "mean", "std"]
        }
    )

Per-Plane Z-Stack Stitching
-------------------------

.. code-block:: python

    from ezstitcher.core.main import process_plate_auto

    # Process Z-stack data with per-plane stitching
    process_plate_auto(
        'path/to/plate_folder',
        **{
            "reference_channels": ["1"],
            "stitcher.tile_overlap": 10,
            "z_stack_processor.create_projections": True,          # Create projections for position detection
            "z_stack_processor.projection_types": ["max"],         # Use max projection
            "z_stack_processor.stitch_z_reference": "max",         # Use max projection for reference positions
            "z_stack_processor.stitch_all_z_planes": True          # Stitch each Z-plane using the same positions
        }
    )

Custom Projection Function
------------------------

.. code-block:: python

    import numpy as np
    from ezstitcher.core.main import process_plate_auto

    # Define a custom projection function
    def weighted_projection(z_stack):
        """
        Create a weighted projection of a Z-stack.

        Args:
            z_stack (list): List of images in the Z-stack

        Returns:
            numpy.ndarray: Weighted projection image
        """
        # Convert to numpy array
        stack = np.array(z_stack)

        # Create weights that emphasize the middle planes
        weights = np.ones(len(z_stack))
        mid_point = len(z_stack) // 2
        for i in range(len(z_stack)):
            weights[i] = 1.0 - 0.5 * abs(i - mid_point) / mid_point

        # Apply weights
        weighted_stack = stack * weights[:, np.newaxis, np.newaxis]

        # Return the sum
        return np.sum(weighted_stack, axis=0) / np.sum(weights)

    # Process Z-stack data with custom projection function
    process_plate_auto(
        'path/to/plate_folder',
        **{
            "reference_channels": ["1"],
            "stitcher.tile_overlap": 10,
            "z_stack_processor.create_projections": True,
            "z_stack_processor.stitch_z_reference": weighted_projection,  # Use custom function
            "z_stack_processor.stitch_all_z_planes": True
        }
    )

Percentile Normalized Projection
------------------------------

.. code-block:: python

    import numpy as np
    from ezstitcher.core.main import process_plate_auto
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Create an ImagePreprocessor instance
    preprocessor = ImagePreprocessor()

    # Define a custom projection function using percentile normalization
    def percentile_normalized_projection(z_stack):
        """
        Create a percentile-normalized projection of a Z-stack.

        This function normalizes the entire stack using percentile-based contrast stretching,
        then creates a maximum intensity projection.

        Args:
            z_stack (list): List of images in the Z-stack

        Returns:
            numpy.ndarray: Normalized projection image
        """
        # Normalize the stack using percentile-based contrast stretching
        normalized_stack = preprocessor.stack_percentile_normalize(
            z_stack,
            low_percentile=2,
            high_percentile=98
        )

        # Create a maximum intensity projection
        projection = np.max(normalized_stack, axis=0)

        return projection

    # Process Z-stack data with custom projection function
    process_plate_auto(
        'path/to/plate_folder',
        **{
            "reference_channels": ["1"],  # Use channel 1 as reference
            "z_stack_processor.create_projections": True,
            "z_stack_processor.projection_types": ["max"],  # Standard projections to create
            "z_stack_processor.stitch_z_reference": percentile_normalized_projection,  # Use our custom function for stitching
            "z_stack_processor.stitch_all_z_planes": True  # Stitch each Z-plane using the same positions
        }
    )
