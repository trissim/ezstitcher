Z-Stack Processing
=================

This example demonstrates how to process Z-stack microscopy images.

Best Focus Detection
------------------

.. code-block:: python

    from ezstitcher.core import process_plate_folder

    # Process Z-stack data with best focus detection
    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1"],
        focus_detect=True,
        focus_method="combined"
    )

Z-Stack Projections
-----------------

.. code-block:: python

    from ezstitcher.core import process_plate_folder

    # Process Z-stack data with projections
    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1"],
        create_projections=True,
        projection_types=["max", "mean", "std"]
    )

Per-Plane Z-Stack Stitching
-------------------------

.. code-block:: python

    from ezstitcher.core import process_plate_folder

    # Process Z-stack data with per-plane stitching
    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1"],
        tile_overlap=10,
        create_projections=True,          # Create projections for position detection
        projection_types=["max"],         # Use max projection
        stitch_z_reference="max",         # Use max projection for reference positions
        stitch_all_z_planes=True          # Stitch each Z-plane using the same positions
    )

Custom Projection Function
------------------------

.. code-block:: python

    import numpy as np
    from ezstitcher.core.config import ZStackProcessorConfig, PlateProcessorConfig
    from ezstitcher.core.plate_processor import PlateProcessor

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

    # Create configuration
    zstack_config = ZStackProcessorConfig(
        create_projections=True,
        stitch_z_reference=weighted_projection,  # Use custom function
        stitch_all_z_planes=True
    )

    plate_config = PlateProcessorConfig(
        reference_channels=["1"],
        z_stack_processor=zstack_config
    )

    # Create and run the plate processor
    processor = PlateProcessor(plate_config)
    processor.run("path/to/plate_folder")

Percentile Normalized Projection
------------------------------

.. code-block:: python

    import numpy as np
    from ezstitcher.core.config import ZStackProcessorConfig, PlateProcessorConfig
    from ezstitcher.core.plate_processor import PlateProcessor
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

    # Create Z-stack processor configuration with the custom projection function
    zstack_config = ZStackProcessorConfig(
        create_projections=True,
        projection_types=["max"],  # Standard projections to create
        stitch_z_reference=percentile_normalized_projection,  # Use our custom function for stitching
        stitch_all_z_planes=True  # Stitch each Z-plane using the same positions
    )

    # Create plate processor configuration
    plate_config = PlateProcessorConfig(
        reference_channels=["1"],  # Use channel 1 as reference
        z_stack_processor=zstack_config
    )

    # Create and run the plate processor
    processor = PlateProcessor(plate_config)
    processor.run("path/to/plate_folder")
