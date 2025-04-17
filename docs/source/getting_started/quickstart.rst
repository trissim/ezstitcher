Quick Start Guide
===============

This guide will help you get started with EZStitcher quickly.

Basic Usage with Function-Based API
----------------------------------

The simplest way to use EZStitcher is with the function-based API:

.. code-block:: python

    from ezstitcher.core.main import process_plate_folder

    # Process a plate folder with automatic microscope detection
    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1"],
        tile_overlap=10.0
    )

This will:

1. Automatically detect the microscope type
2. Process all images in the plate folder
3. Generate stitching positions
4. Stitch images using the specified reference channel
5. Save the stitched images to a new folder

Basic Usage with Object-Oriented API
-----------------------------------

For more control, you can use the object-oriented API:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1"],
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

This approach gives you more flexibility to customize the processing pipeline.

Command-Line Interface
--------------------

EZStitcher also provides a command-line interface:

.. code-block:: bash

    # Basic usage
    ezstitcher /path/to/plate_folder --reference-channels 1 --tile-overlap 10

    # Z-stack processing
    ezstitcher /path/to/plate_folder --reference-channels 1 --focus-detect --focus-method combined

    # Help
    ezstitcher --help

Minimal Working Example
---------------------

Here's a complete example showing how to process a plate folder with Z-stacks:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration for Z-stack processing
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",  # Use max projection for position generation
        stitch_flatten="best_focus",         # Use best focus for final stitching
        focus_method="combined",             # Use combined focus metric
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run("path/to/plate_folder")

    if success:
        print("Processing completed successfully!")
    else:
        print("Processing failed.")

Expected Output
-------------

After running EZStitcher, you'll see the following output directories:

.. code-block:: text

    path/to/plate_folder/                 # Original data
    path/to/plate_folder_processed/       # Processed individual tiles
    path/to/plate_folder_post_processed/  # Post-processed images
    path/to/plate_folder_positions/       # CSV files with stitching positions
    path/to/plate_folder_stitched/        # Final stitched images

Next Steps
---------

- Learn about :doc:`basic_concepts` to understand how EZStitcher works
- Explore the :doc:`../user_guide/core_concepts` for more details
- Check out the :doc:`../examples/basic_stitching` for practical examples
