Z-Stack Processing Examples
=========================

This page provides practical examples of Z-stack processing with EZStitcher.

Basic Z-Stack Processing
---------------------

Process Z-stacks with maximum intensity projection:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration for max projection
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",  # Use max projection for reference
        stitch_flatten="max_projection"      # Use max projection for final stitching
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/zstack_plate")

Best Focus Detection
-----------------

Detect the best focused plane in each Z-stack:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Configure Z-stack processing with best focus detection
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",  # Use max projection for reference
        stitch_flatten="best_focus",         # Use best focus for final images
        focus_config=FocusAnalyzerConfig(
            method="combined",               # Combined focus metrics
            roi=(100, 100, 200, 200)         # Optional ROI for focus detection
        )
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/zstack_plate")

Multiple Projections
-----------------

Create multiple projections from the same Z-stack:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Configure Z-stack processing with multiple projections
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",     # Use max projection for reference
        stitch_flatten="best_focus",            # Use best focus for final images
        additional_projections=["max", "mean"]  # Create additional projections
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/zstack_plate")

Per-Plane Stitching
----------------

Stitch each Z-plane separately:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Configure Z-stack processing with per-plane stitching
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",  # Use max projection for reference
        stitch_flatten=None                  # Stitch each Z-plane separately
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/zstack_plate")

Complete Z-Stack Workflow
----------------------

A complete workflow for Z-stack processing:

.. code-block:: python

    import os
    from ezstitcher.core.config import PipelineConfig, StitcherConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Input directory
    plate_folder = "path/to/zstack_plate"

    # Configure Z-stack processing
    config = PipelineConfig(
        # Basic settings
        reference_channels=["1"],
        well_filter=["A01", "A02"],  # Process only these wells

        # Z-stack settings
        reference_flatten="max_projection",  # Use max projection for reference
        stitch_flatten="best_focus",         # Use best focus for final images

        # Focus detection settings
        focus_config=FocusAnalyzerConfig(
            method="combined",               # Combined focus metrics
            roi=(100, 100, 200, 200)         # Optional ROI for focus detection
        ),

        # Stitching settings
        stitcher=StitcherConfig(
            tile_overlap=10.0,               # 10% overlap between tiles
            max_shift=50                     # Maximum shift in pixels
        )
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(plate_folder)

    if success:
        print(f"Z-stack processing completed successfully!")
        print(f"Stitched images saved to: {os.path.join(plate_folder + '_stitched')}")
    else:
        print("Z-stack processing failed.")

Command Line Interface
--------------------

Z-stack processing can also be done through the command-line interface:

.. code-block:: bash

    # Z-stack processing with max projection
    ezstitcher /path/to/zstack_plate --reference-channels 1 --reference-flatten max --stitch-flatten max

    # Z-stack processing with best focus detection
    ezstitcher /path/to/zstack_plate --reference-channels 1 --reference-flatten max --stitch-flatten best_focus

    # Z-stack processing with multiple projections
    ezstitcher /path/to/zstack_plate --reference-channels 1 --reference-flatten max --stitch-flatten best_focus --additional-projections max,mean
