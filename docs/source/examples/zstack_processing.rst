Z-Stack Processing
================

This example demonstrates how to process Z-stacks with EZStitcher.

Z-Stack Max Projection
--------------------

The simplest way to process Z-stacks is to create a maximum intensity projection:

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

EZStitcher can detect the best focused plane in each Z-stack:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Configure Z-stack processing with best focus detection
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",  # Use max projection for reference
        stitch_flatten="best_focus",         # Use best focus for final images
        focus_method="combined",             # Combined focus metrics
        focus_config=FocusAnalyzerConfig(
            method="combined",
            roi=(100, 100, 200, 200)         # Optional ROI for focus detection
        )
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/zstack_plate")

Multiple Projections
-----------------

EZStitcher can create multiple projections from the same Z-stack:

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

EZStitcher can stitch each Z-plane separately:

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
