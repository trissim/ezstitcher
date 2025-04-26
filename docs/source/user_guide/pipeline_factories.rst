Pipeline Factories
=================

EZStitcher provides a set of factory functions that create pre-configured pipelines for common workflows. These factories provide a higher-level interface for creating pipelines, making it easier to construct pipelines for common use cases.

Available Factory Functions
--------------------------

- ``create_basic_stitching_pipeline``: For single-channel, single-Z stitching
- ``create_multichannel_stitching_pipeline``: For multi-channel stitching
- ``create_zstack_stitching_pipeline``: For Z-stack stitching with projection
- ``create_focus_stitching_pipeline``: For Z-stack stitching with focus selection

Basic Stitching Pipeline
-----------------------

The ``create_basic_stitching_pipeline`` function creates a pipeline for stitching single-channel, single-Z data.

.. code-block:: python

    from ezstitcher.core import create_basic_stitching_pipeline
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Create a basic stitching pipeline
    pipelines = create_basic_stitching_pipeline(
        input_dir="path/to/images",
        output_dir="path/to/output",
        normalize=True
    )

    # Run the pipeline
    orchestrator = PipelineOrchestrator()
    orchestrator.run(pipelines=pipelines)

Multi-Channel Stitching Pipeline
-------------------------------

The ``create_multichannel_stitching_pipeline`` function creates a pipeline for stitching multi-channel data.

.. code-block:: python

    from ezstitcher.core import create_multichannel_stitching_pipeline
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Create a multi-channel stitching pipeline
    pipelines = create_multichannel_stitching_pipeline(
        input_dir="path/to/images",
        output_dir="path/to/output",
        composite_weights=[0.7, 0.3],  # Optional weights for channel compositing
        stitch_channels_separately=True  # Whether to stitch each channel separately
    )

    # Run the pipeline
    orchestrator = PipelineOrchestrator()
    orchestrator.run(pipelines=pipelines)

Z-Stack Stitching Pipeline
-------------------------

The ``create_zstack_stitching_pipeline`` function creates a pipeline for stitching Z-stack data with projection.

.. code-block:: python

    from ezstitcher.core import create_zstack_stitching_pipeline
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Create a Z-stack stitching pipeline with maximum intensity projection
    pipelines = create_zstack_stitching_pipeline(
        input_dir="path/to/images",
        output_dir="path/to/output",
        z_processing_method="projection",
        z_processing_options={'method': 'max'}
    )

    # Run the pipeline
    orchestrator = PipelineOrchestrator()
    orchestrator.run(pipelines=pipelines)

Focus Stitching Pipeline
-----------------------

The ``create_focus_stitching_pipeline`` function creates a pipeline for stitching Z-stack data with focus selection.

.. code-block:: python

    from ezstitcher.core import create_focus_stitching_pipeline
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Create a focus stitching pipeline
    pipelines = create_focus_stitching_pipeline(
        input_dir="path/to/images",
        output_dir="path/to/output",
        focus_metric="laplacian"  # Focus metric to use
    )

    # Run the pipeline
    orchestrator = PipelineOrchestrator()
    orchestrator.run(pipelines=pipelines)
