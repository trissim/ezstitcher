Pipeline Factories
================

EZStitcher provides several factory functions to create common pipeline configurations.

Basic Stitching Pipeline
----------------------

.. code-block:: python

    from ezstitcher.core import create_basic_pipeline
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    pipelines = create_basic_pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir="path/to/output",
        normalize=True  # Apply normalization
    )

Multi-Channel Stitching Pipeline
-----------------------------

.. code-block:: python

    from ezstitcher.core import create_multichannel_pipeline
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    pipelines = create_multichannel_pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir="path/to/output",
        weights=[0.7, 0.3],  # Optional weights for channel compositing
        stitch_channels_separately=True  # Whether to stitch each channel separately
    )

Z-Stack Stitching Pipeline
------------------------

.. code-block:: python

    from ezstitcher.core import create_zstack_pipeline
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    pipelines = create_zstack_pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir="path/to/output",
        method="projection",
        method_options={'method': 'max'}
    )

Focus Stitching Pipeline
----------------------

.. code-block:: python

    from ezstitcher.core import create_focus_pipeline, FocusAnalyzer
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    pipelines = create_focus_pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir="path/to/output",
        metric="variance_of_laplacian"
    )
