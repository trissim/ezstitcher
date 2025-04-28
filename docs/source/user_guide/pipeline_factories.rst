.. warning::
   This documentation is deprecated. The individual factory functions shown here have been replaced by the unified ``AutoPipelineFactory``. Please refer to :doc:`basic_usage` for information about using pipeline factories.

Pipeline Factories (Deprecated)
===========================

This documentation has been deprecated. The individual factory functions shown here have been replaced by the unified ``AutoPipelineFactory``.

See :doc:`basic_usage` for examples of using the ``AutoPipelineFactory`` and :doc:`../concepts/pipeline_factory` for detailed information.

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
