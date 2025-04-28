Pipeline Factory
==============

EZStitcher provides a unified ``AutoPipelineFactory`` class that creates pre-configured pipelines for common workflows. This factory simplifies pipeline creation by automatically configuring the appropriate steps based on input parameters.

Basic Usage
----------

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Create orchestrator
    orchestrator = PipelineOrchestrator(plate_path=plate_path)

    # Create a factory with default settings
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True  # Apply normalization (default)
    )
    
    # Create the pipelines
    pipelines = factory.create_pipelines()
    
    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

The ``AutoPipelineFactory`` creates two pipelines:

1. **Position Generation Pipeline**: Creates position files for stitching
2. **Image Assembly Pipeline**: Stitches images using the position files

Common Use Cases
--------------

Multi-Channel Data
^^^^^^^^^^^^^^^

For multi-channel data, you can specify weights for channel compositing:

.. code-block:: python

    # Create a factory for multi-channel data
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        channel_weights=[0.7, 0.3, 0]  # Use only first two channels for reference image
    )
    pipelines = factory.create_pipelines()

Z-Stack Data
^^^^^^^^^^

For Z-stack data, you can control Z-stack flattening:

.. code-block:: python

    # Create a factory for Z-stack data
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        flatten_z=True,  # Flatten Z-stacks in the assembly pipeline
        z_method="max"   # Use maximum intensity projection
    )
    pipelines = factory.create_pipelines()

Custom Normalization
^^^^^^^^^^^^^^^^^

You can customize the normalization parameters:

.. code-block:: python

    # Create a factory with custom normalization
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        normalization_params={'low_percentile': 0.5, 'high_percentile': 99.5}
    )
    pipelines = factory.create_pipelines()

Configuration Options
------------------

The ``AutoPipelineFactory`` supports several configuration options:

- ``input_dir``: Input directory containing images (required)
- ``output_dir``: Output directory for stitched images (optional)
- ``normalize``: Whether to include normalization (default: True)
- ``normalization_params``: Parameters for normalization (optional)
- ``well_filter``: Wells to process (optional)
- ``flatten_z``: Whether to flatten Z-stacks in the assembly pipeline (default: False)
- ``z_method``: Z-stack flattening method (default: "max")
- ``channel_weights``: Weights for channel compositing in the reference image (optional)

Important behaviors to note:

- Z-stacks are always flattened for position generation regardless of the ``flatten_z`` setting
- Channel compositing is always performed for position generation
- If ``channel_weights`` is None, weights are distributed evenly across all channels

Customizing Pipelines
-------------------

You can customize the pipelines created by the ``AutoPipelineFactory`` after creation:

.. code-block:: python

    # Create basic pipelines
    factory = AutoPipelineFactory(input_dir=orchestrator.workspace_path)
    pipelines = factory.create_pipelines()

    # Access individual pipelines
    position_pipeline = pipelines[0]
    assembly_pipeline = pipelines[1]

    # Add custom step to position generation pipeline
    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_processor import ImageProcessor as IP

    position_pipeline.add_step(
        Step(
            func=IP.sharpen,
            name="Sharpen Images"
        )
    )

    # Run the modified pipelines
    orchestrator.run(pipelines=pipelines)

This approach allows you to leverage the convenience of the factory while still maintaining the flexibility to customize the pipelines for specific needs.

.. seealso::
   - :ref:`pipeline-factory-concept` for more information about pipeline factories
   - :doc:`../concepts/specialized_steps` for more information about specialized steps
   - :doc:`intermediate_usage` for more advanced examples
