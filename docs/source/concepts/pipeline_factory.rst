.. _pipeline-factory-concept:

===============
Pipeline Factory
===============

.. _pipeline-factory-overview:

Overview
--------

The ``AutoPipelineFactory`` is a unified factory class that creates pre-configured pipelines for all common stitching workflows. It simplifies pipeline creation by automatically configuring the appropriate steps based on input parameters, with no need to differentiate between different types of pipelines.

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Create a factory with custom configuration
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        flatten_z=True,  # Flatten Z-stacks in the assembly pipeline
        z_method="max",  # Use maximum intensity projection
        channel_weights=[0.7, 0.3, 0]  # Use only first two channels for reference image
    )

    # Create the pipelines
    pipelines = factory.create_pipelines()

    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

The ``AutoPipelineFactory`` handles all types of stitching workflows with a single implementation:

- 2D single-channel stitching
- 2D multi-channel stitching
- Z-stack per plane stitching
- Z-stack projection stitching

This unified approach simplifies the API and makes it easier to create pipelines for common use cases.

.. _pipeline-factory-structure:

Pipeline Structure
-----------------

The ``AutoPipelineFactory`` creates two pipelines with a consistent structure:

1. **Position Generation Pipeline**: Creates position files for stitching
   - Steps: [flatten Z (if flatten_z=True), normalize (if normalize=True), create_composite (always), generate positions (always)]
   - Purpose: Process images and generate position files for stitching

2. **Image Assembly Pipeline**: Stitches images using the position files
   - Steps: [normalize (if normalize=True), flatten Z (if flatten_z=True), stitch_images (always)]
   - Purpose: Process and stitch images using the position files

This structure is consistent regardless of data type (single/multi-channel, single/multi-Z), with parameters controlling step behavior rather than pipeline structure.

.. _pipeline-factory-parameters:

Parameters
---------

The ``AutoPipelineFactory`` accepts the following parameters:

- ``input_dir``: Input directory containing images
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

.. _pipeline-factory-specialized-steps:

Specialized Steps
---------------

The ``AutoPipelineFactory`` uses specialized steps from the :doc:`specialized_steps` module:

- ``ZFlatStep``: For Z-stack flattening (used in both pipelines when appropriate)
- ``CompositeStep``: For channel compositing (always used in position generation)
- ``PositionGenerationStep``: For generating position files
- ``ImageStitchingStep``: For stitching images

These specialized steps simplify the pipeline creation process by encapsulating common operations with appropriate defaults.

.. _pipeline-factory-examples:

Examples
-------

Basic Single-Channel Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True
    )
    pipelines = factory.create_pipelines()

Multi-Channel Pipeline with Custom Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        channel_weights=[0.7, 0.3, 0]  # Use only first two channels for reference image
    )
    pipelines = factory.create_pipelines()

Z-Stack Pipeline with Projection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        flatten_z=True,  # Flatten Z-stacks in the assembly pipeline
        z_method="max"   # Use maximum intensity projection
    )
    pipelines = factory.create_pipelines()

Pipeline with Custom Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        normalization_params={'low_percentile': 0.5, 'high_percentile': 99.5}
    )
    pipelines = factory.create_pipelines()

.. _pipeline-factory-customization:

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
   - :doc:`pipeline` for more information about pipelines
   - :doc:`specialized_steps` for more information about specialized steps
   - :doc:`../user_guide/basic_usage` for beginner examples
   - :doc:`../user_guide/intermediate_usage` for intermediate examples
