.. warning::
   This documentation is deprecated. Please refer to :doc:`basic_usage` for information about using pipeline factories.

Pipeline Factory (Deprecated)
==========================

This documentation has been moved to :doc:`basic_usage`.

The AutoPipelineFactory is now the recommended way to create pipelines for common workflows.
See :doc:`basic_usage` for examples and :doc:`../concepts/pipeline_factory` for detailed information.

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

For Z-stack data, you can control Z-stack processing using either projection methods or focus detection:

.. code-block:: python

    # Create a factory for Z-stack data with projection
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        flatten_z=True,  # Flatten Z-stacks in the assembly pipeline
        z_method="max"   # Use maximum intensity projection
    )
    pipelines = factory.create_pipelines()

    # Create a factory for Z-stack data with focus detection
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        flatten_z=True,  # Flatten Z-stacks in the assembly pipeline
        z_method="combined"   # Use combined focus metric
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
- ``z_method``: Z-stack processing method (default: "max")
  - Projection methods: "max", "mean", "median", etc.
  - Focus detection methods: "combined", "laplacian", "tenengrad", "normalized_variance", "fft"
- ``channel_weights``: Weights for channel compositing in the reference image (optional)

Important behaviors to note:

- Z-stacks are always flattened for position generation regardless of the ``flatten_z`` setting
- Channel compositing is always performed for position generation
- If ``channel_weights`` is None, weights are distributed evenly across all channels

Creating Custom Pipelines
-------------------

For workflows that require customization beyond what AutoPipelineFactory parameters provide, creating custom pipelines from scratch is the recommended approach:

.. code-block:: python

    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step
    from ezstitcher.core.step_factories import ZFlatStep, CompositeStep, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create a custom position generation pipeline
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Normalize images
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 2: Apply custom enhancement
            Step(
                name="Sharpen Images",
                func=IP.sharpen
            ),

            # Step 3: Create composite for position generation
            CompositeStep(weights=[0.7, 0.3, 0]),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Custom Position Generation Pipeline"
    )

    # Create a custom assembly pipeline
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Normalize images
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 2: Stitch images
            ImageStitchingStep()
        ],
        name="Custom Assembly Pipeline"
    )

    # Run the custom pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

This approach provides several benefits:

1. **Readability**: The pipeline structure is explicit and easy to understand
2. **Maintainability**: Changes can be made directly to the pipeline definition
3. **Flexibility**: Complete control over each step and its parameters
4. **Robustness**: No risk of unexpected behavior from modifying factory pipelines

.. important::
   While it is technically possible to modify pipelines created by AutoPipelineFactory after creation,
   this approach is generally not recommended. Creating custom pipelines from scratch is usually more
   readable, maintainable, and less error-prone for any workflow that requires customization beyond
   what AutoPipelineFactory parameters provide.

.. seealso::
   - :ref:`pipeline-factory-concept` for more information about pipeline factories
   - :doc:`../concepts/specialized_steps` for more information about specialized steps
   - :doc:`intermediate_usage` for more advanced examples
