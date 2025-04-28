==========
Basic Usage
==========

EZStitcher offers two main approaches for creating stitching pipelines:

1. Using ``AutoPipelineFactory`` for convenient, pre-configured pipelines
2. Building custom pipelines for maximum flexibility and control

Both approaches are valid and powerful, with different strengths depending on your needs. This guide will show you how to use both approaches for common stitching tasks.

Using EZStitcher
-------------------

Using AutoPipelineFactory
^^^^^^^^^^^^^^^^^^^^^

Here's a basic example of using EZStitcher with the pipeline factory:

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    from pathlib import Path

    # Path to your plate folder
    plate_path = Path("/path/to/your/plate")

    # Create orchestrator
    orchestrator = PipelineOrchestrator(plate_path=plate_path)

    # Create a factory with default settings
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        output_dir=plate_path.parent / f"{plate_path.name}_stitched",
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
^^^^^^^^^^^^^

Multi-Channel Data
""""""""""""""

For multi-channel data, you can specify weights for channel compositing:

.. code-block:: python

    # Create a factory for multi-channel data
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        channel_weights=[0.7, 0.3, 0]  # Use only first two channels for reference image
    )
    pipelines = factory.create_pipelines()

Z-Stack Data
""""""""""

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
"""""""""""""""

You can customize the normalization parameters:

.. code-block:: python

    # Create a factory with custom normalization
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        normalization_params={'low_percentile': 0.5, 'high_percentile': 99.5}
    )
    pipelines = factory.create_pipelines()

For more information about the pipeline factory, see :ref:`pipeline-factory-concept` in the concepts documentation.

Building Custom Pipelines
^^^^^^^^^^^^^^^^^^^^^

For maximum flexibility, you can build custom pipelines by directly specifying each step:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.step_factories import ZFlatStep, CompositeStep
    from ezstitcher.core.image_processor import ImageProcessor as IP
    from pathlib import Path

    # Create configuration with single-threaded processing
    config = PipelineConfig(
        num_workers=1  # Use a single worker thread
    )

    # Path to your plate folder
    plate_path = Path("/path/to/your/plate")

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path=plate_path
    )

    # Create position generation pipeline
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Create composite for position generation
            CompositeStep(),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    positions_dir = position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=plate_path.parent / f"{plate_path.name}_stitched",
        steps=[
            # Step 1: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(positions_dir=positions_dir)
        ],
        name="Image Assembly Pipeline"
    )

Finally, run the pipeline:

.. code-block:: python

    # Run the pipelines
    success = orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

    if success:
        print("Pipelines completed successfully!")
        print(f"Stitched images are in: {plate_path.parent / f'{plate_path.name}_stitched'}")
    else:
        print("Pipelines failed. Check logs for details.")

Choosing Between Approaches
------------------------

Both approaches have their strengths:

**AutoPipelineFactory:**
- Convenient for common workflows
- Requires less code
- Handles many details automatically
- Good for getting started quickly

**Custom Pipelines:**
- Maximum flexibility and control
- Terse and elegant for specific use cases
- Direct access to all pipeline features
- Ability to create highly customized workflows

The choice between them depends on your specific requirements and preferences. Many users start with ``AutoPipelineFactory`` for simple tasks and move to custom pipelines as their needs become more specialized.

Understanding Pipeline Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a detailed explanation of pipeline parameters, see :ref:`pipeline-parameters`.

In the example above, we used several key parameters:

* **input_dir**: Set to `orchestrator.workspace_path` to use the workspace directory as input
* **output_dir**: Set to a custom path for the final stitched images
* **steps**: A list of processing steps to execute in sequence
* **name**: A descriptive name for the pipeline for logging purposes

For detailed information about step parameters, including variable_components and group_by, see :ref:`step-parameters` in the :doc:`../concepts/step` documentation.

Directory Management
^^^^^^^^^^^^^^^^^

In the example above, we used EZStitcher's automatic directory resolution system (see :ref:`directory-resolution` for details):

* Set `input_dir=orchestrator.workspace_path` to use workspace copies of images
* Set a custom output directory for the final stitched images
* Only specified an output directory for the first step
* Let specialized steps automatically resolve their directories

This minimizes manual directory management while ensuring proper data flow. See :ref:`directory-best-practices` for recommended practices.

Processing a Plate Folder
------------------------

When working with plate-based experiments, you'll often want to process multiple wells. The PipelineOrchestrator handles this automatically, but you can also specify which wells to process.

Processing All Wells
^^^^^^^^^^^^^^^^^^^

By default, the orchestrator processes all wells in the plate:

.. code-block:: python

    # Process all wells
    orchestrator.run(pipelines=[pipeline])

Processing Specific Wells
^^^^^^^^^^^^^^^^^^^^^^^

To process only specific wells, use the well_filter parameter:

.. code-block:: python

    # Process only wells A01 and B02
    orchestrator.run(
        pipelines=[pipeline],
        well_filter=["A01", "B02"]
    )

Multithreaded Processing
^^^^^^^^^^^^^^^^^^^^^^

For faster processing, you can use multiple worker threads. For detailed information on multithreaded processing, see :ref:`pipeline-multithreaded`.

.. code-block:: python

    # Create configuration with multithreaded processing
    config = PipelineConfig(
        num_workers=4  # Use 4 worker threads
    )

    # Create orchestrator with multithreading
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path=plate_path
    )

    # Run the pipeline with multithreading
    orchestrator.run(pipelines=[pipeline])

Common Image Processing Operations
--------------------------------

EZStitcher provides a variety of image processing functions through the ImageProcessor class. For detailed information about function handling patterns, see :ref:`function-handling`. For a comprehensive guide to all image processing operations, see :doc:`../api/image_processing_operations`.

Here are some common operations:

Normalization
^^^^^^^^^^^

Normalize image intensities to a standard range:

.. code-block:: python

    # Percentile-based normalization
    Step(
        name="Normalize Images",
        func=(IP.stack_percentile_normalize, {
            'low_percentile': 1.0,  # Bottom 1% becomes black
            'high_percentile': 99.0  # Top 1% becomes white
        })
    )

Background Removal
^^^^^^^^^^^^^^^

Remove background using tophat filtering:

.. code-block:: python

    from ezstitcher.core.utils import stack

    # Apply tophat filter to each image in the stack
    Step(
        name="Remove Background",
        func=(stack(IP.tophat), {'size': 15})  # Function with filter size
    )

Image Sharpening
^^^^^^^^^^^^^

Enhance image details:

.. code-block:: python

    # Sharpen images
    Step(
        name="Sharpen Images",
        func=(stack(IP.sharpen), {
            'sigma': 1.0,  # Gaussian blur sigma
            'amount': 1.5   # Sharpening amount
        })
    )

Combining Multiple Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can apply multiple operations in sequence:

.. code-block:: python

    # Apply multiple operations in sequence
    Step(
        name="Enhance Images",
        func=[
            (stack(IP.tophat), {'size': 15}),                  # First remove background with args
            (stack(IP.sharpen), {'sigma': 1.0, 'amount': 1.5}),  # Then sharpen with args
            (IP.stack_percentile_normalize, {'low_percentile': 1.0, 'high_percentile': 99.0})  # Finally normalize with args
        ]
    )

Channel-Specific Processing
^^^^^^^^^^^^^^^^^^^^^^^^

Apply different processing to different channels using a dictionary of functions. For detailed information about this pattern, see :ref:`function-dictionaries`.

.. code-block:: python

    # Define channel-specific processing functions
    def process_dapi(images):
        """Process DAPI channel images."""
        # Apply tophat and normalize
        images = [IP.tophat(img, size=15) for img in images]
        return IP.stack_percentile_normalize(images)

    def process_gfp(images):
        """Process GFP channel images."""
        # Apply sharpen and normalize
        images = [IP.sharpen(img, sigma=1.0, amount=1.5) for img in images]
        return IP.stack_percentile_normalize(images)

    # Apply different processing to different channels
    Step(
        name="Channel-Specific Processing",
        func={
            "1": process_dapi,  # Apply process_dapi to channel 1
            "2": process_gfp    # Apply process_gfp to channel 2
        },
        group_by='channel'  # Specifies that keys "1" and "2" refer to channel values
    )

Saving and Loading Pipelines
--------------------------

For information on saving and loading pipelines, see :ref:`pipeline-saving-loading`.

Here's a practical example of how to create a reusable pipeline configuration using pipeline factories:

.. code-block:: python

    # pipeline_config.py
    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    from ezstitcher.core.config import PipelineConfig
    from pathlib import Path

    def run_basic_stitching(plate_path, num_workers=1, normalize=True):
        """Run a basic stitching pipeline on the specified plate."""
        # Create orchestrator with specified number of workers
        orchestrator = PipelineOrchestrator(
            config=PipelineConfig(num_workers=num_workers),
            plate_path=plate_path
        )

        # Create pipelines using AutoPipelineFactory
        factory = AutoPipelineFactory(
            input_dir=plate_path,
            output_dir=plate_path.parent / f"{plate_path.name}_stitched",
            normalize=normalize
        )
        pipelines = factory.create_pipelines()

        # Run the pipeline and return success status
        return orchestrator.run(pipelines=pipelines)

And here's how to use this in another script:

.. code-block:: python

    # run_stitching.py
    from pathlib import Path
    from pipeline_config import run_basic_stitching

    # Path to your plate folder
    plate_path = Path("/path/to/your/plate")

    # Run the stitching pipeline
    success = run_basic_stitching(
        plate_path=plate_path,
        num_workers=4,
        normalize=True
    )

    if success:
        print(f"Stitching completed successfully! Output in: {plate_path.parent / f'{plate_path.name}_stitched'}")
    else:
        print("Stitching failed. Check logs for details.")

Best Practices
^^^^^^^^^^^^^

For comprehensive best practices, see:

* :ref:`best-practices-pipeline` - Best practices for pipeline creation and configuration
* :ref:`best-practices-directory` - Best practices for directory management
* :ref:`best-practices-specialized-steps` - Best practices for specialized steps
* :ref:`best-practices-function-handling` - Best practices for function handling
* :ref:`best-practices-performance` - Best practices for performance optimization

Or visit the complete :doc:`best_practices` guide.

Next Steps
---------

Now that you understand the basics of using EZStitcher, you can:

1. **Learn about specialized steps**: For information about specialized steps like ZFlatStep, FocusStep, and CompositeStep, see :doc:`../concepts/specialized_steps`.

2. **Study pipeline concepts**: For a deeper understanding of pipelines, see :doc:`../concepts/pipeline`.

3. **Dive into intermediate usage**: For more advanced techniques like channel-specific processing and Z-stack handling, see :doc:`intermediate_usage`.

4. **Follow the learning path**: For a comprehensive learning path that will guide you through intermediate and advanced topics, see :ref:`learning-path` in the introduction.
