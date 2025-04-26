==========
Basic Usage
==========

Getting Started
--------------

Here's a basic example of using EZStitcher:

.. code-block:: python

    from ezstitcher.core import create_basic_pipeline
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    from pathlib import Path

    # Path to your plate folder
    plate_path = Path("/path/to/your/plate")

    # Create orchestrator
    orchestrator = PipelineOrchestrator(plate_path=plate_path)

    # Create a basic stitching pipeline
    pipelines = create_basic_pipeline(
        input_dir=plate_path,
        output_dir=plate_path.parent / f"{plate_path.name}_stitched",
        normalize=True  # Apply normalization (default)
    )

    # Run the pipeline
    orchestrator.run(pipelines=pipelines)

Building Custom Pipelines
^^^^^^^^^^^^^^^^^^^^^^^

For more control, you can build custom pipelines by manually specifying each step:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
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

    # Create a pipeline with three steps
    pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,    # Pipeline input directory
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched", # Pipeline output directory
        steps=[
            # Step 1: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize,
                output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_processed"
            ),

            # Step 2: Generate positions for stitching
            PositionGenerationStep(),

            # Step 3: Stitch images
            ImageStitchingStep()
        ],
        name="Basic Processing Pipeline"
    )

Finally, run the pipeline:

.. code-block:: python

    # Run the pipeline
    success = orchestrator.run(pipelines=[pipeline])

    if success:
        print("Pipeline completed successfully!")
        print(f"Stitched images are in: {orchestrator.plate_path.parent / f'{orchestrator.plate_path.name}_stitched'}")
    else:
        print("Pipeline failed. Check logs for details.")

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
    from ezstitcher.core import create_basic_stitching_pipeline
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

        # Create pipelines using factory function
        pipelines = create_basic_stitching_pipeline(
            input_dir=plate_path,
            output_dir=plate_path.parent / f"{plate_path.name}_stitched",
            normalize=normalize
        )

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

Now that you understand the basics of creating and running pipelines, you're ready to explore more advanced topics. For a comprehensive learning path that will guide you through intermediate and advanced topics, see :ref:`learning-path` in the introduction.
