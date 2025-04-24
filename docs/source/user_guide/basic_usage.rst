==========
Basic Usage
==========

This section provides detailed examples of basic EZStitcher usage to help you get started with common tasks.

Setting Up a Simple Pipeline
---------------------------

Let's start by creating a simple pipeline for processing microscopy images. We'll build a basic pipeline that:

1. Normalizes image intensities
2. Generates positions for stitching
3. Stitches the images together

First, import the necessary modules:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
    from pathlib import Path

Next, create a configuration and orchestrator:

.. code-block:: python

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

    # Set up directory structure
    dirs = orchestrator.setup_directories()

Now, create a pipeline with three steps:

.. code-block:: python

    # Create a pipeline
    pipeline = Pipeline(
        steps=[
            # Step 1: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize,
                variable_components=['channel'],
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            ),

            # Step 2: Generate positions for stitching
            PositionGenerationStep(
                name="Generate Positions",
                input_dir=dirs['processed'],
                output_dir=dirs['positions']
            ),

            # Step 3: Stitch images
            ImageStitchingStep(
                name="Stitch Images",
                input_dir=dirs['processed'],
                positions_dir=dirs['positions'],
                output_dir=dirs['stitched']
            )
        ],
        name="Basic Processing Pipeline"
    )

Finally, run the pipeline:

.. code-block:: python

    # Run the pipeline
    success = orchestrator.run(pipelines=[pipeline])

    if success:
        print("Pipeline completed successfully!")
        print(f"Stitched images are in: {dirs['stitched']}")
    else:
        print("Pipeline failed. Check logs for details.")

Understanding Pipeline Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's break down the key parameters used in the pipeline:

* **name**: A human-readable name for the pipeline or step
* **func**: The processing function to apply to images
* **variable_components**: Components that vary across files (e.g., 'channel', 'z_index')
* **input_dir**: The directory containing input images
* **output_dir**: The directory where processed images will be saved
* **positions_dir**: The directory containing position files (for ImageStitchingStep)

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

For faster processing, you can use multiple worker threads:

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
    # Each well will be processed in a separate thread
    orchestrator.run(pipelines=[pipeline])

Common Image Processing Operations
--------------------------------

EZStitcher provides a variety of image processing functions through the ImagePreprocessor class. Here are some common operations:

Normalization
^^^^^^^^^^^

Normalize image intensities to a standard range:

.. code-block:: python

    # Percentile-based normalization
    Step(
        name="Normalize Images",
        func=IP.stack_percentile_normalize,
        processing_args={
            'low_percentile': 1.0,  # Bottom 1% becomes black
            'high_percentile': 99.0  # Top 1% becomes white
        }
    )

Background Removal
^^^^^^^^^^^^^^^

Remove background using tophat filtering:

.. code-block:: python

    from ezstitcher.core.utils import stack

    # Apply tophat filter to each image in the stack
    Step(
        name="Remove Background",
        func=stack(IP.tophat),
        processing_args={'size': 15}  # Filter size
    )

Image Sharpening
^^^^^^^^^^^^^

Enhance image details:

.. code-block:: python

    # Sharpen images
    Step(
        name="Sharpen Images",
        func=stack(IP.sharpen),
        processing_args={
            'sigma': 1.0,  # Gaussian blur sigma
            'amount': 1.5   # Sharpening amount
        }
    )

Combining Multiple Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can apply multiple operations in sequence:

.. code-block:: python

    # Apply multiple operations in sequence
    Step(
        name="Enhance Images",
        func=[
            stack(IP.tophat),             # First remove background
            stack(IP.sharpen),            # Then sharpen
            IP.stack_percentile_normalize  # Finally normalize
        ],
        processing_args=[
            {'size': 15},                  # Args for tophat
            {'sigma': 1.0, 'amount': 1.5},  # Args for sharpen
            {'low_percentile': 1.0, 'high_percentile': 99.0}  # Args for normalize
        ]
    )

Channel-Specific Processing
^^^^^^^^^^^^^^^^^^^^^^^^

Apply different processing to different channels using a dictionary of functions:

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

In this example:
- The dictionary keys ("1" and "2") correspond to channel values
- `group_by='channel'` tells EZStitcher that the keys refer to channels
- Files with channel="1" are processed by `process_dapi`
- Files with channel="2" are processed by `process_gfp`

Saving and Loading Pipelines
--------------------------

While EZStitcher doesn't have built-in functions for saving and loading pipelines, you can easily save your pipeline configurations as Python scripts.

Saving a Pipeline as a Script
^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a Python script with your pipeline configuration:

.. code-block:: python

    # save_pipeline.py
    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
    from pathlib import Path

    def create_basic_pipeline(plate_path, num_workers=1):
        """Create a basic processing pipeline."""
        # Create configuration
        config = PipelineConfig(
            num_workers=num_workers
        )

        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            config=config,
            plate_path=plate_path
        )

        # Set up directory structure
        dirs = orchestrator.setup_directories()

        # Create pipeline
        pipeline = Pipeline(
            steps=[
                # Step 1: Normalize images
                Step(
                    name="Normalize Images",
                    func=IP.stack_percentile_normalize,
                    variable_components=['channel'],
                    input_dir=dirs['input'],
                    output_dir=dirs['processed']
                ),

                # Step 2: Generate positions
                PositionGenerationStep(
                    name="Generate Positions",
                    input_dir=dirs['processed'],
                    output_dir=dirs['positions']
                ),

                # Step 3: Stitch images
                ImageStitchingStep(
                    name="Stitch Images",
                    input_dir=dirs['processed'],
                    positions_dir=dirs['positions'],
                    output_dir=dirs['stitched']
                )
            ],
            name="Basic Processing Pipeline"
        )

        return orchestrator, pipeline, dirs

    if __name__ == "__main__":
        # Example usage
        plate_path = Path("/path/to/your/plate")
        orchestrator, pipeline, dirs = create_basic_pipeline(plate_path, num_workers=4)

        # Run the pipeline
        success = orchestrator.run(pipelines=[pipeline])

        if success:
            print("Pipeline completed successfully!")
            print(f"Stitched images are in: {dirs['stitched']}")
        else:
            print("Pipeline failed. Check logs for details.")

Loading and Using a Saved Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Import and use the saved pipeline in another script:

.. code-block:: python

    # use_pipeline.py
    from pathlib import Path
    from save_pipeline import create_basic_pipeline

    # Path to your plate folder
    plate_path = Path("/path/to/your/plate")

    # Create the pipeline
    orchestrator, pipeline, dirs = create_basic_pipeline(
        plate_path=plate_path,
        num_workers=4
    )

    # Run the pipeline
    success = orchestrator.run(pipelines=[pipeline])

    if success:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed. Check logs for details.")

Best Practices for Pipeline Scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Parameterize your pipelines**: Make key parameters configurable
2. **Use functions to create pipelines**: Encapsulate pipeline creation in functions
3. **Document your pipelines**: Add comments explaining the purpose of each step
4. **Organize by experiment type**: Create separate scripts for different experiment types
5. **Version control your scripts**: Keep track of changes to your pipeline configurations

Next Steps
---------

Now that you understand the basics of creating and running pipelines, you can:

* Learn about more advanced topics in the :doc:`intermediate_usage` section
* Explore Z-stack processing and best focus detection
* Customize your pipelines with channel-specific processing
* Create more complex workflows with multiple pipelines
