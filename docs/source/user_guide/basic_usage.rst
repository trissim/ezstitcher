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
    from ezstitcher.core.image_processor import ImageProcessor as IP
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

    # The orchestrator automatically manages directories
    # Directories are created as needed during pipeline execution

Now, create a pipeline with three steps. EZStitcher's dynamic directory resolution automatically manages the flow of data between steps:

.. code-block:: python

    # Create a pipeline
    pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,    # Pipeline input directory (ImageLocator finds actual image directory)
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched", # Pipeline output directory
        steps=[
            # Step 1: Normalize image intensities
            # Only specify directories for the first step if needed
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize,
                output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_processed"  # Intermediate output directory
            ),

            # Step 2: Generate positions for stitching
            # No need to specify directories - automatically uses previous step's output
            PositionGenerationStep(
                name="Generate Positions"
            ),

            # Step 3: Stitch images
            # No need to specify directories - automatically uses previous step's output
            # and the pipeline's output directory
            ImageStitchingStep(
                name="Stitch Images"
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
        print(f"Stitched images are in: {orchestrator.plate_path.parent / f'{orchestrator.plate_path.name}_stitched'}")
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

Dynamic Directory Resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

EZStitcher features a powerful dynamic directory resolution system that automatically manages the flow of data between pipeline steps:

1. **Pipeline-Level Directories**: You can set input and output directories at the pipeline level
2. **Step-Level Directories**: You can override directories for specific steps when needed
3. **Automatic Resolution**: If directories aren't specified, they're automatically resolved based on the pipeline structure

Here's how directory resolution works:

* If a step doesn't specify an input directory:
  - For the first step, it uses the pipeline's input directory
  - For subsequent steps, it uses the previous step's output directory

* If a step doesn't specify an output directory:
  - It uses the pipeline's output directory (if specified)
  - Otherwise, it uses the step's input directory

* If a step specifies an input directory:
  - The previous step's output directory is updated to match, ensuring coherent data flow

* Specialized steps like `PositionGenerationStep` and `ImageStitchingStep` have additional logic:
  - `PositionGenerationStep` automatically creates a positions directory if needed
  - `ImageStitchingStep` automatically finds the positions directory if not specified

This system ensures that data flows coherently through the pipeline, with each step's output feeding into the next step's input.

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

EZStitcher provides a variety of image processing functions through the ImageProcessor class. Here are some common operations:

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
    from ezstitcher.core.image_processor import ImageProcessor as IP
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

        # The orchestrator automatically manages directories
        # Directories are created as needed during pipeline execution

        # Create pipeline with dynamic directory resolution
        pipeline = Pipeline(
            input_dir=orchestrator.workspace_path,     # Pipeline input directory (ImageLocator finds actual image directory)
            output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched", # Pipeline output directory
            steps=[
                # Step 1: Normalize images
                Step(
                    name="Normalize Images",
                    func=IP.stack_percentile_normalize,
                    output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_processed"  # Intermediate output directory
                ),

                # Step 2: Generate positions
                # No need to specify directories - automatically uses previous step's output
                PositionGenerationStep(
                    name="Generate Positions"
                ),

                # Step 3: Stitch images
                # No need to specify directories - automatically uses previous step's output
                # and the pipeline's output directory
                ImageStitchingStep(
                    name="Stitch Images"
                )
            ],
            name="Basic Processing Pipeline"
        )

        return orchestrator, pipeline

    if __name__ == "__main__":
        # Example usage
        plate_path = Path("/path/to/your/plate")
        orchestrator, pipeline = create_basic_pipeline(plate_path, num_workers=4)

        # Run the pipeline
        success = orchestrator.run(pipelines=[pipeline])

        if success:
            print("Pipeline completed successfully!")
            print(f"Stitched images are in: {orchestrator.plate_path.parent / f'{orchestrator.plate_path.name}_stitched'}")
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
    orchestrator, pipeline = create_basic_pipeline(
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
4. **Leverage dynamic directory resolution**: Set directories at the pipeline level and only override when necessary
5. **Use coherent data flow**: Let each step's output feed into the next step's input
6. **Organize by experiment type**: Create separate scripts for different experiment types
7. **Version control your scripts**: Keep track of changes to your pipeline configurations

Next Steps
---------

Now that you understand the basics of creating and running pipelines, you can:

* Learn about more advanced topics in the :doc:`intermediate_usage` section
* Explore Z-stack processing and best focus detection
* Customize your pipelines with channel-specific processing
* Create more complex workflows with multiple pipelines
