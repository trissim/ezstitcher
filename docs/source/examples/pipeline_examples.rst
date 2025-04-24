Pipeline Examples
================

This page provides examples of using EZStitcher's pipeline architecture for common microscopy image processing tasks.

Basic Pipeline Example
--------------------

Here's a basic example of creating and running a pipeline:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP

    # Create configuration with reference channels and worker settings
    # - reference_channels: Channels to use for position generation
    # - num_workers: Number of worker threads for parallel processing
    config = PipelineConfig(
        reference_channels=["1"],  # Use channel 1 as reference
        num_workers=1              # Single-threaded for simplicity
    )

    # Create orchestrator with configuration and plate path
    # The orchestrator manages the execution of pipelines across wells
    orchestrator = PipelineOrchestrator(config=config, plate_path="path/to/plate")

    # Set up directory structure for processing
    # This creates standard directories for input, processed, positions, etc.
    dirs = orchestrator.setup_directories()

    # Create a simple pipeline with two processing steps
    pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks using maximum projection
            # - variable_components=['z_index']: Process each z-index separately
            # - processing_args={'method': 'max_projection'}: Use maximum intensity projection
            Step(name="Z-Stack Flattening",
                 func=IP.create_projection,
                 variable_components=['z_index'],
                 processing_args={'method': 'max_projection'},
                 input_dir=dirs['input'],
                 output_dir=dirs['processed']),

            # Step 2: Normalize channel intensities
            # - variable_components=['channel']: Process each channel separately
            # - group_by='channel': Group files by channel for processing
            Step(name="Channel Processing",
                 func=IP.stack_percentile_normalize,
                 variable_components=['channel'],
                 group_by='channel'
            )
        ],
        name="Simple Processing Pipeline"
    )

    # Run the orchestrator with the pipeline
    # This processes all wells in the plate using the pipeline
    success = orchestrator.run(pipelines=[pipeline])

Position Generation and Stitching Example
---------------------------------------

Here's an example of creating pipelines for position generation and image stitching:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
    from ezstitcher.core.utils import stack

    # Create configuration for position generation and stitching
    # - reference_channels: Channels to use for position generation
    # - num_workers: Number of worker threads for parallel processing
    config = PipelineConfig(
        reference_channels=["1"],  # Use channel 1 as reference
        num_workers=2              # Use 2 worker threads for parallel processing
    )

    # Create orchestrator to manage the pipelines
    orchestrator = PipelineOrchestrator(config=config, plate_path="path/to/plate")

    # Set up directory structure for processing
    # This creates standard directories for input, processed, positions, stitched, etc.
    dirs = orchestrator.setup_directories()

    # Create position generation pipeline
    # This pipeline processes images and generates position files for stitching
    position_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks using maximum projection
            # This reduces 3D z-stacks to 2D images for position calculation
            Step(name="Z-Stack Flattening",
                 func=IP.create_projection,
                 variable_components=['z_index'],  # Process each z-index separately
                 processing_args={'method': 'max_projection'},  # Use maximum intensity projection
                 input_dir=dirs['input'],
                 output_dir=dirs['processed']),

            # Step 2: Enhance images for better feature detection
            # Apply sharpening followed by normalization to improve contrast
            Step(name="Image Enhancement",
                 func=[stack(IP.sharpen),           # First sharpen the images
                      IP.stack_percentile_normalize],  # Then normalize intensity
            ),

            # Step 3: Generate position files for stitching
            # This specialized step calculates the relative positions of tiles
            PositionGenerationStep(
                name="Generate Positions",
                output_dir=dirs['positions']  # Save position files here
            )
        ],
        name="Position Generation Pipeline"
    )

    # Create image assembly pipeline
    # This pipeline processes and stitches images using the generated positions
    assembly_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks for final images
            # This reduces 3D z-stacks to 2D images for final stitching
            Step(name="Z-Stack Flattening",
                 func=IP.create_projection,
                 variable_components=['z_index'],  # Process each z-index separately
                 processing_args={'method': 'max_projection'},  # Use maximum intensity projection
                 input_dir=dirs['input'],
                 output_dir=dirs['post_processed']
            ),

            # Step 2: Normalize channel intensities
            # This improves image quality and consistency across tiles
            Step(name="Channel Processing",
                 func=IP.stack_percentile_normalize,  # Normalize intensity based on percentiles
            ),

            # Step 3: Stitch images using the generated positions
            # This specialized step combines tiles into a single large image
            ImageStitchingStep(
                name="Stitch Images",
                positions_dir=dirs['positions'],  # Use position files from here
                output_dir=dirs['stitched']       # Save stitched images here
            )
        ],
        name="Image Assembly Pipeline"
    )

    # Run the orchestrator with both pipelines
    # The pipelines will be executed sequentially for each well
    success = orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

Z-Stack Processing with Best Focus
--------------------------------

Here's an example of processing Z-stacks with best focus detection:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
    from ezstitcher.core.utils import stack

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1"],
        num_workers=2
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(config=config, plate_path="path/to/plate")

    # Get directories
    dirs = orchestrator.setup_directories()

    # Create position generation pipeline
    position_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks
            Step(name="Z-Stack Flattening",
                 func=IP.create_projection,
                 variable_components=['z_index'],
                 processing_args={'method': 'max_projection'},
                 input_dir=dirs['input'],
                 output_dir=dirs['processed']),

            # Step 2: Process channels
            Step(name="Feature Enhancement",
                 func=stack(IP.sharpen),
                 variable_components=['site']),

            # Step 3: Generate positions
            PositionGenerationStep(
                name="Generate Positions",
                output_dir=dirs['positions']
            )
        ],
        name="Position Generation Pipeline"
    )

    # Create best focus pipeline
    focus_pipeline = Pipeline(
        steps=[
            # Step 1: Clean images for focus detection
            Step(name="Cleaning",
                 func=[IP.tophat],
                 input_dir=dirs['input'],
                 output_dir=dirs['focus']),

            # Step 2: Apply best focus
            Step(name="Focus",
                 func=IP.create_projection,
                 variable_components=['z_index'],
                 processing_args={'method': 'best_focus'}),

            # Step 3: Stitch focused images
            ImageStitchingStep(
                name="Stitch Focused Images",
                positions_dir=dirs['positions'],
                output_dir=dirs['stitched']),
        ],
        name="Focused Image Assembly Pipeline"
    )

    # Run the orchestrator with the pipelines
    success = orchestrator.run(pipelines=[position_pipeline, focus_pipeline])

Channel-Specific Processing
-------------------------

Here's an example of applying different processing functions to different channels:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
    from ezstitcher.core.utils import stack

    # Define channel-specific processing functions
    def process_dapi(stack):
        """Process DAPI channel images."""
        stack = IP.stack_percentile_normalize(stack, low_percentile=0.1, high_percentile=99.9)
        return [IP.tophat(img) for img in stack]

    def process_calcein(stack):
        """Process Calcein channel images."""
        return [IP.tophat(img) for img in stack]

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1", "2"],
        num_workers=2
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(config=config, plate_path="path/to/plate")

    # Get directories
    dirs = orchestrator.setup_directories()

    # Create pipeline with channel-specific processing
    pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks
            Step(name="Z-Stack Flattening",
                 func=IP.create_projection,
                 variable_components=['z_index'],
                 processing_args={'method': 'max_projection'},
                 input_dir=dirs['input'],
                 output_dir=dirs['processed']),

            # Step 2: Channel-specific processing
            Step(name="Channel Processing",
                 func={"1": process_dapi, "2": process_calcein},  # Dictionary mapping channels to functions
                 variable_components=['channel'],
                 group_by='channel'  # Group by channel for channel-specific processing
            )
        ],
        name="Channel-Specific Processing Pipeline"
    )

    # Run the orchestrator with the pipeline
    success = orchestrator.run(pipelines=[pipeline])

More Examples
-----------

For more examples, see the integration tests in the ``tests/integration`` directory, particularly:

- ``test_pipeline_architecture``: Basic pipeline architecture example
- ``test_zstack_pipeline_architecture``: Z-stack processing example
- ``test_zstack_pipeline_architecture_focus``: Z-stack processing with best focus example
