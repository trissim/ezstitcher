=================
Intermediate Usage
=================

This section covers more advanced topics in EZStitcher, building on the basic concepts and usage patterns introduced earlier.

Z-Stack Processing
----------------

Z-stacks are 3D image stacks where each image represents a different focal plane. EZStitcher provides several methods for processing Z-stacks.

Z-Stack Flattening
^^^^^^^^^^^^^^^

One common operation is to flatten a Z-stack into a single 2D image using a projection method:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
    from pathlib import Path

    # Create configuration and orchestrator
    config = PipelineConfig(num_workers=1)
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path=Path("/path/to/plate")
    )
    dirs = orchestrator.setup_directories()

    # Create a pipeline for Z-stack flattening
    z_flatten_pipeline = Pipeline(
        steps=[
            # Z-stack flattening step
            Step(
                name="Z-Stack Flattening",
                func=IP.create_projection,
                variable_components=['z_index'],  # Process each z-index separately
                group_by='site',  # Group by site to combine z-planes for each site
                processing_args={'method': 'max_projection'},  # Use maximum intensity projection
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            )
        ],
        name="Z-Stack Flattening Pipeline"
    )

    # Run the pipeline
    orchestrator.run(pipelines=[z_flatten_pipeline])

Projection Methods
^^^^^^^^^^^^^^^

EZStitcher supports several projection methods:

1. **Maximum Intensity Projection (max_projection)**: Takes the maximum value at each pixel position across all Z-planes
2. **Minimum Intensity Projection (min_projection)**: Takes the minimum value at each pixel position
3. **Mean Intensity Projection (mean_projection)**: Takes the average value at each pixel position
4. **Standard Deviation Projection (std_projection)**: Shows the standard deviation at each pixel position
5. **Sum Projection (sum_projection)**: Sums the values at each pixel position

Example with different projection methods:

.. code-block:: python

    # Create a pipeline with different projection methods
    multi_projection_pipeline = Pipeline(
        steps=[
            # Maximum intensity projection
            Step(
                name="Max Projection",
                func=IP.create_projection,
                variable_components=['z_index'],
                group_by='site',
                processing_args={'method': 'max_projection'},
                input_dir=dirs['input'],
                output_dir=Path("path/to/max_projection")
            ),

            # Mean intensity projection
            Step(
                name="Mean Projection",
                func=IP.create_projection,
                variable_components=['z_index'],
                group_by='site',
                processing_args={'method': 'mean_projection'},
                input_dir=dirs['input'],
                output_dir=Path("path/to/mean_projection")
            )
        ],
        name="Multi-Projection Pipeline"
    )

Best Focus Detection
^^^^^^^^^^^^^^^^^

Instead of using a projection method, you can select the best-focused plane from a Z-stack:

.. code-block:: python

    # Create a pipeline for best focus detection
    best_focus_pipeline = Pipeline(
        steps=[
            # Best focus detection step
            Step(
                name="Best Focus Detection",
                func=IP.find_best_focus,
                variable_components=['z_index'],
                group_by='site',
                processing_args={'metric': 'variance_of_laplacian'},
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            )
        ],
        name="Best Focus Pipeline"
    )

Focus Metrics
^^^^^^^^^^^

EZStitcher supports several focus metrics:

1. **Variance of Laplacian (variance_of_laplacian)**: Measures local variations in intensity
2. **Normalized Variance (normalized_variance)**: Measures the variance normalized by the mean intensity
3. **Tenengrad (tenengrad)**: Uses the Sobel operator to measure gradient magnitude
4. **Brenner Gradient (brenner_gradient)**: Measures the sum of squared differences between adjacent pixels

Example with different focus metrics:

.. code-block:: python

    from ezstitcher.core.focus_analyzer import FocusAnalyzer

    # Create a pipeline with different focus metrics
    focus_metrics_pipeline = Pipeline(
        steps=[
            # Variance of Laplacian metric
            Step(
                name="Variance of Laplacian",
                func=IP.find_best_focus,
                variable_components=['z_index'],
                group_by='site',
                processing_args={'metric': 'variance_of_laplacian'},
                input_dir=dirs['input'],
                output_dir=Path("path/to/laplacian_focus")
            ),

            # Tenengrad metric
            Step(
                name="Tenengrad",
                func=IP.find_best_focus,
                variable_components=['z_index'],
                group_by='site',
                processing_args={'metric': 'tenengrad'},
                input_dir=dirs['input'],
                output_dir=Path("path/to/tenengrad_focus")
            )
        ],
        name="Focus Metrics Pipeline"
    )

    # You can also use the FocusAnalyzer directly for more control
    focus_analyzer = FocusAnalyzer()
    focus_scores = focus_analyzer.calculate_focus_scores(
        images,  # List of images in a Z-stack
        metric='variance_of_laplacian'
    )
    best_focus_index = focus_analyzer.find_best_focus_index(focus_scores)
    best_focused_image = images[best_focus_index]

Channel-Specific Processing
-------------------------

Different fluorescence channels often require different processing approaches. EZStitcher provides several ways to apply channel-specific processing.

Using Dictionary of Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

The most flexible approach is to use a dictionary of functions, where each key corresponds to a channel:

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

    # Create a pipeline with channel-specific processing
    channel_specific_pipeline = Pipeline(
        steps=[
            # Channel-specific processing step
            Step(
                name="Channel-Specific Processing",
                func={
                    "1": process_dapi,  # Apply process_dapi to channel 1 (DAPI)
                    "2": process_gfp    # Apply process_gfp to channel 2 (GFP)
                },
                group_by='channel',  # Specifies that keys "1" and "2" refer to channel values
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            )
        ],
        name="Channel-Specific Pipeline"
    )

Advanced Channel-Specific Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also use a dictionary of lists of functions with matching processing arguments:

.. code-block:: python

    from ezstitcher.core.utils import stack

    # Create a pipeline with advanced channel-specific processing
    advanced_channel_pipeline = Pipeline(
        steps=[
            # Advanced channel-specific processing step
            Step(
                name="Advanced Channel Processing",
                func={
                    "1": [  # Process channel 1 (DAPI)
                        stack(IP.tophat),             # First apply tophat
                        IP.stack_percentile_normalize  # Then normalize
                    ],
                    "2": [  # Process channel 2 (GFP)
                        stack(IP.sharpen),            # First apply sharpen
                        IP.stack_percentile_normalize  # Then normalize
                    ]
                },
                group_by='channel',  # Specifies that keys "1" and "2" refer to channel values
                processing_args={
                    "1": [
                        {'size': 15},                  # Args for tophat
                        {'low_percentile': 1.0, 'high_percentile': 99.0}  # Args for normalize
                    ],
                    "2": [
                        {'sigma': 1.0, 'amount': 1.5},  # Args for sharpen
                        {'low_percentile': 1.0, 'high_percentile': 99.0}  # Args for normalize
                    ]
                },
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            )
        ],
        name="Advanced Channel Pipeline"
    )

Creating Composite Images
^^^^^^^^^^^^^^^^^^^^^^

You can combine multiple channels into a composite image:

.. code-block:: python

    # Create a pipeline for creating composite images
    composite_pipeline = Pipeline(
        steps=[
            # Process individual channels first
            Step(
                name="Channel Processing",
                func=IP.stack_percentile_normalize,
                variable_components=['channel'],
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            ),

            # Create composite images
            Step(
                name="Create Composite",
                func=IP.create_composite,
                variable_components=['channel'],  # Process each channel separately
                group_by='site',  # Group by site to combine channels for each site
                input_dir=dirs['processed'],
                output_dir=dirs['composite']
            )
        ],
        name="Composite Image Pipeline"
    )

Position Generation and Stitching
-------------------------------

EZStitcher provides specialized steps for generating position files and stitching images.

Basic Stitching Workflow
^^^^^^^^^^^^^^^^^^^^^

A typical stitching workflow involves two main steps:

1. Generate position files that describe how the tiles fit together
2. Stitch the images using these position files

.. code-block:: python

    from ezstitcher.core.steps import PositionGenerationStep, ImageStitchingStep

    # Create a pipeline for stitching
    stitching_pipeline = Pipeline(
        steps=[
            # Step 1: Process images (optional)
            Step(
                name="Image Processing",
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
        name="Stitching Pipeline"
    )

Advanced Position Generation
^^^^^^^^^^^^^^^^^^^^^^^^^

You can customize the position generation process:

.. code-block:: python

    # Create a pipeline with customized position generation
    advanced_position_pipeline = Pipeline(
        steps=[
            # Generate positions with custom parameters
            PositionGenerationStep(
                name="Advanced Position Generation",
                input_dir=dirs['processed'],
                output_dir=dirs['positions'],
                overlap_percent=20,  # Expected overlap percentage
                max_shift_percent=5,  # Maximum allowed shift as percentage of image size
                use_phase_correlation=True,  # Use phase correlation for alignment
                reference_channel="1"  # Use channel 1 as reference for alignment
            )
        ],
        name="Advanced Position Pipeline"
    )

Advanced Stitching
^^^^^^^^^^^^^^^

You can also customize the stitching process:

.. code-block:: python

    # Create a pipeline with customized stitching
    advanced_stitching_pipeline = Pipeline(
        steps=[
            # Stitch images with custom parameters
            ImageStitchingStep(
                name="Advanced Stitching",
                input_dir=dirs['processed'],
                positions_dir=dirs['positions'],
                output_dir=dirs['stitched'],
                blend_method="linear",  # Use linear blending at overlaps
                normalize_intensities=True,  # Normalize intensities across tiles
                background_subtraction=True  # Perform background subtraction
            )
        ],
        name="Advanced Stitching Pipeline"
    )

Combining Multiple Techniques
---------------------------

EZStitcher's pipeline architecture allows you to combine multiple techniques in a single workflow.

Z-Stack Processing and Stitching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Process Z-stacks and then stitch the resulting images:

.. code-block:: python

    # Create a pipeline that combines Z-stack processing and stitching
    z_stack_stitching_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks
            Step(
                name="Z-Stack Flattening",
                func=IP.create_projection,
                variable_components=['z_index'],
                group_by='site',
                processing_args={'method': 'max_projection'},
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
        name="Z-Stack Stitching Pipeline"
    )

Channel-Specific Processing and Stitching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply different processing to different channels and then stitch the results:

.. code-block:: python

    # Create a pipeline that combines channel-specific processing and stitching
    channel_stitching_pipeline = Pipeline(
        steps=[
            # Step 1: Channel-specific processing
            Step(
                name="Channel-Specific Processing",
                func={
                    "1": process_dapi,
                    "2": process_gfp
                },
                group_by='channel',
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            ),

            # Step 2: Generate positions
            PositionGenerationStep(
                name="Generate Positions",
                input_dir=dirs['processed'],
                output_dir=dirs['positions'],
                reference_channel="1"  # Use channel 1 as reference for alignment
            ),

            # Step 3: Stitch images
            ImageStitchingStep(
                name="Stitch Images",
                input_dir=dirs['processed'],
                positions_dir=dirs['positions'],
                output_dir=dirs['stitched']
            )
        ],
        name="Channel Stitching Pipeline"
    )

Complete Workflow Example
^^^^^^^^^^^^^^^^^^^^^^

A complete workflow that combines Z-stack processing, channel-specific processing, and stitching:

.. code-block:: python

    # Create a complete workflow pipeline
    complete_workflow_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks with channel-specific processing
            Step(
                name="Z-Stack Processing",
                func={
                    "1": IP.create_projection,  # Use max projection for channel 1
                    "2": IP.find_best_focus     # Use best focus for channel 2
                },
                group_by='channel',
                variable_components=['z_index'],
                processing_args={
                    "1": {'method': 'max_projection'},
                    "2": {'metric': 'variance_of_laplacian'}
                },
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            ),

            # Step 2: Channel-specific enhancement
            Step(
                name="Channel Enhancement",
                func={
                    "1": stack(IP.tophat),
                    "2": stack(IP.sharpen)
                },
                group_by='channel',
                processing_args={
                    "1": {'size': 15},
                    "2": {'sigma': 1.0, 'amount': 1.5}
                },
                output_dir=dirs['enhanced']
            ),

            # Step 3: Generate positions
            PositionGenerationStep(
                name="Generate Positions",
                input_dir=dirs['enhanced'],
                output_dir=dirs['positions'],
                reference_channel="1"
            ),

            # Step 4: Stitch images
            ImageStitchingStep(
                name="Stitch Images",
                input_dir=dirs['enhanced'],
                positions_dir=dirs['positions'],
                output_dir=dirs['stitched']
            )
        ],
        name="Complete Workflow Pipeline"
    )

Next Steps
---------

Now that you understand intermediate usage patterns, you can:

* Explore advanced usage in the :doc:`advanced_usage` section
* Learn about custom processing functions and multithreaded processing
* See complete workflow examples in the :doc:`practical_examples` section
