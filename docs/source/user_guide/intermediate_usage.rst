=================
Intermediate Usage
=================

This section covers more advanced topics in EZStitcher, building on the basic concepts and usage patterns introduced earlier.

.. note::
   Directory paths are automatically resolved between steps in EZStitcher. The first step should specify
   ``input_dir=orchestrator.workspace_path`` to ensure processing happens on workspace copies,
   but subsequent steps will automatically use the output of the previous step as their input.
   See :doc:`../concepts/directory_structure` for details on how EZStitcher manages directories.

.. important::
   Understanding the relationship between ``variable_components`` and ``group_by`` parameters is crucial for
   correctly configuring pipeline steps. For detailed explanations of these parameters and their relationships,
   see :doc:`../concepts/step`.

Z-Stack Processing
----------------

Z-stacks are 3D image stacks where each image represents a different focal plane. EZStitcher provides several methods for processing Z-stacks. For detailed explanations of Z-stack processing and the `variable_components` parameter, see :ref:`variable-components` in the :doc:`../concepts/step` documentation. For a comprehensive guide to all Z-stack processing operations, see :ref:`operation-z-projection` in the :doc:`../api/image_processing_operations` documentation.

.. important::
   Z-stack flattening is a one-way operation that converts a 3D stack into a single 2D image. Once a Z-stack is flattened, it cannot be flattened again using a different method. You should choose the most appropriate flattening method for your data based on your specific needs.

Z-Stack Flattening
^^^^^^^^^^^^^^^

One common operation is to flatten a Z-stack into a single 2D image using a projection method:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_processor import ImageProcessor as IP
    from pathlib import Path

    # Create configuration and orchestrator
    config = PipelineConfig(num_workers=1)
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path=Path("/path/to/plate")
    )

    # Create a pipeline for Z-stack flattening
    z_flatten_pipeline = Pipeline(
        steps=[
            # Z-stack flattening step
            Step(
                name="Z-Stack Flattening",
                func=(IP.create_projection, {'method': 'max_projection'}),  # Function with projection method
                variable_components=['z_index'],  # Process each z-index separately
                input_dir=orchestrator.workspace_path
            )
        ],
        name="Z-Stack Flattening Pipeline"
    )

    # Run the pipeline
    orchestrator.run(pipelines=[z_flatten_pipeline])

Projection Methods
^^^^^^^^^^^^^^^

EZStitcher supports several alternative projection methods for flattening Z-stacks. You should choose the most appropriate method for your specific data:

1. **Maximum Intensity Projection (max_projection)**: Takes the maximum value at each pixel position across all Z-planes
2. **Mean Intensity Projection (mean_projection)**: Takes the average value at each pixel position
3. **Best Focus (best_focus)**: Selects the best-focused plane using focus metrics

Example with different projection methods:

.. code-block:: python

    # Create separate pipelines for different projection methods
    # Note: You would typically choose ONE method, not run multiple in sequence

    # Maximum intensity projection pipeline
    max_projection_pipeline = Pipeline(
        steps=[
            Step(
                name="Max Projection",
                func=(IP.create_projection, {'method': 'max_projection'}),
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path,
                output_dir=Path("path/to/max_projection")
            )
        ],
        name="Max Projection Pipeline"
    )

    # Mean intensity projection pipeline
    mean_projection_pipeline = Pipeline(
        steps=[
            Step(
                name="Mean Projection",
                func=(IP.create_projection, {'method': 'mean_projection'}),
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path,
                output_dir=Path("path/to/mean_projection")
            )
        ],
        name="Mean Projection Pipeline"
    )

    # Best focus pipeline (requires a focus analyzer)
    from ezstitcher.core.focus_analyzer import FocusAnalyzer

    focus_analyzer = FocusAnalyzer(metric='variance_of_laplacian')
    best_focus_pipeline = Pipeline(
        steps=[
            Step(
                name="Best Focus",
                func=(IP.create_projection, {'method': 'best_focus', 'focus_analyzer': focus_analyzer}),
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path,
                output_dir=Path("path/to/best_focus")
            )
        ],
        name="Best Focus Pipeline"
    )

    # Run only one of these pipelines
    # orchestrator.run(pipelines=[max_projection_pipeline])
    # orchestrator.run(pipelines=[mean_projection_pipeline])
    # orchestrator.run(pipelines=[best_focus_pipeline])

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
                func=(IP.find_best_focus, {'metric': 'variance_of_laplacian'}),
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path
            )
        ],
        name="Best Focus Pipeline"
    )

Focus Metrics
^^^^^^^^^^^

EZStitcher supports several alternative focus metrics for finding the best-focused plane. You should choose the most appropriate metric for your specific data:

1. **Variance of Laplacian (variance_of_laplacian)**: Measures local variations in intensity
2. **Normalized Variance (normalized_variance)**: Measures the variance normalized by the mean intensity
3. **Tenengrad (tenengrad)**: Uses the Sobel operator to measure gradient magnitude
4. **Brenner Gradient (brenner_gradient)**: Measures the sum of squared differences between adjacent pixels

Example with different focus metrics:

.. code-block:: python

    from ezstitcher.core.focus_analyzer import FocusAnalyzer

    # Create separate pipelines for different focus metrics
    # Note: You would typically choose ONE metric, not run multiple in sequence

    # Variance of Laplacian metric pipeline
    laplacian_pipeline = Pipeline(
        steps=[
            Step(
                name="Variance of Laplacian",
                func=(IP.find_best_focus, {'metric': 'variance_of_laplacian'}),
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path,
                output_dir=Path("path/to/laplacian_focus")
            )
        ],
        name="Laplacian Focus Pipeline"
    )

    # Tenengrad metric pipeline
    tenengrad_pipeline = Pipeline(
        steps=[
            Step(
                name="Tenengrad",
                func=(IP.find_best_focus, {'metric': 'tenengrad'}),
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path,
                output_dir=Path("path/to/tenengrad_focus")
            )
        ],
        name="Tenengrad Focus Pipeline"
    )

    # Run the pipelines separately
    # orchestrator.run(pipelines=[laplacian_pipeline])
    # orchestrator.run(pipelines=[tenengrad_pipeline])

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

For detailed explanations of function handling patterns, including dictionaries of functions, see :doc:`../concepts/function_handling`. For a comprehensive guide to all multi-channel operations, see :ref:`operation-composite` in the :doc:`../api/image_processing_operations` documentation.

Using Dictionary of Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

The most flexible approach is to use a dictionary of functions, where each key corresponds to a channel. For detailed explanations of the `group_by` parameter and how it works with dictionaries of functions, see :ref:`group-by` in the :doc:`../concepts/step` documentation.

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
                input_dir=orchestrator.workspace_path

            )
        ],
        name="Channel-Specific Pipeline"
    )

Advanced Channel-Specific Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also use a dictionary of lists of functions with matching processing arguments. For detailed explanations of this pattern, see :doc:`../concepts/function_handling`.

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
                        (stack(IP.tophat), {'size': 15}),  # First apply tophat with args
                        (IP.stack_percentile_normalize, {'low_percentile': 1.0, 'high_percentile': 99.0})  # Then normalize with args
                    ],
                    "2": [  # Process channel 2 (GFP)
                        (stack(IP.sharpen), {'sigma': 1.0, 'amount': 1.5}),  # First apply sharpen with args
                        (IP.stack_percentile_normalize, {'low_percentile': 1.0, 'high_percentile': 99.0})  # Then normalize with args
                    ]
                },
                group_by='channel',  # Specifies that keys "1" and "2" refer to channel values
                input_dir=orchestrator.workspace_path

            )
        ],
        name="Advanced Channel Pipeline"
    )

Creating Composite Images
^^^^^^^^^^^^^^^^^^^^^^

You can combine multiple channels into a composite image. For detailed explanations of composite image creation and the `variable_components=['channel']` parameter, see :ref:`variable-components` in the :doc:`../concepts/step` documentation.

.. note::
   The `create_composite` function can be called with or without the `weights` parameter:

   * Without weights: `func=IP.create_composite` - All channels are weighted equally
   * With weights: `func=(IP.create_composite, {'weights': [0.7, 0.3]})` - Custom weighting for each channel

   The weights list should have the same length as the number of channels being processed.

.. code-block:: python

    # Create a pipeline for creating composite images
    composite_pipeline = Pipeline(
        steps=[
            # Process individual channels first
            Step(
                name="Channel Processing",
                func=IP.stack_percentile_normalize,
                variable_components=['channel'],
                input_dir=orchestrator.workspace_path

            ),

            # Create composite images
            Step(
                func=IP.create_composite,
                variable_components=['channel'],  # Process each channel separately
                output_dir=Path("path/to/composite")
            )
        ],
        name="Composite Image Pipeline"
    )

Position Generation and Stitching
-------------------------------

EZStitcher provides specialized steps for generating position files and stitching images. For detailed explanations of these specialized steps, see :ref:`specialized-steps`. For information about position file formats, see :ref:`position-files`.

.. important::
   When working with multiple channels, always create a composite image before position generation.
   This ensures that position files are generated based on all available information rather than
   defaulting to a single channel, which may not have the best features for alignment.

For typical stitching workflows, including basic stitching, multi-channel stitching, and using original images for stitching, see :ref:`typical-stitching-workflows`.


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
                func=(IP.create_projection, {'method': 'max_projection'}),
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path
            ),

            # Step 2: Process channels (if multiple channels exist)
            Step(
                name="Channel Processing",
                func=IP.stack_percentile_normalize,
                variable_components=['channel']
            ),

            # This is important when working with multiple channels
            Step(
                func=IP.create_composite,  # Equal weighting for all channels
                variable_components=['channel']
            ),

            PositionGenerationStep(),

            # By default, uses previous step's output directory (position files)
            ImageStitchingStep(
                # input_dir=orchestrator.workspace_path  # Uncomment to use original images for stitching
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
                input_dir=orchestrator.workspace_path
            ),

            # This is important when working with multiple channels
            Step(
                func=(IP.create_composite, {'weights': [0.7, 0.3]}),  # Custom weighting: 70% channel 1, 30% channel 2
                variable_components=['channel']
            ),

            PositionGenerationStep(),

            # By default, uses previous step's output directory (position files)
            ImageStitchingStep(
                # input_dir=orchestrator.workspace_path  # Uncomment to use original images for stitching
            )
        ],
        name="Channel Stitching Pipeline"
    )

Complete Workflow Example
^^^^^^^^^^^^^^^^^^^^^^

A complete workflow that combines Z-stack processing, channel-specific processing, and stitching:

.. code-block:: python

    from ezstitcher.core.focus_analyzer import FocusAnalyzer

    # Create a complete workflow pipeline
    complete_workflow_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks with channel-specific processing
            Step(
                name="Z-Stack Processing",
                func={
                    "1": (IP.create_projection, {'method': 'max_projection'}),  # Use max projection for channel 1
                    "2": (IP.create_projection, {'method': 'best_focus', 'focus_analyzer': FocusAnalyzer(metric='variance_of_laplacian')})  # Use best focus for channel 2
                },
                group_by='channel',
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path
            ),

            # Step 2: Channel-specific enhancement
            Step(
                name="Channel Enhancement",
                func={
                    "1": (stack(IP.tophat), {'size': 15}),
                    "2": (stack(IP.sharpen), {'sigma': 1.0, 'amount': 1.5})
                },
                group_by='channel',
            ),

            # This is important when working with multiple channels
            Step(
                func=(IP.create_composite, {'weights': [0.6, 0.4]}),  # Custom weighting: 60% channel 1, 40% channel 2
                variable_components=['channel']
            ),

            PositionGenerationStep(),

            ImageStitchingStep()
        ],
        name="Complete Workflow Pipeline"
    )

Next Steps
---------

Now that you understand intermediate usage patterns, you're ready to explore advanced topics. For a comprehensive learning path that will guide you through advanced topics and mastering EZStitcher, see :ref:`learning-path` in the introduction.
