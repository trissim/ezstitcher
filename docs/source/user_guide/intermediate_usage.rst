=================
Intermediate Usage
=================

This section covers more advanced topics in EZStitcher, building on the basic concepts and usage patterns introduced earlier.

Introduction
-----------

This guide covers intermediate usage patterns for EZStitcher, building on the basic concepts introduced earlier. This guide shows how to:

1. Use ``AutoPipelineFactory`` for more complex scenarios
2. Build custom pipelines for specialized workflows
3. Customize pipelines for maximum flexibility

Both approaches are valid and powerful, with different strengths depending on your needs. This guide will show you how to use both approaches for intermediate-level tasks.

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

Z-stacks can be processed in various ways, including maximum projection, mean projection, and focus detection.

Using AutoPipelineFactory
^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to process Z-stacks is with ``AutoPipelineFactory``:

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    from pathlib import Path

    # Create orchestrator
    orchestrator = PipelineOrchestrator(plate_path=plate_path)

    # Create a factory for Z-stack processing and stitching
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        flatten_z=True,  # Flatten Z-stacks in the assembly pipeline
        z_method="max"   # Use maximum intensity projection
    )
    pipelines = factory.create_pipelines()

    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

You can change the ``z_method`` parameter to use different projection methods:

- ``"max"``: Maximum intensity projection (default)
- ``"mean"``: Mean intensity projection
- ``"median"``: Median intensity projection
- ``"combined"``: Combined focus metric for focus detection
- ``"laplacian"``: Laplacian focus metric
- ``"tenengrad"``: Tenengrad focus metric
- ``"normalized_variance"``: Normalized variance focus metric
- ``"fft"``: FFT-based focus metric

For focus detection, simply change the z_method:

.. code-block:: python

    # Create a factory with focus detection
    focus_factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        flatten_z=True,
        z_method="combined"  # Use combined focus metric
    )
    focus_pipelines = focus_factory.create_pipelines()

Custom Pipeline Approach
^^^^^^^^^^^^^^^^^^^^^

For maximum flexibility, you can build custom pipelines:

.. code-block:: python

    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.step_factories import ZFlatStep, FocusStep, CompositeStep

    # Create position generation pipeline with maximum projection
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="max"  # Use maximum intensity projection
            ),

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

    # Create image assembly pipeline with maximum projection
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=Path("path/to/max_projection"),
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="max"  # Use maximum intensity projection
            ),

            # Step 2: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Stitch images using position files
            ImageStitchingStep(positions_dir=positions_dir)
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

    # Alternative: Create pipelines with focus detection

    # Create position generation pipeline with focus detection
    focus_position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep (always use max for position generation)
            ZFlatStep(
                method="max"
            ),

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
    focus_positions_dir = focus_position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline with focus detection
    focus_assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=Path("path/to/best_focus"),
        steps=[
            # Step 1: Use FocusStep for best focus selection
            FocusStep(focus_options={'metric': 'variance_of_laplacian'}),

            # Step 2: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Stitch images using position files
            ImageStitchingStep(positions_dir=focus_positions_dir)
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[focus_position_pipeline, focus_assembly_pipeline])

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

    from ezstitcher.core.step_factories import ZFlatStep, FocusStep, CompositeStep

    # Maximum intensity projection

    # Create position generation pipeline with max projection
    max_position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="max"  # Use maximum intensity projection
            ),

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
    max_positions_dir = max_position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline with max projection
    max_assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=Path("path/to/max_projection"),
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="max"  # Use maximum intensity projection
            ),

            # Step 2: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Stitch images using position files
            ImageStitchingStep(positions_dir=max_positions_dir)
        ],
        name="Max Projection Assembly Pipeline"
    )

    # Mean intensity projection

    # Create position generation pipeline with max projection (always use max for position generation)
    mean_position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="max"  # Always use max for position generation
            ),

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
    mean_positions_dir = mean_position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline with mean projection
    mean_assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=Path("path/to/mean_projection"),
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="mean"  # Use mean intensity projection
            ),

            # Step 2: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Stitch images using position files
            ImageStitchingStep(positions_dir=mean_positions_dir)
        ],
        name="Mean Projection Assembly Pipeline"
    )

    # Best focus detection

    # Create position generation pipeline with max projection (always use max for position generation)
    focus_position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="max"  # Always use max for position generation
            ),

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
    focus_positions_dir = focus_position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline with focus detection
    focus_assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=Path("path/to/best_focus"),
        steps=[
            # Step 1: Use FocusStep for best focus selection
            FocusStep(
                focus_options={'metric': 'variance_of_laplacian'}  # Use variance of Laplacian metric
            ),

            # Step 2: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Stitch images using position files
            ImageStitchingStep(positions_dir=focus_positions_dir)
        ],
        name="Best Focus Assembly Pipeline"
    )

    # Run only one set of pipelines
    # orchestrator.run(pipelines=[max_position_pipeline, max_assembly_pipeline])
    # orchestrator.run(pipelines=[mean_position_pipeline, mean_assembly_pipeline])
    # orchestrator.run(pipelines=[focus_position_pipeline, focus_assembly_pipeline])

Best Focus Detection
^^^^^^^^^^^^^^^^^

Instead of using a projection method, you can select the best-focused plane from a Z-stack:

.. code-block:: python

    from ezstitcher.core.step_factories import ZFlatStep, FocusStep, CompositeStep

    # Create position generation pipeline with max projection (always use max for position generation)
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="max"  # Always use max for position generation
            ),

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

    # Create image assembly pipeline with focus detection
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Use FocusStep for best focus selection
            FocusStep(
                focus_options={'metric': 'variance_of_laplacian'}  # Use variance of Laplacian metric
            ),

            # Step 2: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Stitch images using position files
            ImageStitchingStep(positions_dir=positions_dir)
        ],
        name="Best Focus Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

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

    from ezstitcher.core.step_factories import ZFlatStep, FocusStep, CompositeStep

    # Variance of Laplacian metric

    # Create position generation pipeline with max projection (always use max for position generation)
    laplacian_position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="max"  # Always use max for position generation
            ),

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
    laplacian_positions_dir = laplacian_position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline with Laplacian focus metric
    laplacian_assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=Path("path/to/laplacian_focus"),
        steps=[
            # Step 1: Use FocusStep with Laplacian metric
            FocusStep(
                focus_options={'metric': 'variance_of_laplacian'}
            ),

            # Step 2: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Stitch images using position files
            ImageStitchingStep(positions_dir=laplacian_positions_dir)
        ],
        name="Laplacian Focus Assembly Pipeline"
    )

    # Tenengrad metric

    # Create position generation pipeline with max projection (always use max for position generation)
    tenengrad_position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="max"  # Always use max for position generation
            ),

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
    tenengrad_positions_dir = tenengrad_position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline with Tenengrad focus metric
    tenengrad_assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=Path("path/to/tenengrad_focus"),
        steps=[
            # Step 1: Use FocusStep with Tenengrad metric
            FocusStep(
                focus_options={'metric': 'tenengrad'}
            ),

            # Step 2: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Stitch images using position files
            ImageStitchingStep(positions_dir=tenengrad_positions_dir)
        ],
        name="Tenengrad Focus Assembly Pipeline"
    )

    # Run the pipelines separately
    # orchestrator.run(pipelines=[laplacian_position_pipeline, laplacian_assembly_pipeline])
    # orchestrator.run(pipelines=[tenengrad_position_pipeline, tenengrad_assembly_pipeline])

    # You can also use the FocusAnalyzer static methods directly for more control
    focus_scores = FocusAnalyzer.compute_focus_metrics(
        images,  # List of images in a Z-stack
        metric='laplacian'
    )
    best_focus_index, _ = FocusAnalyzer.find_best_focus(images, metric='laplacian')
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

    # Create position generation pipeline with channel-specific processing
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Channel-specific processing
            Step(
                name="Channel-Specific Processing",
                func={
                    "1": process_dapi,  # Apply process_dapi to channel 1 (DAPI)
                    "2": process_gfp    # Apply process_gfp to channel 2 (GFP)
                },
                group_by='channel'  # Specifies that keys "1" and "2" refer to channel values
            ),

            # Step 3: Create composite for position generation
            CompositeStep(weights=[0.7, 0.3]),  # 70% DAPI, 30% GFP

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    positions_dir = position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline with channel-specific processing
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Channel-specific processing
            Step(
                name="Channel-Specific Processing",
                func={
                    "1": process_dapi,  # Apply process_dapi to channel 1 (DAPI)
                    "2": process_gfp    # Apply process_gfp to channel 2 (GFP)
                },
                group_by='channel'  # Specifies that keys "1" and "2" refer to channel values
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(
                positions_dir=positions_dir,
                variable_components=['channel']  # Stitch each channel separately
            )
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

Advanced Channel-Specific Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also use a dictionary of lists of functions with matching processing arguments. For detailed explanations of this pattern, see :doc:`../concepts/function_handling`.

.. code-block:: python

    from ezstitcher.core.utils import stack

    # Create position generation pipeline with advanced channel-specific processing
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Advanced channel-specific processing
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
                group_by='channel'  # Specifies that keys "1" and "2" refer to channel values
            ),

            # Step 3: Create composite for position generation
            CompositeStep(weights=[0.7, 0.3]),  # 70% DAPI, 30% GFP

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    positions_dir = position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline with advanced channel-specific processing
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Advanced channel-specific processing
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
                group_by='channel'  # Specifies that keys "1" and "2" refer to channel values
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(
                positions_dir=positions_dir,
                variable_components=['channel']  # Stitch each channel separately
            )
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

Creating Composite Images
^^^^^^^^^^^^^^^^^^^^^^

You can combine multiple channels into a composite image. For detailed explanations of composite image creation and the `variable_components=['channel']` parameter, see :ref:`variable-components` in the :doc:`../concepts/step` documentation.

.. note::
   The `create_composite` function can be called with or without the `weights` parameter:

   * Without weights: `func=IP.create_composite` - All channels are weighted equally
   * With weights: `func=(IP.create_composite, {'weights': [0.7, 0.3]})` - Custom weighting for each channel

   The weights list should have the same length as the number of channels being processed.

.. code-block:: python

    from ezstitcher.core.step_factories import ZFlatStep, CompositeStep

    # Create position generation pipeline with equal channel weights
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Process individual channels
            Step(
                name="Channel Processing",
                func=IP.stack_percentile_normalize,
                variable_components=['channel']
            ),

            # Step 3: Create composite images using CompositeStep with equal weights
            CompositeStep(
                weights=None  # Equal weights for all channels (default)
            ),

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
        output_dir=Path("path/to/composite"),
        steps=[
            # Step 1: Process individual channels
            Step(
                name="Channel Processing",
                func=IP.stack_percentile_normalize,
                variable_components=['channel']
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(
                positions_dir=positions_dir,
                variable_components=['channel']  # Stitch each channel separately
            )
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

    # Alternative with custom weights

    # Create position generation pipeline with custom channel weights
    weighted_position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Process individual channels
            Step(
                name="Channel Processing",
                func=IP.stack_percentile_normalize,
                variable_components=['channel']
            ),

            # Step 3: Create composite images with custom weights
            CompositeStep(
                weights=[0.7, 0.3, 0]  # 70% channel 1, 30% channel 2, 0% channel 3
            ),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    weighted_positions_dir = weighted_position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline
    weighted_assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=Path("path/to/weighted_composite"),
        steps=[
            # Step 1: Process individual channels
            Step(
                name="Channel Processing",
                func=IP.stack_percentile_normalize,
                variable_components=['channel']
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(
                positions_dir=weighted_positions_dir,
                variable_components=['channel']  # Stitch each channel separately
            )
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[weighted_position_pipeline, weighted_assembly_pipeline])

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

    from ezstitcher.core.step_factories import ZFlatStep, CompositeStep

    # Create position generation pipeline for Z-stack processing
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="max"  # Use maximum intensity projection
            ),

            # Step 2: Process channels (if multiple channels exist)
            Step(
                name="Channel Processing",
                func=IP.stack_percentile_normalize,
                variable_components=['channel']
            ),

            # Step 3: Create composite for position generation
            CompositeStep(),  # Equal weighting for all channels (default)

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    positions_dir = position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline for Z-stack processing
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(
                method="max"  # Use maximum intensity projection
            ),

            # Step 2: Process channels (if multiple channels exist)
            Step(
                name="Channel Processing",
                func=IP.stack_percentile_normalize,
                variable_components=['channel']
            ),

            # Step 3: Stitch images using position files
            ImageStitchingStep(
                positions_dir=positions_dir,
                variable_components=['channel']  # Stitch each channel separately
            )
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

    # Alternatively, use AutoPipelineFactory for a simpler approach
    from ezstitcher.core import AutoPipelineFactory

    # Create a factory for Z-stack processing and stitching
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        flatten_z=True,  # Flatten Z-stacks in the assembly pipeline
        z_method="max"   # Use maximum intensity projection
    )
    pipelines = factory.create_pipelines()

    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

Benefits of Custom Pipelines
-------------------------

Custom pipelines offer several advantages for intermediate-level tasks:

1. **Precise Control**: Directly specify each step and its parameters
2. **Flexible Workflows**: Create pipelines that match your exact requirements
3. **Terse Implementation**: Write concise code for specific use cases
4. **Direct Access**: Access all pipeline features without abstraction
5. **Custom Logic**: Implement specialized processing logic

Here's an example of a terse custom pipeline that performs channel-specific processing:

.. code-block:: python

    from ezstitcher.core.step_factories import ZFlatStep, CompositeStep

    # Create position generation pipeline with channel-specific processing
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Apply different processing to each channel
            Step(
                name="Channel Processing",
                func={
                    "1": (IP.tophat, {'size': 15}),  # Apply tophat to channel 1
                    "2": (IP.sharpen, {'sigma': 1.0, 'amount': 1.5})  # Apply sharpening to channel 2
                },
                group_by='channel'
            ),

            # Step 3: Create composite for position generation
            CompositeStep(weights=[0.7, 0.3]),

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
        steps=[
            # Step 1: Apply different processing to each channel
            Step(
                name="Channel Processing",
                func={
                    "1": (IP.tophat, {'size': 15}),  # Apply tophat to channel 1
                    "2": (IP.sharpen, {'sigma': 1.0, 'amount': 1.5})  # Apply sharpening to channel 2
                },
                group_by='channel'
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(
                positions_dir=positions_dir,
                variable_components=['channel']  # Stitch each channel separately
            )
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

Customizing Pre-Built Pipelines
---------------------------

You can customize pipelines regardless of how they were created:

Customizing AutoPipelineFactory Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can customize pipelines created by ``AutoPipelineFactory`` to add additional processing steps or modify existing steps:

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create pipelines with AutoPipelineFactory
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True
    )
    pipelines = factory.create_pipelines()

    # Access individual pipelines
    position_pipeline = pipelines[0]
    assembly_pipeline = pipelines[1]

    # Add a custom processing step to the position generation pipeline
    position_pipeline.add_step(
        Step(
            name="Custom Enhancement",
            func=(custom_enhance, {'sigma': 1.5, 'contrast_factor': 2.0})
        ),
        index=1  # Insert after normalization but before composite step
    )

    # Run the customized pipelines
    orchestrator.run(pipelines=pipelines)

This approach combines the simplicity of ``AutoPipelineFactory`` with the flexibility of custom processing.

Customizing Custom Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^

Similarly, you can customize pipelines you've created manually:

.. code-block:: python

    from ezstitcher.core.step_factories import ZFlatStep, CompositeStep

    # Create a basic position generation pipeline
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

    # Add a custom processing step after normalization
    position_pipeline.add_step(
        Step(
            name="Custom Enhancement",
            func=(custom_enhance, {'sigma': 1.5, 'contrast_factor': 2.0})
        ),
        index=2  # Insert after normalization but before composite step
    )

Channel-Specific Processing and Stitching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply different processing to different channels and then stitch the results:

.. code-block:: python

    from ezstitcher.core.step_factories import ZFlatStep, CompositeStep

    # Create position generation pipeline for channel-specific processing
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Channel-specific processing
            Step(
                name="Channel-Specific Processing",
                func={
                    "1": process_dapi,
                    "2": process_gfp
                },
                group_by='channel'
            ),

            # Step 3: Create composite for position generation
            CompositeStep(
                weights=[0.7, 0.3]  # Custom weighting: 70% channel 1, 30% channel 2
            ),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    positions_dir = position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline for channel-specific processing
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Channel-specific processing
            Step(
                name="Channel-Specific Processing",
                func={
                    "1": process_dapi,
                    "2": process_gfp
                },
                group_by='channel'
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(
                positions_dir=positions_dir,
                variable_components=['channel']  # Stitch each channel separately
            )
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

    # Alternatively, use AutoPipelineFactory with channel weights
    from ezstitcher.core import AutoPipelineFactory

    # Create a factory for channel-specific processing and stitching
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        channel_weights=[0.7, 0.3]  # Custom weighting: 70% channel 1, 30% channel 2
    )
    pipelines = factory.create_pipelines()

    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

Complete Workflow Example
^^^^^^^^^^^^^^^^^^^^^^

A complete workflow that combines Z-stack processing, channel-specific processing, and stitching:

.. code-block:: python

    from ezstitcher.core.step_factories import ZFlatStep, FocusStep, CompositeStep
    from ezstitcher.core.focus_analyzer import FocusAnalyzer

    # Create position generation pipeline for complete workflow
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks with channel-specific processing
            Step(
                name="Z-Stack Processing",
                func={
                    "1": (IP.create_projection, {'method': 'max_projection'}),  # Use max projection for channel 1
                    "2": (IP.create_projection, {'method': 'max_projection'})  # Always use max for position generation
                },
                group_by='channel',
                variable_components=['z_index']
            ),

            # Step 2: Channel-specific enhancement
            Step(
                name="Channel Enhancement",
                func={
                    "1": (stack(IP.tophat), {'size': 15}),
                    "2": (stack(IP.sharpen), {'sigma': 1.0, 'amount': 1.5})
                },
                group_by='channel'
            ),

            # Step 3: Create composite for position generation
            CompositeStep(
                weights=[0.6, 0.4]  # Custom weighting: 60% channel 1, 40% channel 2
            ),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    positions_dir = position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline for complete workflow
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks with channel-specific processing
            Step(
                name="Z-Stack Processing",
                func={
                    "1": (IP.create_projection, {'method': 'max_projection'}),  # Use max projection for channel 1
                    "2": (IP.create_projection, {'method': 'best_focus', 'metric': 'laplacian'})  # Use best focus for channel 2
                },
                group_by='channel',
                variable_components=['z_index']
            ),

            # Step 2: Channel-specific enhancement
            Step(
                name="Channel Enhancement",
                func={
                    "1": (stack(IP.tophat), {'size': 15}),
                    "2": (stack(IP.sharpen), {'sigma': 1.0, 'amount': 1.5})
                },
                group_by='channel'
            ),

            # Step 3: Stitch images using position files
            ImageStitchingStep(
                positions_dir=positions_dir,
                variable_components=['channel']  # Stitch each channel separately
            )
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

    # Alternatively, use AutoPipelineFactory and customize the pipelines
    from ezstitcher.core import AutoPipelineFactory

    # Create a factory for a complete workflow
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        flatten_z=True,
        z_method="max",
        channel_weights=[0.6, 0.4]  # Custom weighting: 60% channel 1, 40% channel 2
    )
    pipelines = factory.create_pipelines()

    # Access individual pipelines for customization
    position_pipeline = pipelines[0]
    assembly_pipeline = pipelines[1]

    # Add channel-specific enhancement to position generation pipeline
    position_pipeline.add_step(
        Step(
            name="Channel Enhancement",
            func={
                "1": (stack(IP.tophat), {'size': 15}),
                "2": (stack(IP.sharpen), {'sigma': 1.0, 'amount': 1.5})
            },
            group_by='channel',
        ),
        index=1  # Insert after normalization but before composite step
    )

    # Run the customized pipelines
    orchestrator.run(pipelines=pipelines)

Choosing the Right Approach for Intermediate Tasks
---------------------------------------------

When working on intermediate-level tasks, consider these factors when choosing between approaches:

**Choose Custom Pipelines When:**
- You need precise control over each step
- You're implementing specialized workflows
- You want the most concise code for your specific case
- You need to use features not directly exposed by AutoPipelineFactory

**Choose AutoPipelineFactory When:**
- You're working with standard stitching workflows
- You want to minimize boilerplate code
- You prefer a higher-level interface
- You're building on common patterns

Many experienced users mix both approaches, using AutoPipelineFactory as a starting point for standard workflows and custom pipelines for specialized tasks.

Next Steps
---------

Now that you understand intermediate usage patterns, you're ready to explore advanced topics. For a comprehensive learning path that will guide you through advanced topics and mastering EZStitcher, see :ref:`learning-path` in the introduction.
