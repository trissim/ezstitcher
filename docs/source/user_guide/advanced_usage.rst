==============
Advanced Usage
==============

This section explores advanced features of EZStitcher for users who need to extend its functionality or optimize performance.

.. note::
   For common operations like Z-stack flattening and channel compositing, use specialized step subclasses
   like ``ZFlatStep`` and ``CompositeStep`` instead of manually configuring ``variable_components``.

   For channel-specific processing, using a dictionary of functions with ``group_by='channel'`` is the
   appropriate approach, as shown in various examples in this guide.

   For more information about specialized steps, see :doc:`../concepts/specialized_steps`.

Introduction
-----------

This guide covers advanced usage patterns for EZStitcher, building on the concepts introduced in the basic and intermediate guides. This guide shows how to:

1. Create custom processing functions for specialized needs
2. Implement multithreaded processing for improved performance
3. Build highly customized pipelines for advanced workflows

Both the AutoPipelineFactory and custom pipeline approaches can be extended with advanced features. This guide will show you how to use both approaches for advanced-level tasks.

.. note::
   Directory paths are automatically resolved between steps in EZStitcher. The first step should specify
   ``input_dir=orchestrator.workspace_path`` to ensure processing happens on workspace copies,
   but subsequent steps will automatically use the output of the previous step as their input.
   See :doc:`../concepts/directory_structure` for details on how EZStitcher manages directories.

.. important::
   Understanding the relationship between ``variable_components`` and ``group_by`` parameters is crucial for
   correctly configuring pipeline steps. For detailed explanations of these parameters and their relationships,
   see :doc:`../concepts/step`.

Custom Processing Functions
-------------------------

While EZStitcher provides many built-in processing functions, you can easily create custom functions to meet specific needs. For detailed explanations of function handling patterns, see :doc:`../concepts/function_handling`.

Creating Custom Functions
^^^^^^^^^^^^^^^^^^^^^^

Custom processing functions should follow these guidelines:

1. Accept a list of images as input
2. Return a list of processed images as output
3. Preserve the order and number of images (unless explicitly combining or filtering)

Here's a simple example of a custom processing function:

.. code-block:: python

    import numpy as np
    from skimage import filters

    def custom_enhance(images, sigma=1.0, contrast_factor=1.5):
        """
        Custom enhancement function that combines Gaussian blur and contrast adjustment.

        Args:
            images: List of input images
            sigma: Sigma for Gaussian blur
            contrast_factor: Factor to increase contrast

        Returns:
            List of processed images
        """
        result = []
        for img in images:
            # Apply Gaussian blur
            blurred = filters.gaussian(img, sigma=sigma)

            # Enhance contrast
            mean_val = np.mean(blurred)
            enhanced = mean_val + contrast_factor * (blurred - mean_val)

            # Clip values to valid range
            enhanced = np.clip(enhanced, 0, 1)

            result.append(enhanced)

        return result

Creating Advanced Custom Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For advanced workflows that require custom processing functions, creating custom pipelines from scratch is the recommended approach:

.. code-block:: python

    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step
    from ezstitcher.core.specialized_steps import ZFlatStep, CompositeStep, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_processor import ImageProcessor as IP
    from pathlib import Path

    # Define custom processing functions
    def custom_denoise(images, strength=0.5):
        """Custom denoising function."""
        denoised = []
        for img in images:
            # Apply denoising (example implementation)
            from skimage.restoration import denoise_nl_means
            denoised_img = denoise_nl_means(img, h=strength)
            denoised.append(denoised_img)
        return denoised

    def custom_enhance(images, sigma=1.5, contrast_factor=2.0):
        """Custom enhancement function."""
        enhanced = []
        for img in images:
            # Apply sharpening
            sharpened = IP.sharpen(img, sigma=sigma)
            # Apply contrast adjustment
            enhanced_img = IP.adjust_contrast(sharpened, factor=contrast_factor)
            enhanced.append(enhanced_img)
        return enhanced

    # Create a custom position generation pipeline with advanced processing
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Apply custom denoising
            Step(
                name="Custom Denoising",
                func=(custom_denoise, {'strength': 0.5})
            ),

            # Step 2: Normalize images
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Apply custom enhancement
            Step(
                name="Custom Enhancement",
                func=(custom_enhance, {'sigma': 1.5, 'contrast_factor': 2.0})
            ),

            # Step 4: Create composite for position generation
            CompositeStep(weights=[0.7, 0.3, 0]),

            # Step 5: Generate positions
            PositionGenerationStep()
        ],
        name="Advanced Position Generation Pipeline"
    )

    # Create a custom assembly pipeline with advanced processing
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=Path("path/to/output"),
        steps=[
            # Step 1: Apply custom denoising
            Step(
                name="Custom Denoising",
                func=(custom_denoise, {'strength': 0.5})
            ),

            # Step 2: Normalize images
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Stitch images
            ImageStitchingStep()
        ],
        name="Advanced Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

This approach provides several benefits for advanced workflows:

1. **Readability**: The pipeline structure is explicit and easy to understand
2. **Maintainability**: Changes can be made directly to the pipeline definition
3. **Flexibility**: Complete control over each step and its parameters
4. **Robustness**: No risk of unexpected behavior from modifying factory pipelines

.. important::
   While it is technically possible to modify pipelines created by AutoPipelineFactory after creation,
   this approach is generally not recommended for advanced workflows. Creating custom pipelines from scratch
   is usually more readable, maintainable, and less error-prone for any workflow that requires customization
   beyond what AutoPipelineFactory parameters provide.

Using Custom Functions in Custom Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For complete control, you can create custom pipelines with your functions:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.specialized_steps import ZFlatStep, CompositeStep
    from pathlib import Path

    # Create configuration and orchestrator
    config = PipelineConfig(num_workers=1)
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path=Path("/path/to/plate")
    )

    # Create position generation pipeline with custom function
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Use custom enhancement function
            Step(
                name="Custom Enhancement",
                func=(custom_enhance, {'sigma': 1.5, 'contrast_factor': 2.0})
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

    # Create image assembly pipeline with custom function
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Use custom enhancement function
            Step(
                name="Custom Enhancement",
                func=(custom_enhance, {'sigma': 1.5, 'contrast_factor': 2.0})
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(positions_dir=positions_dir)
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

Handling Single Images vs. Image Stacks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your function is designed to process a single image but you want to apply it to a stack, use the ``stack()`` utility. For detailed explanations of the `stack()` utility and how it works, see :doc:`../concepts/function_handling`.

.. code-block:: python

    from ezstitcher.core.utils import stack

    # Function that processes a single image
    def enhance_single_image(img, factor=1.5):
        """Enhance a single image."""
        return np.clip(img * factor, 0, 1)

    # Create position generation pipeline that applies the function to each image
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Apply single-image function to each image in the stack
            Step(
                name="Enhance Images",
                func=(stack(enhance_single_image), {'factor': 2.0})  # Convert to stack function with args
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
        steps=[
            # Step 1: Apply single-image function to each image in the stack
            Step(
                name="Enhance Images",
                func=(stack(enhance_single_image), {'factor': 2.0})  # Convert to stack function with args
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(positions_dir=positions_dir)
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

Advanced Custom Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

For more complex processing, you can create functions that handle specific components differently. For detailed explanations of how component information is passed to functions, see :ref:`variable-components` and :ref:`group-by` in the :doc:`../concepts/step` documentation.

.. code-block:: python

    def process_by_channel(images, channel_info):
        """
        Process images differently based on channel information.

        Args:
            images: List of input images
            channel_info: Dictionary with channel information

        Returns:
            List of processed images
        """
        result = []
        for i, img in enumerate(images):
            channel = channel_info.get('channel')

            if channel == '1':  # DAPI channel
                # Enhance nuclei
                processed = filters.gaussian(img, sigma=1.0)
                processed = filters.unsharp_mask(processed, radius=1.0, amount=2.0)
            elif channel == '2':  # GFP channel
                # Enhance cell structures
                processed = filters.gaussian(img, sigma=0.5)
                processed = filters.unsharp_mask(processed, radius=0.5, amount=1.5)
            else:
                # Default processing
                processed = img

            result.append(processed)

        return result

    # Create position generation pipeline with channel-aware processing
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Channel-aware processing
            Step(
                name="Channel-Aware Processing",
                func=process_by_channel,
                group_by='channel'  # Group by channel to pass channel info
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

    # Create image assembly pipeline with channel-aware processing
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Channel-aware processing
            Step(
                name="Channel-Aware Processing",
                func=process_by_channel,
                group_by='channel'  # Group by channel to pass channel info
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

Dictionary of Lists with Matching Processing Args
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A more elegant approach is to use a dictionary of lists of functions with matching processing arguments. This is one of the most powerful function handling patterns in EZStitcher. For detailed explanations of this pattern and other function handling patterns, see :doc:`../concepts/function_handling`.

.. code-block:: python

    from ezstitcher.core.utils import stack
    from skimage import filters

    # Create position generation pipeline with dictionary of functions
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
                        (stack(filters.gaussian), {'sigma': 1.0}),        # First apply Gaussian blur with args
                        (stack(filters.unsharp_mask), {'radius': 1.0, 'amount': 2.0}),    # Then apply unsharp mask with args
                        (IP.stack_percentile_normalize, {'low_percentile': 1.0, 'high_percentile': 99.0})   # Finally normalize with args
                    ],
                    "2": [  # Process channel 2 (GFP)
                        (stack(filters.median), {'selem': None}),          # First apply median filter with args
                        (stack(filters.unsharp_mask), {'radius': 0.5, 'amount': 1.5}),    # Then apply unsharp mask with args
                        (IP.stack_percentile_normalize, {'low_percentile': 1.0, 'high_percentile': 99.0})   # Finally normalize with args
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

    # Create image assembly pipeline with dictionary of functions
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Advanced channel-specific processing
            Step(
                name="Advanced Channel Processing",
                func={
                    "1": [  # Process channel 1 (DAPI)
                        (stack(filters.gaussian), {'sigma': 1.0}),        # First apply Gaussian blur with args
                        (stack(filters.unsharp_mask), {'radius': 1.0, 'amount': 2.0}),    # Then apply unsharp mask with args
                        (IP.stack_percentile_normalize, {'low_percentile': 1.0, 'high_percentile': 99.0})   # Finally normalize with args
                    ],
                    "2": [  # Process channel 2 (GFP)
                        (stack(filters.median), {'selem': None}),          # First apply median filter with args
                        (stack(filters.unsharp_mask), {'radius': 0.5, 'amount': 1.5}),    # Then apply unsharp mask with args
                        (IP.stack_percentile_normalize, {'low_percentile': 1.0, 'high_percentile': 99.0})   # Finally normalize with args
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

This approach provides several advantages:
- More concise and readable than a custom function with conditionals
- Easier to modify and extend with additional channels or processing steps
- Clearer separation between processing logic and parameters
- More flexible for experimentation with different parameter values

Conditional Processing
^^^^^^^^^^^^^^^^^^^^^

You can implement conditional processing based on well, site, or other context information:

.. code-block:: python

    from ezstitcher.core.specialized_steps import ZFlatStep, CompositeStep

    # Create position generation pipeline with conditional processing
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Apply different processing based on well
            Step(
                name="Conditional Processing",
                func=lambda images, context: (
                    process_control(images) if context.well == 'A01' else
                    process_treatment(images)
                )
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
        steps=[
            # Step 1: Apply different processing based on well
            Step(
                name="Conditional Processing",
                func=lambda images, context: (
                    process_control(images) if context.well == 'A01' else
                    process_treatment(images)
                )
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(positions_dir=positions_dir)
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

Multithreaded Processing
----------------------

EZStitcher supports multithreaded processing to improve performance when working with large datasets.

Configuring Multithreading
^^^^^^^^^^^^^^^^^^^^^^^^^^

Multithreading is configured through the ``PipelineConfig`` class:

Using AutoPipelineFactory:

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with multithreaded processing
    config = PipelineConfig(
        num_workers=4  # Use 4 worker threads
    )

    # Create orchestrator with multithreading
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="/path/to/plate"
    )

    # Create pipelines with AutoPipelineFactory
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True
    )
    pipelines = factory.create_pipelines()

    # Run the pipelines with multithreading
    orchestrator.run(pipelines=pipelines)

Pipeline Composition
^^^^^^^^^^^^^^^^^

You can create pipelines that build on each other's outputs:

.. code-block:: python

    from ezstitcher.core.specialized_steps import ZFlatStep, CompositeStep

    # Create a preprocessing pipeline
    preprocess_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            Step(
                name="Preprocessing",
                func=preprocess_images
            )
        ],
        name="Preprocessing Pipeline"
    )

    # Create a position generation pipeline that uses the output of the preprocessing pipeline
    position_pipeline = Pipeline(
        input_dir=preprocess_pipeline.output_dir,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Create composite for position generation
            CompositeStep(),

            # Step 3: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    positions_dir = position_pipeline.steps[-1].output_dir

    # Create an image assembly pipeline
    assembly_pipeline = Pipeline(
        input_dir=preprocess_pipeline.output_dir,
        steps=[
            # Stitch images using position files
            ImageStitchingStep(positions_dir=positions_dir)
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines in sequence
    orchestrator.run(pipelines=[preprocess_pipeline, position_pipeline, assembly_pipeline])

Using Manual Pipeline Creation:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with multithreading
    config = PipelineConfig(
        num_workers=4  # Use 4 worker threads
    )

    # Create orchestrator with multithreading
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="/path/to/plate"
    )

    # Run pipelines with multithreading
    orchestrator.run(pipelines=[pipeline1, pipeline2])

How Multithreading Works
^^^^^^^^^^^^^^^^^^^^^^^^

In EZStitcher, multithreading processes each well in a separate thread, with the number of concurrent threads limited by ``num_workers``. Pipelines are executed sequentially for each well, and steps within a pipeline are executed sequentially. This approach provides good performance while avoiding race conditions.

Performance Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^

When using multithreading, consider these factors:

* **Memory Usage**: Each thread requires memory for loading and processing images
* **CPU Cores**: For optimal performance, set ``num_workers`` to match available CPU cores
* **Image Size**: For large images, use fewer threads to avoid memory issues

For example:

.. code-block:: python

    # For a system with 8 cores processing small images
    config = PipelineConfig(num_workers=8)  # Use all cores

    # For a system with 8 cores processing large images
    config = PipelineConfig(num_workers=4)  # Use fewer threads

Extending AutoPipelineFactory
--------------------------

For advanced use cases, you can extend ``AutoPipelineFactory`` to create a custom factory that includes your specialized functionality:

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.steps import Step
    from ezstitcher.core.pipeline import Pipeline

    class CustomPipelineFactory(AutoPipelineFactory):
        """Custom pipeline factory with additional functionality."""

        def __init__(self, input_dir, custom_param=None, **kwargs):
            """Initialize with custom parameters."""
            super().__init__(input_dir, **kwargs)
            self.custom_param = custom_param

        def create_pipelines(self):
            """Create pipelines with custom functionality."""
            # Get standard pipelines from parent class
            pipelines = super().create_pipelines()

            # Access individual pipelines
            position_pipeline = pipelines[0]
            assembly_pipeline = pipelines[1]

            # Add custom processing to position generation pipeline
            if self.custom_param:
                position_pipeline.add_step(
                    Step(
                        name="Custom Processing",
                        func=(self.custom_process, {'param': self.custom_param})
                    ),
                    index=1  # Insert after normalization
                )

            # Add a third pipeline for additional processing
            analysis_pipeline = Pipeline(
                steps=[
                    # Add steps for analysis
                    Step(
                        name="Analysis",
                        func=self.analyze_results,
                        input_dir=assembly_pipeline.output_dir
                    )
                ],
                name="Analysis Pipeline"
            )

            # Add the analysis pipeline to the list
            pipelines.append(analysis_pipeline)

            return pipelines

        @staticmethod
        def custom_process(images, param=None):
            """Custom processing function."""
            # Implement custom processing
            return images

        @staticmethod
        def analyze_results(images):
            """Analyze stitched images."""
            # Implement analysis
            return images

    # Use the custom factory
    factory = CustomPipelineFactory(
        input_dir=orchestrator.workspace_path,
        custom_param="value",
        normalize=True,
        flatten_z=True,
        z_method="max"
    )
    pipelines = factory.create_pipelines()

    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

Extending with New Microscope Types
--------------------------------

EZStitcher can be extended to support additional microscope types by implementing custom microscope handlers. This allows you to process images from microscopes with different file naming conventions and directory structures.

Microscope handlers are responsible for:

1. Parsing file names to extract components (well, site, channel, etc.)
2. Locating images based on these components
3. Providing metadata about the microscope setup

For detailed information about creating and registering custom microscope handlers, see :doc:`../development/extending`.

Choosing the Right Approach for Advanced Tasks
---------------------------------------------

When working on advanced-level tasks, consider these factors when choosing between approaches:

**Choose Custom Pipelines When:**
- You need to implement complex, specialized workflows
- You're working with custom processing functions
- You need precise control over pipeline structure
- You're implementing conditional processing logic
- You want maximum readability and maintainability for complex pipelines

**Choose AutoPipelineFactory When:**
- You're working with standard stitching workflows
- The built-in parameters (normalize, flatten_z, z_method, etc.) are sufficient
- You want to minimize boilerplate code
- You prefer a higher-level interface

.. important::
   For advanced workflows that require custom processing steps, creating custom pipelines from scratch
   is the recommended approach. This provides maximum flexibility, readability, and maintainability.

Next Steps
----------

Now that you understand advanced usage patterns, you're ready to master EZStitcher and explore integration with other tools. For a comprehensive learning path that covers mastering EZStitcher, see :ref:`learning-path` in the introduction.

For more information on integrating with other tools, see the :doc:`integration` section.
