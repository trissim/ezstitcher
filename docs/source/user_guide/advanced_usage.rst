==============
Advanced Usage
==============

This section explores advanced features of EZStitcher for users who need to extend its functionality or optimize performance.

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

Using Custom Functions in Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use custom functions in pipelines just like built-in functions:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step
    from pathlib import Path

    # Create configuration and orchestrator
    config = PipelineConfig(num_workers=1)
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path=Path("/path/to/plate")
    )

    # Create a pipeline with custom function
    custom_pipeline = Pipeline(
        steps=[
            # Use custom function
            Step(
                name="Custom Enhancement",
                func=(custom_enhance, {'sigma': 1.5, 'contrast_factor': 2.0}),
                input_dir=orchestrator.workspace_path
            )
        ],
        name="Custom Processing Pipeline"
    )

    # Run the pipeline
    orchestrator.run(pipelines=[custom_pipeline])

Handling Single Images vs. Image Stacks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your function is designed to process a single image but you want to apply it to a stack, use the ``stack()`` utility. For detailed explanations of the `stack()` utility and how it works, see :doc:`../concepts/function_handling`.

.. code-block:: python

    from ezstitcher.core.utils import stack

    # Function that processes a single image
    def enhance_single_image(img, factor=1.5):
        """Enhance a single image."""
        return np.clip(img * factor, 0, 1)

    # Create a pipeline that applies the function to each image in a stack
    pipeline = Pipeline(
        steps=[
            Step(
                name="Enhance Images",
                func=(stack(enhance_single_image), {'factor': 2.0}),  # Convert to stack function with args
                input_dir=orchestrator.workspace_path
            )
        ],
        name="Single Image Function Pipeline"
    )

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

    # Use the function in a pipeline
    pipeline = Pipeline(
        steps=[
            Step(
                name="Channel-Aware Processing",
                func=process_by_channel,
                group_by='channel',  # Group by channel to pass channel info
                input_dir=orchestrator.workspace_path
            )
        ],
        name="Advanced Custom Pipeline"
    )

Dictionary of Lists with Matching Processing Args
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A more elegant approach is to use a dictionary of lists of functions with matching processing arguments. This is one of the most powerful function handling patterns in EZStitcher. For detailed explanations of this pattern and other function handling patterns, see :doc:`../concepts/function_handling`.

.. code-block:: python

    from ezstitcher.core.utils import stack
    from skimage import filters

    # Create a pipeline with dictionary of lists of functions and matching kwargs
    advanced_pipeline = Pipeline(
        steps=[
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
                group_by='channel',  # Specifies that keys "1" and "2" refer to channel values
                input_dir=orchestrator.workspace_path
            )
        ],
        name="Advanced Dictionary Pipeline"
    )

This approach provides several advantages:
- More concise and readable than a custom function with conditionals
- Easier to modify and extend with additional channels or processing steps
- Clearer separation between processing logic and parameters
- More flexible for experimentation with different parameter values

Multithreaded Processing
----------------------

EZStitcher supports multithreaded processing to improve performance when working with large datasets.

Configuring Multithreading
^^^^^^^^^^^^^^^^^^^^^^^^^^

Multithreading is configured through the ``PipelineConfig`` class:

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

Extending with New Microscope Types
--------------------------------

EZStitcher can be extended to support additional microscope types by implementing custom microscope handlers. This allows you to process images from microscopes with different file naming conventions and directory structures.

Microscope handlers are responsible for:

1. Parsing file names to extract components (well, site, channel, etc.)
2. Locating images based on these components
3. Providing metadata about the microscope setup

For detailed information about creating and registering custom microscope handlers, see :doc:`../development/extending`.

Next Steps
----------

Now that you understand advanced usage patterns, you're ready to master EZStitcher and explore integration with other tools. For a comprehensive learning path that covers mastering EZStitcher, see :ref:`learning-path` in the introduction.

For more information on integrating with other tools, see the :doc:`integration` section.
