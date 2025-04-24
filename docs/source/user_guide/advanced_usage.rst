==============
Advanced Usage
==============

This section explores advanced features of EZStitcher for users who need to extend its functionality or optimize performance.

Custom Processing Functions
-------------------------

While EZStitcher provides many built-in processing functions, you can easily create custom functions to meet specific needs.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    dirs = orchestrator.setup_directories()

    # Create a pipeline with custom function
    custom_pipeline = Pipeline(
        steps=[
            # Use custom function
            Step(
                name="Custom Enhancement",
                func=custom_enhance,
                processing_args={'sigma': 1.5, 'contrast_factor': 2.0},
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            )
        ],
        name="Custom Processing Pipeline"
    )

    # Run the pipeline
    orchestrator.run(pipelines=[custom_pipeline])

Handling Single Images vs. Image Stacks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your function is designed to process a single image but you want to apply it to a stack, use the ``stack()`` utility:

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
                func=stack(enhance_single_image),  # Convert to stack function
                processing_args={'factor': 2.0},
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            )
        ],
        name="Single Image Function Pipeline"
    )

Advanced Custom Functions
^^^^^^^^^^^^^^^^^^^^^^

For more complex processing, you can create functions that handle specific components differently:

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
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            )
        ],
        name="Advanced Custom Pipeline"
    )

Dictionary of Lists with Matching Processing Args
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A more elegant approach is to use a dictionary of lists of functions with matching processing arguments:

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
                        stack(filters.gaussian),        # First apply Gaussian blur
                        stack(filters.unsharp_mask),    # Then apply unsharp mask
                        IP.stack_percentile_normalize   # Finally normalize
                    ],
                    "2": [  # Process channel 2 (GFP)
                        stack(filters.median),          # First apply median filter
                        stack(filters.unsharp_mask),    # Then apply unsharp mask
                        IP.stack_percentile_normalize   # Finally normalize
                    ]
                },
                group_by='channel',  # Specifies that keys "1" and "2" refer to channel values
                processing_args={
                    "1": [
                        {'sigma': 1.0},                  # Args for gaussian
                        {'radius': 1.0, 'amount': 2.0},  # Args for unsharp_mask
                        {'low_percentile': 1.0, 'high_percentile': 99.0}  # Args for normalize
                    ],
                    "2": [
                        {'selem': None},                 # Args for median
                        {'radius': 0.5, 'amount': 1.5},  # Args for unsharp_mask
                        {'low_percentile': 1.0, 'high_percentile': 99.0}  # Args for normalize
                    ]
                },
                input_dir=dirs['input'],
                output_dir=dirs['processed']
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
^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^

In EZStitcher, multithreading processes each well in a separate thread, with the number of concurrent threads limited by ``num_workers``. Pipelines are executed sequentially for each well, and steps within a pipeline are executed sequentially. This approach provides good performance while avoiding race conditions.

Performance Considerations
^^^^^^^^^^^^^^^^^^^^^^

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

EZStitcher can be extended to support additional microscope types by implementing custom microscope handlers.

Understanding Microscope Handlers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Microscope handlers are responsible for:

1. Parsing file names to extract components (well, site, channel, etc.)
2. Locating images based on these components
3. Providing metadata about the microscope setup

Creating a Custom Microscope Handler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a custom microscope handler, subclass ``BaseMicroscopeHandler``:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import BaseMicroscopeHandler
    import re
    from pathlib import Path

    class CustomMicroscopeHandler(BaseMicroscopeHandler):
        """Handler for a custom microscope format."""

        # Regular expression for parsing file names
        # Example: Sample_A01_s3_w2_z1.tif
        FILE_PATTERN = re.compile(
            r'(?P<prefix>.+)_'
            r'(?P<well>[A-Z][0-9]{2})_'
            r's(?P<site>[0-9]+)_'
            r'w(?P<channel>[0-9]+)_'
            r'z(?P<z_index>[0-9]+)'
            r'\.tif$'
        )

        def __init__(self, plate_path):
            """Initialize the handler."""
            super().__init__(plate_path)

        def get_wells(self):
            """Get list of wells in the plate."""
            wells = set()
            for file_path in Path(self.plate_path).glob('**/*.tif'):
                match = self.FILE_PATTERN.match(file_path.name)
                if match:
                    wells.add(match.group('well'))
            return sorted(list(wells))

        def get_sites(self, well):
            """Get list of sites for a well."""
            sites = set()
            for file_path in Path(self.plate_path).glob(f'**/*_{well}_*.tif'):
                match = self.FILE_PATTERN.match(file_path.name)
                if match:
                    sites.add(match.group('site'))
            return sorted(list(sites))

        def get_channels(self, well, site=None):
            """Get list of channels for a well/site."""
            channels = set()
            pattern = f'**/*_{well}_s{site}_*.tif' if site else f'**/*_{well}_*.tif'
            for file_path in Path(self.plate_path).glob(pattern):
                match = self.FILE_PATTERN.match(file_path.name)
                if match:
                    channels.add(match.group('channel'))
            return sorted(list(channels))

        def get_z_indices(self, well, site=None, channel=None):
            """Get list of z-indices for a well/site/channel."""
            z_indices = set()
            pattern = f'**/*_{well}_s{site}_w{channel}_*.tif'
            for file_path in Path(self.plate_path).glob(pattern):
                match = self.FILE_PATTERN.match(file_path.name)
                if match:
                    z_indices.add(match.group('z_index'))
            return sorted(list(z_indices))

        def get_image_path(self, well, site, channel, z_index=None):
            """Get path to a specific image."""
            z_part = f'_z{z_index}' if z_index else ''
            pattern = f'**/*_{well}_s{site}_w{channel}{z_part}.tif'
            for file_path in Path(self.plate_path).glob(pattern):
                if self.FILE_PATTERN.match(file_path.name):
                    return str(file_path)
            return None

        def parse_file_name(self, file_path):
            """Parse components from a file name."""
            match = self.FILE_PATTERN.match(Path(file_path).name)
            if match:
                return {
                    'well': match.group('well'),
                    'site': match.group('site'),
                    'channel': match.group('channel'),
                    'z_index': match.group('z_index')
                }
            return None

        @classmethod
        def can_handle(cls, plate_path):
            """Check if this handler can handle the given plate."""
            # Check if any files match the pattern
            for file_path in Path(plate_path).glob('**/*.tif'):
                if cls.FILE_PATTERN.match(file_path.name):
                    return True
            return False

Registering a Custom Microscope Handler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Register your custom handler with EZStitcher:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import register_microscope_handler

    # Register the custom handler
    register_microscope_handler(CustomMicroscopeHandler)

    # Now EZStitcher will automatically detect and use your handler
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="/path/to/custom/plate"
    )

Using a Specific Microscope Handler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also explicitly specify which handler to use:

.. code-block:: python

    # Create orchestrator with specific handler
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="/path/to/plate",
        microscope_handler=CustomMicroscopeHandler
    )

Integration with Other Tools
-------------------------

EZStitcher can be integrated with other image processing and analysis tools to create comprehensive workflows.

Exporting Data for Analysis
^^^^^^^^^^^^^^^^^^^^^^^

After processing with EZStitcher, you can export data for analysis with other tools:

.. code-block:: python

    import numpy as np
    from skimage import io
    import pandas as pd

    def export_for_analysis(stitched_image_path, output_csv):
        """Export image data for analysis."""
        # Load the stitched image
        image = io.imread(stitched_image_path)

        # Extract features (example: mean intensity in regions)
        regions = []
        for i in range(0, image.shape[0], 100):
            for j in range(0, image.shape[1], 100):
                region = image[i:i+100, j:j+100]
                regions.append({
                    'x': j,
                    'y': i,
                    'mean_intensity': np.mean(region),
                    'std_intensity': np.std(region),
                    'min_intensity': np.min(region),
                    'max_intensity': np.max(region)
                })

        # Save as CSV for analysis
        df = pd.DataFrame(regions)
        df.to_csv(output_csv, index=False)

        return df

    # Use in a pipeline
    from ezstitcher.core.steps import Step

    # Create a pipeline with export step
    export_pipeline = Pipeline(
        steps=[
            # Process and stitch images
            # ...

            # Export data for analysis
            Step(
                name="Export Data",
                func=lambda images: export_for_analysis(
                    stitched_image_path=dirs['stitched'] / "A01_stitched.tif",
                    output_csv=dirs['stitched'] / "A01_analysis.csv"
                ) and images,  # Return images unchanged
                input_dir=dirs['stitched'],
                output_dir=dirs['stitched']
            )
        ],
        name="Export Pipeline"
    )

Integration with Deep Learning Frameworks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can integrate EZStitcher with deep learning frameworks like TensorFlow or PyTorch:

.. code-block:: python

    import tensorflow as tf

    # Load a pre-trained model
    model = tf.keras.models.load_model('/path/to/model')

    def apply_deep_learning(images):
        """Apply deep learning model to images."""
        result = []
        for img in images:
            # Preprocess image for the model
            input_tensor = tf.convert_to_tensor(img[np.newaxis, ..., np.newaxis], dtype=tf.float32)

            # Run inference
            predictions = model.predict(input_tensor)

            # Post-process predictions
            segmentation_map = predictions[0, ..., 0]

            # Return the segmentation map
            result.append(segmentation_map)

        return result

    # Use in a pipeline
    deep_learning_pipeline = Pipeline(
        steps=[
            # Preprocess images
            Step(
                name="Preprocess",
                func=IP.stack_percentile_normalize,
                input_dir=dirs['input'],
                output_dir=dirs['processed']
            ),

            # Apply deep learning model
            Step(
                name="Deep Learning Segmentation",
                func=apply_deep_learning,
                input_dir=dirs['processed'],
                output_dir=dirs['segmented']
            )
        ],
        name="Deep Learning Pipeline"
    )

Command-Line Integration
^^^^^^^^^^^^^^^^^^^^

You can create command-line scripts that use EZStitcher:

.. code-block:: python

    #!/usr/bin/env python
    # process_plate.py

    import argparse
    from pathlib import Path
    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP

    def main():
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Process microscopy plate')
        parser.add_argument('--plate-path', required=True, help='Path to plate folder')
        parser.add_argument('--output-dir', help='Output directory')
        parser.add_argument('--num-workers', type=int, default=1, help='Number of worker threads')
        parser.add_argument('--wells', nargs='+', help='Wells to process (default: all)')
        args = parser.parse_args()

        # Create configuration
        config = PipelineConfig(num_workers=args.num_workers)

        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            config=config,
            plate_path=Path(args.plate_path)
        )

        # Set up directories
        dirs = orchestrator.setup_directories()

        # Create pipeline
        pipeline = Pipeline(
            steps=[
                # Process images
                Step(
                    name="Image Processing",
                    func=IP.stack_percentile_normalize,
                    variable_components=['channel'],
                    input_dir=dirs['input'],
                    output_dir=dirs['processed']
                ),

                # Generate positions
                PositionGenerationStep(
                    name="Generate Positions",
                    input_dir=dirs['processed'],
                    output_dir=dirs['positions']
                ),

                # Stitch images
                ImageStitchingStep(
                    name="Stitch Images",
                    input_dir=dirs['processed'],
                    positions_dir=dirs['positions'],
                    output_dir=dirs['stitched']
                )
            ],
            name="Processing Pipeline"
        )

        # Run pipeline
        orchestrator.run(
            pipelines=[pipeline],
            well_filter=args.wells
        )

        print(f"Processing complete. Results in {dirs['stitched']}")

    if __name__ == '__main__':
        main()

Usage:

.. code-block:: bash

    python process_plate.py --plate-path /path/to/plate --num-workers 4 --wells A01 B02

Next Steps
---------

Now that you understand advanced usage patterns, you can:

* Create custom processing functions tailored to your specific needs
* Optimize performance with multithreaded processing
* Extend EZStitcher to support new microscope types
* Integrate EZStitcher with other tools in your workflow

For complete workflow examples, see the :doc:`practical_examples` section.
