Extending EZStitcher
==================

This guide explains how to extend EZStitcher with new functionality.

Adding a New Microscope Type
-------------------------

EZStitcher is designed to be easily extended with support for new microscope types. To add a new microscope type:

1. Create a new file in the `ezstitcher/microscopes/` directory, e.g., `new_microscope.py`
2. Implement the `FilenameParser` and `MetadataHandler` interfaces
3. Register the new microscope type in `ezstitcher/microscopes/__init__.py`

Here's an example implementation:

.. code-block:: python

    """
    NewMicroscope implementations for ezstitcher.

    This module provides concrete implementations of FilenameParser and MetadataHandler
    for NewMicroscope microscopes.
    """

    import re
    import logging
    from pathlib import Path
    from typing import Dict, List, Optional, Union, Any, Tuple

    from ezstitcher.core.microscope_interfaces import FilenameParser, MetadataHandler

    logger = logging.getLogger(__name__)


    class NewMicroscopeFilenameParser(FilenameParser):
        """Filename parser for NewMicroscope microscopes."""

        # Define the regex pattern as a class attribute
        FILENAME_PATTERN = r'([A-Z]\d{2})_s(\d+)_w(\d+)(?:_z(\d+))?\.(?:tif|tiff)'

        @classmethod
        def can_parse(cls, filename: str) -> bool:
            """Check if this parser can parse the given filename."""
            # Use the class attribute pattern
            return bool(re.match(cls.FILENAME_PATTERN, filename))

        def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
            """Parse a NewMicroscope filename into its components."""
            match = re.match(self.FILENAME_PATTERN, filename)

            if not match:
                return None

            well, site, channel, z_index = match.groups()

            return {
                'well': well,
                'site': int(site),
                'channel': int(channel),
                'z_index': int(z_index) if z_index else None,
                'extension': Path(filename).suffix
            }

        def construct_filename(self, well: str, site: Optional[Union[int, str]] = None,
                              channel: Optional[int] = None,
                              z_index: Optional[Union[int, str]] = None,
                              extension: str = '.tif',
                              site_padding: int = 3, z_padding: int = 3) -> str:
            """Construct a NewMicroscope filename from components."""
            # Format site number with padding
            if site is None:
                site_str = ""
            elif isinstance(site, str) and site == self.PLACEHOLDER_PATTERN:
                site_str = f"_s{site}"
            else:
                site_str = f"_s{int(site):0{site_padding}d}"

            # Format channel number
            if channel is None:
                channel_str = ""
            else:
                channel_str = f"_w{int(channel)}"

            # Format z-index with padding
            if z_index is None:
                z_str = ""
            elif isinstance(z_index, str) and z_index == self.PLACEHOLDER_PATTERN:
                z_str = f"_z{z_index}"
            else:
                z_str = f"_z{int(z_index):0{z_padding}d}"

            # Ensure extension starts with a dot
            if not extension.startswith('.'):
                extension = f".{extension}"

            return f"{well}{site_str}{channel_str}{z_str}{extension}"


    class NewMicroscopeMetadataHandler(MetadataHandler):
        """Metadata handler for NewMicroscope microscopes."""

        def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
            """Find the metadata file for a NewMicroscope plate."""
            plate_path = Path(plate_path)

            # Look for metadata file
            metadata_file = plate_path / "metadata.xml"
            if metadata_file.exists():
                return metadata_file

            return None

        def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
            """Get grid dimensions for stitching from NewMicroscope metadata."""
            metadata_file = self.find_metadata_file(plate_path)
            if not metadata_file:
                # Default grid size if metadata file not found
                return (3, 3)

            # Parse metadata file to extract grid dimensions
            # This is just an example, implement your own parsing logic
            try:
                # Parse XML or other format
                # ...

                # Return grid dimensions
                return (4, 4)
            except Exception as e:
                logger.error(f"Error parsing metadata file: {e}")
                return (3, 3)

        def get_pixel_size(self, plate_path: Union[str, Path]) -> Optional[float]:
            """Get the pixel size from NewMicroscope metadata."""
            metadata_file = self.find_metadata_file(plate_path)
            if not metadata_file:
                return None

            # Parse metadata file to extract pixel size
            # This is just an example, implement your own parsing logic
            try:
                # Parse XML or other format
                # ...

                # Return pixel size in micrometers
                return 0.65
            except Exception as e:
                logger.error(f"Error parsing metadata file: {e}")
                return None

Then, register the new microscope type in `ezstitcher/microscopes/__init__.py`:

.. code-block:: python

    """
    Microscope-specific implementations for ezstitcher.

    This package contains modules for different microscope types, each providing
    concrete implementations of FilenameParser and MetadataHandler interfaces.
    """

    # Import microscope handlers for easier access
    from ezstitcher.microscopes.imagexpress import ImageXpressFilenameParser, ImageXpressMetadataHandler
    from ezstitcher.microscopes.opera_phenix import OperaPhenixFilenameParser, OperaPhenixMetadataHandler
    from ezstitcher.microscopes.new_microscope import NewMicroscopeFilenameParser, NewMicroscopeMetadataHandler

Adding Custom Preprocessing Functions
---------------------------------

You can add custom preprocessing functions to the `ImagePreprocessor` class:

.. code-block:: python

    from ezstitcher.core.image_preprocessor import ImagePreprocessor
    import numpy as np
    from scipy import ndimage

    # Add a new static method to ImagePreprocessor
    @staticmethod
    def my_custom_preprocessing(image, param1=1.0, param2=2.0):
        """
        Custom preprocessing function.

        Args:
            image (numpy.ndarray): Input image
            param1 (float): First parameter
            param2 (float): Second parameter

        Returns:
            numpy.ndarray: Processed image
        """
        # Implement your custom preprocessing logic here
        processed = image.copy()

        # Example: Apply some processing
        processed = ndimage.gaussian_filter(processed, sigma=param1)
        processed = np.clip(processed * param2, 0, 65535).astype(np.uint16)

        return processed

    # Add the method to the ImagePreprocessor class
    ImagePreprocessor.my_custom_preprocessing = my_custom_preprocessing

    # Use the custom preprocessing function
    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Example 1: Using a dictionary mapping channels to functions
    config = PipelineConfig(
        reference_channels=["1", "2"],
        reference_processing={
            "1": lambda img: ImagePreprocessor.my_custom_preprocessing(img, param1=2.0, param2=1.5),
            "2": ImagePreprocessor.equalize_histogram
        }
    )

    # Example 2: Using a single function for all channels
    config = PipelineConfig(
        reference_channels=["1", "2"],
        reference_processing=ImagePreprocessor.my_custom_preprocessing
    )

    # Example 3: Using a list of functions to apply in sequence
    config = PipelineConfig(
        reference_channels=["1"],
        reference_processing=[
            ImagePreprocessor.background_subtract,
            lambda img: ImagePreprocessor.my_custom_preprocessing(img, param1=2.0, param2=1.5)
        ]
    )

    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Adding Custom Focus Detection Methods
---------------------------------

You can add custom focus detection methods to the `FocusAnalyzer` class. The FocusAnalyzer currently supports the following methods:

- `normalized_variance`: Measures the variance of pixel intensities
- `laplacian_energy`: Uses the Laplacian operator to detect edges
- `tenengrad_variance`: Based on gradient magnitude
- `adaptive_fft_focus`: Uses frequency domain analysis
- `combined_focus_measure`: Combines multiple methods with weights

Here's how to add a new focus detection method:

.. code-block:: python

    from ezstitcher.core.focus_analyzer import FocusAnalyzer
    import numpy as np
    from scipy import ndimage

    # Add a new method to FocusAnalyzer
    def gradient_magnitude_variance(self, image):
        """
        Calculate gradient magnitude variance as a focus measure.

        Args:
            image (numpy.ndarray): Input grayscale image

        Returns:
            float: Focus quality score
        """
        grad_x = ndimage.sobel(image, axis=0)
        grad_y = ndimage.sobel(image, axis=1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.var(magnitude)

    # Add the method to the FocusAnalyzer class
    FocusAnalyzer.gradient_magnitude_variance = gradient_magnitude_variance

    # Update the _get_focus_function method to include the new method
    original_get_focus_function = FocusAnalyzer._get_focus_function

    def new_get_focus_function(self, method):
        """
        Get the appropriate focus measure function based on method name.

        Args:
            method (str): Focus detection method name

        Returns:
            callable: The focus measure function

        Raises:
            ValueError: If the method is unknown
        """
        if method == 'gradient_magnitude':
            return self.gradient_magnitude_variance
        else:
            return original_get_focus_function(self, method)

    # Replace the original method
    FocusAnalyzer._get_focus_function = new_get_focus_function

    # Use the custom focus detection method
    from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Example 1: Using a single focus method
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_config=FocusAnalyzerConfig(
            method="gradient_magnitude"
        )
    )

    # Example 2: Using a combined focus method with custom weights
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_config=FocusAnalyzerConfig(
            method="combined",
            weights={
                'nvar': 0.3,
                'lap': 0.3,
                'ten': 0.2,
                'fft': 0.2
            }
        )
    )

    # Example 3: Using a focus method with a region of interest
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_config=FocusAnalyzerConfig(
            method="gradient_magnitude",
            roi=(100, 100, 200, 200)  # (x, y, width, height)
        )
    )

    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Creating a Custom Pipeline
-----------------------

You can create a custom pipeline by subclassing `PipelineOrchestrator`. The PipelineOrchestrator provides a flexible framework for processing microscopy images, with the key method `process_patterns_with_variable_components` that handles pattern detection and processing.

.. code-block:: python

    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.config import PipelineConfig
    from pathlib import Path

    class CustomPipeline(PipelineOrchestrator):
        """Custom pipeline with additional functionality."""

        def __init__(self, config=None):
            """Initialize with configuration."""
            super().__init__(config or PipelineConfig())
            # Add custom initialization here

        def run(self, plate_folder):
            """Process a plate through the custom pipeline."""
            plate_path = Path(plate_folder)

            # Add custom pre-processing steps
            self._custom_preprocessing(plate_path)

            # Call the parent implementation
            result = super().run(plate_folder)

            # Add custom post-processing steps
            self._custom_postprocessing(plate_path)

            return result

        def _custom_preprocessing(self, plate_path):
            """Custom preprocessing step."""
            # Implement your custom preprocessing logic here
            print(f"Custom preprocessing for {plate_path}")

        def _custom_postprocessing(self, plate_path):
            """Custom postprocessing step."""
            # Implement your custom postprocessing logic here
            print(f"Custom postprocessing for {plate_path}")

        def process_custom_patterns(self, well, dirs):
            """Process custom patterns for a well."""
            # Use the process_patterns_with_variable_components method
            # to process patterns with custom logic
            return self.process_patterns_with_variable_components(
                input_dir=dirs['input'],
                output_dir=dirs['processed'],
                well_filter=[well],
                variable_components=['site', 'channel'],
                group_by='z_index',
                processing_funcs=self._custom_processing_function
            )

        def _custom_processing_function(self, images, **kwargs):
            """Custom processing function for image stacks."""
            # Implement your custom processing logic here
            # This function will be called with a list of images
            # and should return a processed list of images
            return [self.image_preprocessor.normalize(img) for img in images]

    # Use the custom pipeline
    custom_pipeline = CustomPipeline()
    custom_pipeline.run("path/to/plate_folder")

    # Example with custom configuration
    config = PipelineConfig(
        reference_channels=["1", "2"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        cleanup_processed=False,  # Keep processed files
        cleanup_post_processed=False,  # Keep post-processed files
        num_workers=1  # Use single-threaded processing
    )
    custom_pipeline = CustomPipeline(config)
    custom_pipeline.run("path/to/plate_folder")
