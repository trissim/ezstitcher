Custom Preprocessing
===================

This example demonstrates how to use custom preprocessing functions with EZStitcher.

Built-in Preprocessing Functions
-----------------------------

EZStitcher provides several built-in preprocessing functions:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Create configuration with built-in preprocessing functions
    config = PipelineConfig(
        reference_channels=["1", "2"],
        reference_processing={
            "1": ImagePreprocessor.background_subtract,  # Background subtraction for channel 1
            "2": ImagePreprocessor.equalize_histogram    # Histogram equalization for channel 2
        }
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Custom Preprocessing Functions
---------------------------

You can define your own preprocessing functions:

.. code-block:: python

    import numpy as np
    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Define custom preprocessing function
    def enhance_contrast(image):
        """Enhance contrast using percentile normalization."""
        p_low, p_high = np.percentile(image, (2, 98))
        return np.clip((image - p_low) * (65535 / (p_high - p_low)), 0, 65535).astype(np.uint16)

    # Create configuration with custom preprocessing function
    config = PipelineConfig(
        reference_channels=["1"],
        reference_processing={
            "1": enhance_contrast
        }
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Chaining Preprocessing Functions
-----------------------------

You can chain multiple preprocessing functions:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Create configuration with chained preprocessing functions
    config = PipelineConfig(
        reference_channels=["1"],
        reference_processing={
            "1": [
                ImagePreprocessor.background_subtract,  # First apply background subtraction
                ImagePreprocessor.equalize_histogram    # Then apply histogram equalization
            ]
        }
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Different Processing for Reference and Final Images
-----------------------------------------------

You can use different preprocessing for reference and final images:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Create configuration with different preprocessing for reference and final images
    config = PipelineConfig(
        reference_channels=["1"],
        reference_processing={
            "1": ImagePreprocessor.background_subtract  # For position generation
        },
        final_processing={
            "1": ImagePreprocessor.equalize_histogram,  # For final stitching (channel 1)
            "2": ImagePreprocessor.normalize,           # For final stitching (channel 2)
            "3": ImagePreprocessor.percentile_normalize # For final stitching (channel 3)
        }
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Stack Processing Functions
-----------------------

EZStitcher supports both single-image and stack-processing functions:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Create configuration with stack processing functions
    config = PipelineConfig(
        reference_channels=["1", "2"],
        reference_processing={
            "1": ImagePreprocessor.equalize_histogram,      # Single-image function
            "2": ImagePreprocessor.stack_equalize_histogram # Stack-processing function
        }
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/zstack_plate")

Dynamic Preprocessing Based on Image Properties
-------------------------------------------

You can dynamically select preprocessing functions based on image properties:

.. code-block:: python

    import numpy as np
    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.image_preprocessor import ImagePreprocessor
    from ezstitcher.core.file_system_manager import FileSystemManager
    from pathlib import Path

    def analyze_and_configure(plate_folder):
        """Analyze images and create appropriate configuration."""
        # Find a sample image
        fs_manager = FileSystemManager()
        sample_files = fs_manager.list_image_files(Path(plate_folder))
        if not sample_files:
            return PipelineConfig(reference_channels=["1"])
            
        sample_image = fs_manager.load_image(sample_files[0])
        
        # Analyze image properties
        mean_intensity = np.mean(sample_image)
        std_intensity = np.std(sample_image)
        
        # Determine optimal parameters based on image properties
        if std_intensity / mean_intensity < 0.2:
            # Low contrast image - use contrast enhancement
            preprocessing_func = ImagePreprocessor.equalize_histogram
        else:
            # Normal contrast - use background subtraction
            preprocessing_func = lambda img: ImagePreprocessor.background_subtract(img, radius=50)
        
        # Create configuration with dynamic parameters
        config = PipelineConfig(
            reference_channels=["1"],
            reference_processing=preprocessing_func
        )
        
        return config

    # Analyze images and create configuration
    config = analyze_and_configure("path/to/plate_folder")

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")
