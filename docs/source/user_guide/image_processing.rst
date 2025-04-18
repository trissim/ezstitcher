Image Processing Pipeline
=======================

This page explains EZStitcher's image processing pipeline.

Pipeline Stages
-------------

The image processing pipeline consists of several stages:

1. **Tile Processing**: Process individual tiles
2. **Channel Selection/Composition**: Select or compose channels for position generation
3. **Z-Stack Flattening for Position Generation**: Flatten Z-stacks for position generation
4. **Position Generation**: Generate stitching positions
5. **Final Processing**: Process images for final stitching (can preserve individual channels and Z-planes)
6. **Stitching**: Stitch images

Each stage is configurable and can be customized for specific needs. The pipeline is designed to be flexible, allowing you to process images in different ways for position generation and final stitching.

Tile Processing
-------------

Tile processing applies preprocessing functions to individual tiles:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Define preprocessing functions
    def enhance_contrast(image):
        """Enhance contrast using histogram equalization."""
        return ImagePreprocessor.equalize_histogram(image)

    def denoise(image):
        """Apply denoising."""
        return ImagePreprocessor.blur(image, sigma=1)

    # Create configuration with preprocessing functions
    config = PipelineConfig(
        reference_channels=["1", "2"],
        reference_processing={
            "1": enhance_contrast,
            "2": denoise
        }
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Available preprocessing functions:

- **blur**: Apply Gaussian blur
- **normalize**: Normalize image to specified range
- **equalize_histogram**: Apply histogram equalization
- **background_subtract**: Subtract background
- **apply_mask**: Apply a mask to an image
- **create_weight_mask**: Create a weight mask for blending

You can also define custom preprocessing functions:

.. code-block:: python

    import numpy as np

    def custom_preprocess(image):
        """Custom preprocessing function."""
        # Apply custom processing
        # Must return an image of the same shape as the input
        return image

Preprocessing functions are applied to individual tiles before stitching.

Channel Selection/Composition
---------------------------

Channel selection/composition allows you to select specific channels or create composite images for position generation:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with channel selection
    config = PipelineConfig(
        reference_channels=["1", "2"],  # Use channels 1 and 2 for position generation
        reference_composite_weights={
            "1": 0.7,
            "2": 0.3
        }  # Create a composite image with these weights
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Channel selection options:

- **reference_channels**: Channels to use for position generation
- **reference_composite_weights**: Weights for creating composite images

If reference_composite_weights is provided, EZStitcher will create a composite image from the specified channels. Otherwise, it will process each channel separately.

Z-Stack Flattening
----------------

Z-stack flattening combines multiple Z-planes into a single image. For position generation, Z-stacks must be flattened to create a single reference image per well. For final stitching, you can either flatten Z-stacks or preserve individual Z-planes:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with Z-stack flattening
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",  # Use max projection for position generation
        stitch_flatten="best_focus",         # Use best focus for final stitching
        focus_method="combined"              # Use combined focus metric
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Z-stack flattening options:

- **reference_flatten**: Method for flattening Z-stacks for position generation (required)
- **stitch_flatten**: Method for flattening Z-stacks for final stitching (optional)

Available flattening methods:

- **max_projection**: Maximum intensity projection
- **mean_projection**: Mean intensity projection
- **best_focus**: Select the best focused plane
- **None**: Process each Z-plane separately (only valid for stitch_flatten)

When stitch_flatten is set to None, EZStitcher will preserve individual Z-planes in the final stitched output, creating separate stitched images for each Z-plane. This is useful for preserving 3D information in the stitched output.

You can also define custom flattening functions:

.. code-block:: python

    def custom_flatten(stack):
        """Custom Z-stack flattening function."""
        # Apply custom flattening
        # Must return a single image
        import numpy as np
        return np.mean(stack, axis=0)  # Example: mean projection

Position Generation
-----------------

Position generation calculates the relative positions of tiles for stitching:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with stitching parameters
    config = PipelineConfig(
        reference_channels=["1"],
        stitcher=StitcherConfig(
            tile_overlap=10.0,  # 10% overlap between tiles
            max_shift=50,       # Maximum allowed shift in pixels
            margin_ratio=0.1    # Ratio of image size to use as margin for blending
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Position generation parameters:

- **tile_overlap**: Percentage overlap between tiles
- **max_shift**: Maximum allowed shift in pixels
- **margin_ratio**: Ratio of image size to use as margin for blending

The position generation process:

1. Load reference images
2. Calculate relative positions using cross-correlation
3. Refine positions to subpixel precision
4. Save positions to CSV files

Stitching
--------

Stitching combines multiple tiles into a single image:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with stitching parameters
    config = PipelineConfig(
        reference_channels=["1"],
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

The stitching process:

1. Load images
2. Load positions from CSV files
3. Create a canvas large enough to hold all tiles
4. Place tiles on the canvas according to their positions
5. Blend overlapping regions
6. Save the stitched image

Stitching parameters:

- **tile_overlap**: Percentage overlap between tiles
- **max_shift**: Maximum allowed shift in pixels
- **margin_ratio**: Ratio of image size to use as margin for blending

Complete Pipeline Example
-----------------------

Here's a complete example of the image processing pipeline:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Define preprocessing functions
    def enhance_contrast(image):
        """Enhance contrast using histogram equalization."""
        return ImagePreprocessor.equalize_histogram(image)

    def denoise(image):
        """Apply denoising."""
        return ImagePreprocessor.blur(image, sigma=1)

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1", "2"],
        reference_processing={
            "1": enhance_contrast,
            "2": denoise
        },
        reference_composite_weights={
            "1": 0.7,
            "2": 0.3
        },
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_method="combined",
        focus_config=FocusAnalyzerConfig(
            method="combined",
            roi=None,  # Use entire image
            weights={
                "nvar": 0.4,
                "lap": 0.3,
                "ten": 0.2,
                "fft": 0.1
            }
        ),
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

This example:

1. Applies different preprocessing functions to channels 1 and 2
2. Creates a composite image with weights 0.7 and 0.3
3. Uses max projection for position generation
4. Uses best focus for final stitching
5. Uses combined focus metric with custom weights
6. Configures stitching parameters
