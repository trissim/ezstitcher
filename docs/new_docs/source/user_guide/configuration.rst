Configuration System
===================

This page explains EZStitcher's configuration system.

Configuration Classes
-------------------

EZStitcher uses a hierarchical configuration system with several configuration classes:

PipelineConfig
~~~~~~~~~~~~

The main configuration class for the pipeline orchestrator:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig

    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_method="combined"
    )

Key parameters:

- **reference_channels**: Channels to use for position generation
- **reference_processing**: Preprocessing functions for reference channels
- **reference_flatten**: Method for flattening Z-stacks for position generation
- **stitch_flatten**: Method for flattening Z-stacks for final stitching
- **focus_method**: Focus detection method
- **well_filter**: List of wells to process

StitcherConfig
~~~~~~~~~~~~

Configuration for the Stitcher class:

.. code-block:: python

    from ezstitcher.core.config import StitcherConfig

    stitcher_config = StitcherConfig(
        tile_overlap=10.0,
        max_shift=50,
        margin_ratio=0.1
    )

Key parameters:

- **tile_overlap**: Percentage overlap between tiles
- **max_shift**: Maximum allowed shift in pixels
- **margin_ratio**: Ratio of image size to use as margin for blending
- **pixel_size**: Pixel size in micrometers
- **grid_size**: Grid dimensions (number of tiles in X and Y directions)

FocusAnalyzerConfig
~~~~~~~~~~~~~~~~~

Configuration for the FocusAnalyzer class:

.. code-block:: python

    from ezstitcher.core.config import FocusAnalyzerConfig

    focus_config = FocusAnalyzerConfig(
        method="combined",
        roi=(100, 100, 200, 200),
        weights={
            "nvar": 0.4,
            "lap": 0.3,
            "ten": 0.2,
            "fft": 0.1
        }
    )

Key parameters:

- **method**: Focus detection method
- **roi**: Region of interest as (x, y, width, height)
- **weights**: Weights for combined focus measure

ImagePreprocessorConfig
~~~~~~~~~~~~~~~~~~~~~

Configuration for the ImagePreprocessor class:

.. code-block:: python

    from ezstitcher.core.config import ImagePreprocessorConfig

    preprocessor_config = ImagePreprocessorConfig(
        preprocessing_funcs={
            "1": lambda img: ImagePreprocessor.equalize_histogram(img),
            "2": lambda img: ImagePreprocessor.background_subtract(img, radius=50)
        },
        composite_weights={
            "1": 0.7,
            "2": 0.3
        }
    )

Key parameters:

- **preprocessing_funcs**: Dictionary mapping channels to preprocessing functions
- **composite_weights**: Dictionary mapping channels to weights for composite images

Configuration Presets
-------------------

EZStitcher includes several configuration presets for common use cases:

.. code-block:: python

    from ezstitcher.core import process_plate_folder_with_config

    # Process using a predefined configuration preset
    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_preset='z_stack_best_focus'
    )

Available presets:

- **default**: Basic stitching with default parameters
- **z_stack_best_focus**: Z-stack processing using best focus
- **z_stack_per_plane**: Z-stack processing with per-plane stitching
- **high_resolution**: High-resolution stitching with smaller overlap

Configuration Files (JSON/YAML)
-----------------------------

You can save and load configurations to/from files:

Saving a configuration:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    import json

    config = PipelineConfig(
        reference_channels=["1", "2"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus"
    )

    # Save to JSON
    with open("my_config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

Loading a configuration:

.. code-block:: python

    from ezstitcher.core import process_plate_folder_with_config

    # Process using a configuration file
    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_file='my_config.json'
    )

Configuration Validation
----------------------

EZStitcher validates configuration parameters to ensure they are valid:

- **Type checking**: Ensures parameters have the correct type
- **Value validation**: Ensures parameters have valid values
- **Default values**: Provides sensible defaults for optional parameters
- **Required parameters**: Ensures required parameters are provided

If a configuration parameter is invalid, EZStitcher will raise an error with a helpful message.

Configuration Inheritance
----------------------

You can create new configurations by inheriting from existing ones:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig

    # Create a base configuration
    base_config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection"
    )

    # Create a derived configuration
    derived_config = PipelineConfig(
        **base_config.__dict__,  # Inherit all base config properties
        reference_channels=["1", "2"],  # Override reference channels
        stitch_flatten="best_focus"      # Add stitch_flatten
    )

This allows you to create specialized configurations based on existing ones.
