Custom Configuration
===================

This example demonstrates how to use custom configurations with EZStitcher.

Using Configuration Presets
-------------------------

.. code-block:: python

    from ezstitcher.core import process_plate_folder_with_config

    # Process using a predefined configuration preset
    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_preset='z_stack_best_focus'
    )

    # Process using a different preset
    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_preset='z_stack_per_plane'
    )

    # Process using the high-resolution preset
    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_preset='high_resolution'
    )

Creating and Saving Configuration Files
-------------------------------------

.. code-block:: python

    from ezstitcher.core.pydantic_config import (
        PlateProcessorConfig,
        StitcherConfig,
        ZStackProcessorConfig
    )

    # Create a custom configuration
    config = PlateProcessorConfig(
        reference_channels=["1", "2"],
        well_filter=["A01", "A02"],
        stitcher=StitcherConfig(
            tile_overlap=15.0,
            max_shift=75,
            margin_ratio=0.15
        ),
        z_stack_processor=ZStackProcessorConfig(
            focus_detect=True,
            focus_method="laplacian",
            create_projections=True,
            stitch_z_reference="max",
            projection_types=["max", "mean"]
        )
    )

    # Save to JSON
    config.to_json("my_config.json")

    # Save to YAML
    config.to_yaml("my_config.yaml")

Loading Configuration Files
-------------------------

.. code-block:: python

    from ezstitcher.core import process_plate_folder_with_config

    # Process using a JSON configuration file
    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_file='my_config.json'
    )

    # Process using a YAML configuration file
    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_file='my_config.yaml'
    )

Overriding Configuration Values
-----------------------------

.. code-block:: python

    from ezstitcher.core import process_plate_folder_with_config

    # Process with configuration overrides
    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_preset='default',
        reference_channels=["2"],
        well_filter=["A03", "A04"],
        tile_overlap=12.5
    )
