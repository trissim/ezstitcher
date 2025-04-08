Quickstart
==========

This guide will help you get started with EZStitcher quickly.

Command Line Usage
-----------------

EZStitcher can be used from the command line for common tasks:

.. code-block:: bash

    # Process a plate folder
    ezstitcher /path/to/plate_folder --reference-channels 1 2 --tile-overlap 10

    # Process a plate folder with Z-stacks
    ezstitcher /path/to/plate_folder --focus-detect --focus-method combined

    # Create Z-stack projections
    ezstitcher /path/to/plate_folder --create-projections --projection-types max,mean,std

    # Full Z-stack workflow with best focus detection and stitching
    ezstitcher /path/to/plate_folder --focus-detect --create-projections --stitch-method best_focus

Basic Python Usage
-----------------

For basic usage in Python, you can use the `process_plate_folder` function:

.. code-block:: python

    from ezstitcher.core import process_plate_folder

    # Process a single plate folder
    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1", "2"],
        tile_overlap=10,
        max_shift=50
    )

Z-Stack Processing
-----------------

For Z-stack processing, you can use additional parameters:

.. code-block:: python

    from ezstitcher.core import process_plate_folder

    # Process Z-stack data with focus detection and projections
    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1"],
        tile_overlap=10,
        focus_detect=True,                # Enable best focus detection for Z-stacks
        focus_method="combined",          # Use combined focus metrics
        create_projections=True,          # Create Z-stack projections
        projection_types=["max", "mean"], # Types of projections to create
        stitch_z_reference="max",         # Use max projection images for stitching
        stitch_all_z_planes=True          # Stitch all Z-planes using projection-derived positions
    )

Configuration-Based Usage
------------------------

For more advanced usage, you can use the configuration-based API:

.. code-block:: python

    from ezstitcher.core import process_plate_folder_with_config

    # Process using a predefined configuration preset
    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_preset='z_stack_best_focus'
    )

    # Process using a configuration file
    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_file='path/to/config.json'
    )

    # Process with configuration overrides
    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_preset='default',
        reference_channels=["1", "2"],
        well_filter=["A01", "A02"]
    )

Object-Oriented Usage
--------------------

For more control, you can use the object-oriented API:

.. code-block:: python

    from ezstitcher.core.config import (
        StitcherConfig,
        FocusAnalyzerConfig,
        ImagePreprocessorConfig,
        ZStackProcessorConfig,
        PlateProcessorConfig
    )
    from ezstitcher.core.plate_processor import PlateProcessor

    # Create configuration objects
    stitcher_config = StitcherConfig(
        tile_overlap=10.0,
        max_shift=50,
        margin_ratio=0.1
    )

    focus_config = FocusAnalyzerConfig(
        method="combined"
    )

    zstack_config = ZStackProcessorConfig(
        focus_detect=True,
        focus_method="combined",
        create_projections=True,
        stitch_z_reference="max",
        save_projections=True,
        stitch_all_z_planes=True,
        projection_types=["max", "mean"]
    )

    plate_config = PlateProcessorConfig(
        reference_channels=["1", "2"],
        well_filter=["A01", "A02"],
        stitcher=stitcher_config,
        focus_analyzer=focus_config,
        z_stack_processor=zstack_config
    )

    # Create and run the plate processor
    processor = PlateProcessor(plate_config)
    processor.run("path/to/plate_folder")
