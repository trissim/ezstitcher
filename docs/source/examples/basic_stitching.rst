Basic Stitching
==============

This example demonstrates how to stitch a plate of microscopy images.

Command Line
-----------

.. code-block:: bash

    # Process a plate folder with basic stitching
    ezstitcher /path/to/plate_folder --reference-channels 1 --tile-overlap 10

Python API
---------

.. code-block:: python

    from ezstitcher.core.main import process_plate_auto

    # Process a plate folder with basic stitching (auto-detects microscope type)
    process_plate_auto(
        'path/to/plate_folder',
        microscope_type="auto",  # This is the default, so you can omit it
        **{
            "reference_channels": ["1"],
            "stitcher.tile_overlap": 10,
            "stitcher.max_shift": 50
        }
    )

Object-Oriented API
-----------------

.. code-block:: python

    from ezstitcher.core.config import StitcherConfig, PlateProcessorConfig
    from ezstitcher.core.plate_processor import PlateProcessor

    # Create configuration objects
    stitcher_config = StitcherConfig(
        tile_overlap=10.0,
        max_shift=50,
        margin_ratio=0.1
    )

    plate_config = PlateProcessorConfig(
        reference_channels=["1"],
        stitcher=stitcher_config
    )

    # Create and run the plate processor
    processor = PlateProcessor(plate_config)
    processor.run("path/to/plate_folder")
