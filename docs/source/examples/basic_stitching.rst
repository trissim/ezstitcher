Basic Stitching
==============

This example demonstrates how to stitch a plate of microscopy images.

Object-Oriented API
-----------------

The recommended way to use EZStitcher is through the object-oriented API, which provides the most flexibility and control:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1", "2"],  # Use channels 1 and 2 as reference
        well_filter=["A01", "B02"],    # Only process these wells
        stitcher=StitcherConfig(
            tile_overlap=10.0,         # 10% overlap between tiles
            max_shift=50               # Maximum shift in pixels
        )
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Command Line Interface
--------------------

EZStitcher also provides a command-line interface for quick tasks:

.. code-block:: bash

    # Basic stitching
    ezstitcher /path/to/plate_folder --reference-channels 1

    # Stitching with custom overlap
    ezstitcher /path/to/plate_folder --reference-channels 1 --tile-overlap 15 --max-shift 75

    # Stitching with well filtering
    ezstitcher /path/to/plate_folder --reference-channels 1 --wells A01 A02 B01 B02

Multi-Channel Stitching
---------------------

EZStitcher can use multiple channels for position generation and stitch all available channels:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with multiple reference channels
    config = PipelineConfig(
        reference_channels=["1", "2"],  # Use channels 1 and 2 as reference
        reference_composite_weights={   # Weights for creating composite reference
            "1": 0.7,
            "2": 0.3
        },
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50
        )
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Automatic Microscope Detection
---------------------------

EZStitcher can automatically detect the microscope type based on the file structure and naming conventions:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with default settings
    config = PipelineConfig(
        reference_channels=["1"]
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")  # Microscope type will be auto-detected
