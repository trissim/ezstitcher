Microscope Support
=================

This page explains how to use EZStitcher with different microscope types. For detailed information about file formats and directory structures, see the :doc:`../appendices/microscope_formats` appendix.

Supported Microscope Types
------------------------

EZStitcher currently supports the following microscope types:

- **ImageXpress**: Molecular Devices ImageXpress microscopes
- **Opera Phenix**: PerkinElmer Opera Phenix microscopes
- **Auto**: Automatic detection based on file naming and directory structure (recommended)

Using Microscope Handlers
----------------------

EZStitcher provides a unified interface for working with different microscope types through the ``MicroscopeHandler`` class:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import MicroscopeHandler

    # Create a handler with auto-detection (recommended)
    handler = MicroscopeHandler(plate_folder="path/to/plate_folder")

    # Or specify a microscope type explicitly
    handler = MicroscopeHandler(plate_folder="path/to/plate_folder", microscope_type="ImageXpress")
    handler = MicroscopeHandler(plate_folder="path/to/plate_folder", microscope_type="OperaPhenix")

The ``MicroscopeHandler`` provides methods for:

- Parsing filenames
- Extracting metadata
- Finding image files
- Organizing Z-stacks

Auto-Detection
------------

In most cases, you can rely on EZStitcher's auto-detection capability:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import MicroscopeHandler
    from pathlib import Path

    # Create a handler with auto-detection
    plate_folder = Path("path/to/plate_folder")
    handler = MicroscopeHandler(plate_folder=plate_folder)

    # Print the detected microscope type
    print(f"Detected microscope type: {handler.microscope_type}")

The auto-detection process examines the directory structure, checks for characteristic files, and analyzes file naming patterns to determine the most likely microscope type.

Working with ImageXpress Data
--------------------------

For ImageXpress data, EZStitcher automatically handles:

- Well, site, and channel identification from filenames
- Z-stack organization (both folder-based and suffix-based)
- Metadata extraction from HTD files

Example workflow:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1"],  # Use channel 1 as reference
        well_filter=["A01", "A02"]  # Process only wells A01 and A02
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/imagexpress_plate")

Working with Opera Phenix Data
---------------------------

For Opera Phenix data, EZStitcher automatically handles:

- Well, field, channel, and plane identification from filenames
- Z-stack organization based on plane identifiers
- Metadata extraction from XML files
- Field position mapping from metadata

Example workflow:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1"],  # Use channel 1 as reference
        well_filter=["A01", "A02"]  # Process only wells A01 and A02
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/opera_phenix_plate")

Accessing Microscope-Specific Functionality
---------------------------------------

If you need to access microscope-specific functionality, you can use the ``MicroscopeHandler`` directly:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import MicroscopeHandler
    from pathlib import Path

    # Create a handler
    plate_folder = Path("path/to/plate_folder")
    handler = MicroscopeHandler(plate_folder=plate_folder)

    # Parse a filename
    components = handler.parse_filename("A01_s1_w1.tif")
    print(f"Well: {components['well']}, Site: {components['site']}, Channel: {components['channel']}")

    # Get grid dimensions
    grid_dimensions = handler.get_grid_dimensions()
    print(f"Grid dimensions: {grid_dimensions}")

    # Get pixel size
    pixel_size = handler.get_pixel_size()
    print(f"Pixel size: {pixel_size} µm")

Customizing Microscope Support
---------------------------

If you need to customize how EZStitcher handles a specific microscope type, you can provide custom configuration:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with custom file pattern
    config = PipelineConfig(
        file_pattern="{well}_site{site}_channel{channel}.tif",  # Custom file pattern
        reference_channels=["1"]
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/custom_plate")

For information on adding support for new microscope types, see the :doc:`../development/extending` guide.
