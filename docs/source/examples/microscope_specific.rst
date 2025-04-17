Microscope-Specific Examples
=========================

This example demonstrates how to use EZStitcher with specific microscope types.

ImageXpress
----------

### ImageXpress File Structure

ImageXpress microscopes typically organize files in the following structure:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── A01_s1_w1.tif  # Well A01, Site 1, Channel 1
    │   ├── A01_s1_w2.tif  # Well A01, Site 1, Channel 2
    │   ├── A01_s2_w1.tif  # Well A01, Site 2, Channel 1
    │   └── ...
    └── ...

For Z-stacks, the structure is typically:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── ZStep_1/
    │   │   ├── A01_s1_w1.tif
    │   │   └── ...
    │   ├── ZStep_2/
    │   │   ├── A01_s1_w1.tif
    │   │   └── ...
    │   └── ...
    └── ...

### ImageXpress Basic Stitching

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration for ImageXpress
    config = PipelineConfig(
        reference_channels=["1"],
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/imagexpress_plate")

### ImageXpress Z-Stacks

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration for ImageXpress Z-stacks
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_method="combined"
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/imagexpress_zstack_plate")

Opera Phenix
-----------

### Opera Phenix File Structure

Opera Phenix microscopes typically organize files in the following structure:

.. code-block:: text

    plate_folder/
    ├── Images/
    │   ├── r01c01f01p01-ch1sk1fk1fl1.tif  # Well A01, Channel 1, Field 1, Plane 1 
    │   ├── r01c01f01p02-ch1sk1fk1fl1.tif  # Well A01, Channel 1, Field 1, Plane 2 
    │   ├── r01c01f02p01-ch1sk1fk1fl1.tif  # Well A01, Channel 1, Field 2, Plane 1 
    │   └── ...
    ├── Index.xml
    └── ...

### Opera Phenix Basic Stitching

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration for Opera Phenix
    config = PipelineConfig(
        reference_channels=["1"],  # Channel 1 (K1 in Opera Phenix)
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/opera_phenix_plate")

### Opera Phenix Z-Stacks

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration for Opera Phenix Z-stacks
    config = PipelineConfig(
        reference_channels=["1"],  # Channel 1 (K1 in Opera Phenix)
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_method="combined"
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/opera_phenix_zstack_plate")

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

Accessing Microscope-Specific Metadata
-----------------------------------

You can access microscope-specific metadata:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import MicroscopeHandler
    from pathlib import Path

    # Create a microscope handler
    handler = MicroscopeHandler(plate_folder=Path("path/to/plate_folder"))

    # Get grid dimensions
    grid_size_x, grid_size_y = handler.get_grid_dimensions(Path("path/to/plate_folder"))
    print(f"Grid dimensions: {grid_size_x}x{grid_size_y}")

    # Get pixel size
    pixel_size = handler.get_pixel_size(Path("path/to/plate_folder"))
    print(f"Pixel size: {pixel_size} µm")

    # Find metadata file
    metadata_file = handler.find_metadata_file(Path("path/to/plate_folder"))
    print(f"Metadata file: {metadata_file}")

Working with Different Microscope Types
------------------------------------

You can explicitly specify the microscope type:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.microscope_interfaces import MicroscopeHandler

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1"]
    )

    # Create pipeline
    pipeline = PipelineOrchestrator(config)

    # Explicitly set microscope handler
    pipeline.microscope_handler = MicroscopeHandler(
        plate_folder="path/to/plate_folder",
        microscope_type="OperaPhenix"  # Explicitly specify microscope type
    )

    # Run pipeline
    pipeline.run("path/to/plate_folder")
