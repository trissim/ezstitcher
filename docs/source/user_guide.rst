User Guide
==========

This guide provides detailed information on how to use EZStitcher for various microscopy image processing tasks.

Basic Concepts
--------------

EZStitcher is designed to process microscopy images, particularly those from high-content screening platforms like ImageXpress. It handles:

- **Plate-based experiments**: Organized by wells (e.g., A01, A02, etc.)
- **Multi-channel fluorescence**: Different wavelengths/channels for different stains
- **Z-stacks**: 3D image stacks with multiple focal planes
- **Tiled images**: Multiple images that need to be stitched together

File Organization
-----------------

EZStitcher expects a specific file organization:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── A01_s001_w1.tif
    │   ├── A01_s001_w2.tif
    │   ├── A01_s002_w1.tif
    │   ├── A01_s002_w2.tif
    │   ├── ...
    │   ├── B01_s001_w1.tif
    │   └── ...
    └── ...

For Z-stacks, there are two supported formats:

1. **Folder-based Z-stacks**:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── ZStep_1/
    │   │   ├── A01_s001_w1.tif
    │   │   ├── A01_s001_w2.tif
    │   │   └── ...
    │   ├── ZStep_2/
    │   │   ├── A01_s001_w1.tif
    │   │   ├── A01_s001_w2.tif
    │   │   └── ...
    │   └── ...
    └── ...

2. **Filename-based Z-stacks**:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── A01_s001_w1_z001.tif
    │   ├── A01_s001_w1_z002.tif
    │   ├── A01_s001_w2_z001.tif
    │   ├── A01_s001_w2_z002.tif
    │   └── ...
    └── ...

Output Organization
-------------------

EZStitcher creates several output directories:

.. code-block:: text

    parent_directory/
    ├── plate_folder/
    │   └── ...
    ├── plate_folder_processed/
    │   └── ...
    ├── plate_folder_positions/
    │   └── ...
    ├── plate_folder_stitched/
    │   └── ...
    ├── plate_folder_best_focus/
    │   └── ...
    └── plate_folder_projections/
        └── ...

Basic Stitching
---------------

For basic stitching of non-Z-stack data:

1. **Command Line**:

.. code-block:: bash

    ezstitcher /path/to/plate_folder --reference-channels 1 --tile-overlap 10

2. **Python API**:

.. code-block:: python

    from ezstitcher.core import process_plate_folder

    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1"],
        tile_overlap=10
    )

Z-Stack Processing
------------------

EZStitcher provides several options for Z-stack processing:

1. **Best Focus Detection**:

.. code-block:: python

    process_plate_folder(
        'path/to/plate_folder',
        focus_detect=True,
        focus_method="combined"
    )

2. **Z-Stack Projections**:

.. code-block:: python

    process_plate_folder(
        'path/to/plate_folder',
        create_projections=True,
        projection_types=["max", "mean"]
    )

3. **Per-Plane Z-Stack Stitching**:

.. code-block:: python

    process_plate_folder(
        'path/to/plate_folder',
        create_projections=True,
        stitch_z_reference="max",
        stitch_all_z_planes=True
    )

Focus Detection Methods
-----------------------

EZStitcher supports several focus detection methods:

- **combined**: A weighted combination of multiple metrics (default)
- **laplacian**: Based on the Laplacian operator (edge detection)
- **normalized_variance**: Based on image variance
- **tenengrad**: Based on the Tenengrad operator (gradient-based)
- **fft**: Based on frequency domain analysis
- **adaptive_fft**: Adaptive FFT-based method

You can specify the method using the `focus_method` parameter:

.. code-block:: python

    process_plate_folder(
        'path/to/plate_folder',
        focus_detect=True,
        focus_method="laplacian"
    )

Projection Types
----------------

EZStitcher supports several projection types:

- **max**: Maximum intensity projection (default)
- **mean**: Mean intensity projection
- **std**: Standard deviation projection
- **median**: Median intensity projection
- **min**: Minimum intensity projection

You can specify the projection types using the `projection_types` parameter:

.. code-block:: python

    process_plate_folder(
        'path/to/plate_folder',
        create_projections=True,
        projection_types=["max", "mean", "std"]
    )

Configuration Management
------------------------

EZStitcher provides a flexible configuration system:

1. **Using Configuration Presets**:

.. code-block:: python

    from ezstitcher.core import process_plate_folder_with_config

    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_preset='z_stack_best_focus'
    )

2. **Using Configuration Files**:

.. code-block:: python

    process_plate_folder_with_config(
        'path/to/plate_folder',
        config_file='path/to/config.json'
    )

3. **Creating Custom Configurations**:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig, FocusAnalyzerConfig

    config = PipelineConfig(
        reference_channels=["1", "2"],
        stitcher=StitcherConfig(
            tile_overlap=15.0,
            max_shift=75
        ),
        focus_config=FocusAnalyzerConfig(
            method="combined"
        ),
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        additional_projections=["max", "mean"]
    )

    config.to_json("my_config.json")

Troubleshooting
---------------

Common issues and solutions:

1. **No images found**: Check that the file organization matches what EZStitcher expects.

2. **Stitching fails**: Try increasing the `max_shift` parameter if tiles are far apart.

3. **Poor focus detection**: Try different focus methods or specify an ROI.

4. **Memory errors**: Process one well at a time using the `well_filter` parameter.

5. **Slow processing**: Use the `reference_channels` parameter to limit processing to specific channels.
