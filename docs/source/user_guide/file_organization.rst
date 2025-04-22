File Organization
================

This page explains how EZStitcher organizes and processes files.

Input Directory Structure
-----------------------

EZStitcher supports various input directory structures, with automatic detection of microscope types and file organizations.

Basic Structure
~~~~~~~~~~~~~

The basic input structure is a plate folder containing image files. EZStitcher automatically detects the microscope type based on the file naming and directory structure.

For detailed information about supported microscope formats, see the :doc:`../appendices/microscope_formats` appendix.

Output Directory Structure
------------------------

EZStitcher creates several output directories during processing:

.. code-block:: text

    plate_folder/                 # Original data
    plate_folder_processed/       # Processed individual tiles
    plate_folder_post_processed/  # Post-processed images
    plate_folder_positions/       # CSV files with stitching positions
    plate_folder_stitched/        # Final stitched images

Each directory serves a specific purpose in the processing pipeline:

Processed Directory
~~~~~~~~~~~~~~~~~

Contains processed individual tiles after applying preprocessing functions:

.. code-block:: text

    plate_folder_processed/
    ├── A01/
    │   ├── A01_s1_w1.tif
    │   ├── A01_s1_w2.tif
    │   ├── A01_s2_w1.tif
    │   └── ...
    ├── A02/
    │   └── ...
    └── ...

Processing operations may include:
- Background subtraction
- Contrast enhancement
- Noise reduction
- Custom preprocessing functions

Post-Processed Directory
~~~~~~~~~~~~~~~~~~~~~~

Contains post-processed images after channel selection/composition and Z-stack flattening:

.. code-block:: text

    plate_folder_post_processed/
    ├── A01/
    │   ├── A01_w1.tif
    │   ├── A01_w2.tif
    │   └── ...
    ├── A02/
    │   └── ...
    └── ...

Post-processing operations may include:
- Z-stack flattening (max projection, best focus, etc.)
- Channel composition
- Reference channel selection

Positions Directory
~~~~~~~~~~~~~~~~~

Contains CSV files with stitching positions for each well and channel:

.. code-block:: text

    plate_folder_positions/
    ├── A01_w1.csv
    ├── A01_w2.csv
    ├── A02_w1.csv
    └── ...

Each CSV file contains the positions of tiles for a specific well and channel:

.. code-block:: text

    filename,x,y
    A01_s1_w1.tif,0.0,0.0
    A01_s2_w1.tif,1024.5,0.0
    A01_s3_w1.tif,2049.2,0.0
    A01_s4_w1.tif,0.0,1024.3
    ...

Stitched Directory
~~~~~~~~~~~~~~~~

Contains final stitched images for each well and channel:

.. code-block:: text

    plate_folder_stitched/
    ├── A01_w1.tif
    ├── A01_w2.tif
    ├── A02_w1.tif
    └── ...

Well-Based Organization
---------------------

EZStitcher processes images on a per-well basis. Each well is processed independently, allowing for parallel processing and efficient memory usage.

For each well, EZStitcher:

1. Finds all images for that well
2. Processes them according to the configuration
3. Generates stitching positions
4. Stitches the images
5. Saves the results

Z-Stack Organization
-----------------

EZStitcher supports different Z-stack organizations:

- **Folder-based**: Z-planes organized in separate folders
- **Filename-based**: Z-plane index included in the filename

For details on how EZStitcher handles Z-stacks, see the :doc:`zstack_handling` guide.

Custom File Organization
---------------------

If your files don't match the standard patterns, you can customize how EZStitcher finds and processes them:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Custom file pattern
    config = PipelineConfig(
        file_pattern="custom_{well}_site{site}_channel{channel}.tif"
    )

    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

For more advanced customization, see the :doc:`../api/microscope_interfaces` API reference.

File Formats
-----------

For detailed information about supported file formats, see the :doc:`../appendices/file_formats` appendix.
