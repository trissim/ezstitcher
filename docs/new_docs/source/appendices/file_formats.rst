File Formats
============

This page describes the file formats used by EZStitcher.

Image Formats
-----------

EZStitcher supports the following image formats:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Format
     - Extensions
     - Description
   * - TIFF
     - .tif, .tiff
     - Tagged Image File Format, the primary format for microscopy images. EZStitcher works with 8-bit, 16-bit, and 32-bit TIFF images.
   * - JPEG
     - .jpg, .jpeg
     - Joint Photographic Experts Group format, a compressed image format. Not recommended for scientific images due to lossy compression.
   * - PNG
     - .png
     - Portable Network Graphics format, a lossless compressed image format.

Positions CSV Format
-----------------

EZStitcher uses a CSV file to store tile positions for stitching. The format is:

.. code-block:: text

    file: <filename>; grid: (col, row); position: (x, y)

For example:

.. code-block:: text

    file: A01_s1_w1.tif; grid: (0, 0); position: (0.0, 0.0)
    file: A01_s2_w1.tif; grid: (1, 0); position: (1024.5, 0.0)
    file: A01_s3_w1.tif; grid: (0, 1); position: (0.0, 1024.5)
    file: A01_s4_w1.tif; grid: (1, 1); position: (1024.5, 1024.5)

The fields are:

- **file**: The filename of the tile
- **grid**: The grid position of the tile as (column, row)
- **position**: The subpixel position of the tile as (x, y) in pixels

Configuration Format
-----------------

EZStitcher uses Python objects for configuration, but these can be serialized to JSON for storage and sharing:

.. code-block:: json

    {
      "reference_channels": ["1", "2"],
      "well_filter": ["A01", "A02"],
      "stitcher": {
        "tile_overlap": 15.0,
        "max_shift": 75,
        "margin_ratio": 0.15
      },
      "focus_config": {
        "method": "laplacian",
        "roi": [100, 100, 200, 200]
      },
      "reference_flatten": "max_projection",
      "stitch_flatten": "best_focus",
      "additional_projections": ["max", "mean"]
    }

The configuration can be loaded from JSON:

.. code-block:: python

    import json
    from ezstitcher.core.config import PipelineConfig

    # Load from JSON
    with open("config.json", "r") as f:
        config_dict = json.load(f)
        config = PipelineConfig(**config_dict)

Metadata Formats
-------------

EZStitcher extracts metadata from microscope-specific files:

ImageXpress Metadata
^^^^^^^^^^^^^^^^^

ImageXpress metadata is stored in XML files with the following structure:

.. code-block:: xml

    <MetaData>
      <PlateType>
        <SiteRows>3</SiteRows>
        <SiteColumns>3</SiteColumns>
      </PlateType>
      <ImageSize>
        <PixelWidthUM>0.65</PixelWidthUM>
      </ImageSize>
    </MetaData>

Opera Phenix Metadata
^^^^^^^^^^^^^^^^^

Opera Phenix metadata is stored in XML files with the following structure:

.. code-block:: xml

    <OperaDB>
      <MeasurementDetail>
        <ImageResolutionX Unit="m">0.00000065</ImageResolutionX>
        <ImageResolutionY Unit="m">0.00000065</ImageResolutionY>
      </MeasurementDetail>
      <Image>
        <PositionX Unit="m">0.0</PositionX>
        <PositionY Unit="m">0.0</PositionY>
      </Image>
    </OperaDB>

Output Directory Structure
-----------------------

EZStitcher creates the following directory structure:

.. code-block:: text

    plate_folder/
    ├── plate_folder_processed/
    │   ├── A01_s1_w1.tif
    │   ├── A01_s2_w1.tif
    │   └── ...
    ├── plate_folder_post_processed/
    │   ├── A01_s1_w1.tif
    │   ├── A01_s2_w1.tif
    │   └── ...
    ├── plate_folder_positions/
    │   ├── A01_w1_positions.csv
    │   ├── A02_w1_positions.csv
    │   └── ...
    └── plate_folder_stitched/
        ├── A01_w1.tif
        ├── A01_w2.tif
        └── ...

The directories are:

- **plate_folder_processed**: Contains preprocessed individual tiles
- **plate_folder_post_processed**: Contains post-processed tiles (e.g., after Z-stack flattening)
- **plate_folder_positions**: Contains CSV files with tile positions
- **plate_folder_stitched**: Contains final stitched images
