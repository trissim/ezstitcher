File Formats
===========

This appendix provides technical specifications for file formats and directory structures supported by EZStitcher.

Image File Formats
----------------

EZStitcher supports the following image file formats:

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

Bit Depth Support
~~~~~~~~~~~~~~

EZStitcher supports various bit depths for image processing:

- **8-bit**: Values from 0-255 (uint8)
- **16-bit**: Values from 0-65535 (uint16) - Recommended for most microscopy images
- **32-bit float**: Floating-point values (float32) - Used for some specialized processing

Position Files
------------

Position files are CSV files with the following format:

.. code-block:: text

    filename,x,y
    A01_s1_w1.tif,0.0,0.0
    A01_s2_w1.tif,1024.5,0.0
    A01_s3_w1.tif,2049.2,0.0
    A01_s4_w1.tif,0.0,1024.3
    ...

Where:
- **filename**: The filename of the tile
- **x, y**: Pixel coordinates in the final stitched image (floating-point values for subpixel precision)

Alternative format with grid positions:

.. code-block:: text

    file,i,j,x,y
    A01_s1_w1.tif,0,0,0.0,0.0
    A01_s2_w1.tif,1,0,1024.5,0.0
    A01_s3_w1.tif,0,1,0.0,1024.5
    A01_s4_w1.tif,1,1,1024.5,1024.5
    ...

Where:
- **file**: The filename of the tile
- **i, j**: Grid coordinates (column, row)
- **x, y**: Pixel coordinates in the final stitched image

Configuration Files
----------------

EZStitcher supports JSON and YAML configuration files:

JSON Example:

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

YAML Example:

.. code-block:: yaml

    reference_channels:
      - "1"
      - "2"
    well_filter:
      - "A01"
      - "A02"
    stitcher:
      tile_overlap: 15.0
      max_shift: 75
      margin_ratio: 0.15
    focus_config:
      method: laplacian
      roi: [100, 100, 200, 200]
    reference_flatten: max_projection
    stitch_flatten: best_focus
    additional_projections:
      - max
      - mean

Metadata Formats
-------------

EZStitcher extracts metadata from microscope-specific files:

ImageXpress Metadata
^^^^^^^^^^^^^^^^^

ImageXpress metadata is stored in HTD files (text-based) or XML files with the following structure:

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

HTD files have a similar structure but in a text-based format:

.. code-block:: text

    [General]
    Plate Type=96 Well
    ...
    [Sites]
    SiteCount=9
    GridRows=3
    GridColumns=3
    ...
    [Wavelengths]
    WavelengthCount=3
    ...
    [Scale]
    PixelSize=0.65
    ...

Opera Phenix Metadata
^^^^^^^^^^^^^^^^^

Opera Phenix metadata is stored in XML files (Index.xml) with the following structure:

.. code-block:: xml

    <EvaluationInputData>
      <Plates>
        <Plate>
          <PlateID>plate_name</PlateID>
          <PlateTypeName>96well</PlateTypeName>
          <PlateRows>8</PlateRows>
          <PlateColumns>12</PlateColumns>
        </Plate>
      </Plates>
      <Images>
        <Image id="r01c01f001p01-ch1sk1fk1fl1">
          <URL>Images/r01c01f001p01-ch1sk1fk1fl1.tiff</URL>
          <ChannelID>1</ChannelID>
          <FieldID>1</FieldID>
          <PlaneID>1</PlaneID>
          <PositionX>0.0</PositionX>
          <PositionY>0.0</PositionY>
          <ImageResolutionX>0.65</ImageResolutionX>
          <ImageResolutionY>0.65</ImageResolutionY>
        </Image>
      </Images>
    </EvaluationInputData>

File Naming Conventions
--------------------

For detailed information about file naming conventions for different microscope types, see the :doc:`microscope_formats` appendix.

Output File Structure
------------------

EZStitcher creates the following directory structure during processing:

.. code-block:: text

    plate_folder/                 # Original data
    plate_folder_processed/       # Processed individual tiles
    plate_folder_post_processed/  # Post-processed images
    plate_folder_positions/       # CSV files with stitching positions
    plate_folder_stitched/        # Final stitched images

For detailed information about how EZStitcher organizes and processes files, see the :doc:`../user_guide/file_organization` guide.
