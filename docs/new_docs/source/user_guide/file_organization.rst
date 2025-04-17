File Organization
================

This page explains how EZStitcher organizes files and directories.

Expected Input Structure
----------------------

EZStitcher expects specific input directory structures depending on the microscope type.

ImageXpress
~~~~~~~~~~

Standard structure:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── A01_s1_w1.tif
    │   ├── A01_s1_w2.tif
    │   ├── A01_s2_w1.tif
    │   └── ...
    └── ...

Z-stack structure:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── ZStep_1/
    │   │   ├── A01_s1_w1.tif
    │   │   ├── A01_s1_w2.tif
    │   │   └── ...
    │   ├── ZStep_2/
    │   │   ├── A01_s1_w1.tif
    │   │   ├── A01_s1_w2.tif
    │   │   └── ...
    │   └── ...
    └── ...

Opera Phenix
~~~~~~~~~~~

Standard structure:

.. code-block:: text

    plate_folder/
    ├── Images/
    │   ├── 0101K1F1P1R1.tiff  # Well A01, Channel 1, Field 1, Plane 1, Round 1
    │   ├── 0101K1F1P2R1.tiff  # Well A01, Channel 1, Field 1, Plane 2, Round 1
    │   ├── 0101K1F2P1R1.tiff  # Well A01, Channel 1, Field 2, Plane 1, Round 1
    │   └── ...
    ├── Index.xml
    └── ...

Output Directory Structure
------------------------

EZStitcher creates several output directories:

.. code-block:: text

    parent_directory/
    ├── plate_folder/                  # Original data
    │   └── ...
    ├── plate_folder_processed/        # Processed individual tiles
    │   └── ...
    ├── plate_folder_post_processed/   # Post-processed images
    │   └── ...
    ├── plate_folder_positions/        # CSV files with stitching positions
    │   └── ...
    └── plate_folder_stitched/         # Final stitched images
        └── ...

Processed Directory
~~~~~~~~~~~~~~~~~

Contains processed individual tiles:

.. code-block:: text

    plate_folder_processed/
    ├── A01_s1_w1.tif
    ├── A01_s1_w2.tif
    ├── A01_s2_w1.tif
    └── ...

Post-Processed Directory
~~~~~~~~~~~~~~~~~~~~~~

Contains post-processed images (after channel selection/composition):

.. code-block:: text

    plate_folder_post_processed/
    ├── A01_w1.tif
    ├── A01_w2.tif
    └── ...

Positions Directory
~~~~~~~~~~~~~~~~~

Contains CSV files with stitching positions:

.. code-block:: text

    plate_folder_positions/
    ├── A01_w1.csv
    ├── A01_w2.csv
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

Contains final stitched images:

.. code-block:: text

    plate_folder_stitched/
    ├── A01_w1.tif
    ├── A01_w2.tif
    └── ...

Naming Conventions
----------------

EZStitcher uses specific naming conventions for files:

Well Identifiers
~~~~~~~~~~~~~~

- **ImageXpress**: A01, A02, B01, B02, etc.
- **Opera Phenix**: 0101 (A01), 0102 (A02), 0201 (B01), 0202 (B02), etc. (row and column as 2-digit numbers)

Site Identifiers
~~~~~~~~~~~~~~

- **ImageXpress**: s1, s2, s3, etc.
- **Opera Phenix**: F1, F2, F3, etc.

Channel Identifiers
~~~~~~~~~~~~~~~~

- **ImageXpress**: w1, w2, w3, etc.
- **Opera Phenix**: CH1, CH2, CH3, etc.

Z-Stack Identifiers
~~~~~~~~~~~~~~~~

- **ImageXpress**: ZStep_1, ZStep_2, etc. (folder-based) or _z1, _z2, etc. (suffix-based)
- **Opera Phenix**: P1, P2, P3, etc.

File Formats
-----------

EZStitcher supports several image formats:

- **TIFF**: Preferred format for microscopy images
- **PNG**: Supported for input and output
- **JPEG**: Supported for input and output

Metadata formats:

- **ImageXpress**: HTD files (text-based)
- **Opera Phenix**: XML files (Index.xml)

Position CSV format:

.. code-block:: text

    filename,x,y
    A01_s1_w1.tif,0.0,0.0
    A01_s2_w1.tif,1024.5,0.0
    ...

Configuration file formats:

- **JSON**: JSON configuration files
- **YAML**: YAML configuration files

Metadata Files
------------

EZStitcher extracts metadata from microscope-specific files:

ImageXpress HTD Files
~~~~~~~~~~~~~~~~~~~

HTD files contain metadata for ImageXpress acquisitions:

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

Opera Phenix XML Files
~~~~~~~~~~~~~~~~~~~~

Index.xml files contain metadata for Opera Phenix acquisitions:

.. code-block:: xml

    <?xml version="1.0" encoding="utf-8"?>
    <EvaluationInputData xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" Version="1" xmlns="http://www.perkinelmer.com/PEHH/HarmonyV6">
      <Plates>
        <Plate>
          <PlateID>plate_name</PlateID>
          <PlateTypeName>96well</PlateTypeName>
          <PlateRows>8</PlateRows>
          <PlateColumns>12</PlateColumns>
          ...
        </Plate>
      </Plates>
      <Images>
        <Image id="0101CH1F1P1R1">
          <URL>Images/0101CH1F1P1R1.tiff</URL>
          <ChannelID>1</ChannelID>
          <FieldID>1</FieldID>
          <PlaneID>1</PlaneID>
          <PositionX>0.0</PositionX>
          <PositionY>0.0</PositionY>
          <ImageResolutionX>0.65</ImageResolutionX>
          <ImageResolutionY>0.65</ImageResolutionY>
          <ImageResolutionXUnit>m</ImageResolutionXUnit>
          <ImageResolutionYUnit>m</ImageResolutionYUnit>
          ...
        </Image>
        ...
      </Images>
    </EvaluationInputData>

EZStitcher extracts the following information from metadata files:

- Grid dimensions (number of tiles in X and Y directions)
- Pixel size (in micrometers)
- Well information
- Channel information
