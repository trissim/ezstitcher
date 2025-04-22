Microscope Formats
==================

This page describes the file naming conventions and directory structures for different microscope types supported by EZStitcher.

ImageXpress
-----------

ImageXpress microscopes use the following file naming convention:

.. code-block:: text

    <well>_s<site>_w<channel>[_z<z_index>].tif

For example:

- ``A01_s1_w1.tif``: Well A01, Site 1, Channel 1
- ``A01_s1_w2.tif``: Well A01, Site 1, Channel 2
- ``A01_s2_w1.tif``: Well A01, Site 2, Channel 1
- ``A01_s1_w1_z1.tif``: Well A01, Site 1, Channel 1, Z-index 1

Directory Structure
^^^^^^^^^^^^^^^^^^^

ImageXpress typically organizes files in the following structure:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── A01_s1_w1.tif
    │   ├── A01_s1_w2.tif
    │   ├── A01_s2_w1.tif
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

Alternatively, Z-stacks may be organized with Z-index in the filename:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── A01_s1_w1_z1.tif
    │   ├── A01_s1_w1_z2.tif
    │   ├── A01_s1_w2_z1.tif
    │   └── ...
    └── ...

Metadata
^^^^^^^^

ImageXpress metadata is stored in HTD (Hardware Test Definition) files with names like:

- ``<plate_name>.HTD``
- ``<plate_name>_meta.HTD``
- ``MetaData/<plate_name>.HTD``

The metadata contains information about:

- Grid dimensions (number of sites in x and y directions)
- Acquisition settings

Pixel size information is typically stored in the TIFF files themselves.

Opera Phenix
------------

Opera Phenix microscopes use the following file naming convention:

.. code-block:: text

    r<row>c<col>f<field>p<plane>-ch<channel>sk<skew>fk<focus>fl<flim>.tiff

For example:


- ``r01c01f01p01-ch1sk1fk1fl1.tiff``: Well A01, Channel 1, Field 1, Plane 1
- ``r01c01f01p01-ch2sk1fk1fl1.tiff``: Well A01, Channel 2, Field 1, Plane 1
- ``r01c01f02p01-ch2sk1fk1fl1.tiff``: Well A01, Channel 1, Field 2, Plane 1
- ``r01c01f01p02-ch2sk1fk1fl1.tiff``: Well A01, Channel 1, Field 1, Plane 2

Components:

- ``r<row>``: Row number (r01 = A, r02 = B, etc.)
- ``c<col>``: Column number (c01, c02, etc.)
- ``ch<channel>``: Channel number (ch1, ch2, etc.)
- ``f<field>``: Field/site number (f1, f2, etc.)
- ``p<plane>``: Z-plane number (p1, p2, etc.)

Directory Structure
^^^^^^^^^^^^^^^^^^^

Opera Phenix typically organizes files in the following structure:

.. code-block:: text

    plate_folder/
    ├── Images/
    │   ├── r01c01f01p01-ch1sk1fk1fl1.tiff
    │   ├── r01c01f02p01-ch1sk1fk1fl1.tiff
    │   ├── r01c01f03p01-ch1sk1fk1fl1.tiff
    │   └── ...
    ├── Index.xml
    └── ...

Metadata
^^^^^^^^

Opera Phenix metadata is stored in XML files with names like:

- ``Index.xml``
- ``MeasurementDetail.xml``

The metadata contains information about:

- Image resolution (pixel size)
- Position coordinates for each field
- Acquisition settings

Automatic Detection
-------------------

EZStitcher can automatically detect the microscope type based on the file structure and naming conventions:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import MicroscopeHandler
    from pathlib import Path

    plate_folder = Path("path/to/plate_folder")
    handler = MicroscopeHandler(plate_folder=plate_folder)
    print(f"Detected microscope type: {handler.__class__.__name__}")

The detection algorithm:

1. Examines the directory structure
2. Checks for characteristic metadata files
3. Samples image filenames and tries to parse them with different parsers
4. Selects the most likely microscope type based on the results

Adding Support for New Microscopes
----------------------------------

To add support for a new microscope type:

1. Create a new file in the `ezstitcher/microscopes/` directory
2. Implement the `FilenameParser` and `MetadataHandler` interfaces
3. Register the new microscope type in `ezstitcher/microscopes/__init__.py`

See the :doc:`../development/extending` section for details.

Comparison of Microscope Formats
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - ImageXpress
     - Opera Phenix
   * - File Extension
     - .tif
     - .tiff
   * - Well Format
     - A01, B02, etc.
     - r01c01, r02c02, etc.
   * - Channel Identifier
     - w1, w2, etc.
     - ch1, ch2, etc.
   * - Site/Field Identifier
     - s1, s2, etc.
     - f1, f2, etc.
   * - Z-Stack Organization
     - ZStep folders or _z suffix
     - p1, p2, etc. in filename
   * - Metadata Format
     - HTD files with SiteRows/SiteColumns
     - XML with PositionX/Y coordinates
   * - Pixel Size Location
     - TIFF file metadata
     - ImageResolutionX/Y elements in XML
