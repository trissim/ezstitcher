Microscope Support
=================

This page explains EZStitcher's support for different microscope types.

Supported Microscope Types
------------------------

EZStitcher currently supports the following microscope types:

- **ImageXpress**: Molecular Devices ImageXpress microscopes
- **Opera Phenix**: PerkinElmer Opera Phenix microscopes
- **Auto**: Automatic detection based on file naming and directory structure

You can specify the microscope type when creating a MicroscopeHandler:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import MicroscopeHandler

    # Create a handler for ImageXpress
    handler = MicroscopeHandler(plate_folder="path/to/plate_folder", microscope_type="ImageXpress")

    # Create a handler for Opera Phenix
    handler = MicroscopeHandler(plate_folder="path/to/plate_folder", microscope_type="OperaPhenix")

    # Create a handler with auto-detection
    handler = MicroscopeHandler(plate_folder="path/to/plate_folder", microscope_type="auto")

ImageXpress Specifics
-------------------

ImageXpress microscopes use a specific file naming convention and directory structure:

File Naming
~~~~~~~~~~

.. code-block:: text

    A01_s1_w1.tif

- **A01**: Well identifier (row A, column 01)
- **s1**: Site identifier (site 1)
- **w1**: Channel identifier (wavelength 1)

Directory Structure
~~~~~~~~~~~~~~~~

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

Metadata Extraction
~~~~~~~~~~~~~~~~

ImageXpress metadata is stored in HTD files:

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

EZStitcher extracts the following information from HTD files:

- Grid dimensions (number of tiles in X and Y directions)
- Pixel size (in micrometers)
- Well information
- Channel information

Z-Stack Handling
~~~~~~~~~~~~~

ImageXpress supports two Z-stack formats:

1. **Folder-based Z-stacks**: Organized in separate folders (ZStep_1, ZStep_2, etc.)
2. **Suffix-based Z-stacks**: Using a z-index suffix in the filename (e.g., A01_s1_w1_z1.tif)

EZStitcher automatically detects and processes both Z-stack formats.

Opera Phenix Specifics
--------------------

Opera Phenix microscopes use a different file naming convention and directory structure:

File Naming
~~~~~~~~~~

.. code-block:: text

    r01c01f001p01-ch1sk1fk1fl1.tiff

- **r01c01**: Well identifier (row 01, column 01, equivalent to A01)
- **CH1**: Channel identifier (channel 1)
- **F1**: Field identifier (field 1)
- **P1**: Plane identifier (plane 1)
- **R1**: Round identifier (round 1)

Directory Structure
~~~~~~~~~~~~~~~~

.. code-block:: text

    plate_folder/
    ├── Images/
    │   ├── r01c01f001p01-ch1sk1fk1fl1.tiff
    │   ├── r01c01f001p02-ch1sk1fk1fl1.tiff
    │   ├── r01c01f002p01-ch1sk1fk1fl1.tiff
    │   └── ...
    ├── Index.xml
    └── ...

Metadata Extraction
~~~~~~~~~~~~~~~~

Opera Phenix metadata is stored in XML files (Index.xml):

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
        <Image id="r01c01f001p01-ch1sk1fk1fl1">
          <URL>Images/r01c01f001p01-ch1sk1fk1fl1.tiff</URL>
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

EZStitcher extracts the following information from XML files:

- Grid dimensions (by analyzing PositionX/Y coordinates)
- Pixel size (from ImageResolutionX/Y)
- Well information
- Channel information
- Field information
- Plane information

Z-Stack Handling
~~~~~~~~~~~~~

Opera Phenix Z-stacks are identified by the plane identifier (P1, P2, etc.) in the filename. EZStitcher automatically detects and processes these Z-stacks.

Auto-Detection
------------

EZStitcher can automatically detect the microscope type based on the file naming and directory structure:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import MicroscopeHandler

    # Create a handler with auto-detection
    handler = MicroscopeHandler(plate_folder="path/to/plate_folder", microscope_type="auto")

The auto-detection process:

1. Examines the directory structure
2. Checks for characteristic files (HTD files for ImageXpress, Index.xml for Opera Phenix)
3. Examines file naming patterns
4. Selects the most likely microscope type

If auto-detection fails, EZStitcher falls back to the default microscope type (ImageXpress).

Adding Support for New Microscopes
--------------------------------

You can add support for new microscope types by implementing the following interfaces:

FilenameParser
~~~~~~~~~~~~

Implement the FilenameParser interface to parse filenames for your microscope type:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import FilenameParser
    from typing import Dict, Any, Optional

    class MyMicroscopeFilenameParser(FilenameParser):
        def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
            """Parse a filename into its components."""
            # Implement parsing logic
            # Return a dictionary of components, or None if parsing fails
            pass

        def get_components(self, filename: str) -> Optional[Dict[str, Any]]:
            """Get components from a filename."""
            # Implement component extraction logic
            # Return a dictionary of components, or None if extraction fails
            pass

MetadataHandler
~~~~~~~~~~~~~

Implement the MetadataHandler interface to extract metadata for your microscope type:

.. code-block:: python

    from ezstitcher.core.microscope_interfaces import MetadataHandler
    from pathlib import Path
    from typing import Optional, Tuple

    class MyMicroscopeMetadataHandler(MetadataHandler):
        def find_metadata_file(self, plate_path: Path) -> Optional[Path]:
            """Find the metadata file for a plate."""
            # Implement metadata file finding logic
            # Return the path to the metadata file, or None if not found
            pass

        def get_grid_dimensions(self, plate_path: Path) -> Optional[Tuple[int, int]]:
            """Get the grid dimensions from metadata."""
            # Implement grid dimensions extraction logic
            # Return a tuple of (grid_size_x, grid_size_y), or None if not available
            pass

        def get_pixel_size(self, plate_path: Path) -> Optional[float]:
            """Get the pixel size from metadata."""
            # Implement pixel size extraction logic
            # Return the pixel size in micrometers, or None if not available
            pass

Registration
~~~~~~~~~~

Register your microscope type by placing your implementation in a module within the `ezstitcher.microscopes` package. The MicroscopeHandler will automatically discover and register your implementation.
