Core Concepts
============

This page explains the core concepts of microscopy image stitching and how EZStitcher handles them.

Introduction to Microscopy Image Stitching
-----------------------------------------

Microscopy image stitching is the process of combining multiple overlapping images (tiles) into a single larger image. This is necessary because:

1. The field of view of a microscope is limited
2. High-resolution imaging often requires capturing multiple tiles
3. Stitching allows visualization and analysis of larger areas

EZStitcher addresses these challenges by:

- Automatically detecting tile positions
- Aligning tiles with subpixel precision
- Blending overlapping regions smoothly
- Handling multi-channel fluorescence and Z-stacks

Key Microscopy Concepts
----------------------

Plate-Based Experiments
~~~~~~~~~~~~~~~~~~~~~

In high-content screening, samples are typically organized in plates:

.. image:: ../_static/plate_diagram.png
   :width: 400
   :alt: Plate Diagram

Key concepts:

- **Plate**: A container with multiple wells (e.g., 96-well plate)
- **Well**: A single compartment in a plate, identified by a row letter and column number (e.g., A01, B02)
- **Site**: A specific location within a well where an image is captured
- **Grid**: The arrangement of sites within a well (e.g., 3×3 grid)

EZStitcher processes images on a per-well basis, stitching together all sites within each well.

Multi-Channel Fluorescence
~~~~~~~~~~~~~~~~~~~~~~~

Fluorescence microscopy captures images at different wavelengths to visualize different structures:

.. image:: ../_static/multichannel_diagram.png
   :width: 400
   :alt: Multi-Channel Diagram

Key concepts:

- **Channel**: A specific wavelength or color used for imaging
- **Channel ID**: An identifier for a channel (e.g., "1", "2", "DAPI", "GFP")
- **Composite**: A combined image created from multiple channels

EZStitcher can:

- Process each channel independently
- Create composite images from multiple channels
- Use one channel as a reference for stitching all channels

Z-Stacks
~~~~~~~

Z-stacks are 3D image stacks captured at different focal planes:

.. image:: ../_static/zstack_diagram.png
   :width: 400
   :alt: Z-Stack Diagram

Key concepts:

- **Z-Stack**: A series of images captured at different focal planes
- **Z-Plane**: A single image at a specific focal depth
- **Projection**: A 2D representation of a 3D stack (e.g., maximum intensity projection)
- **Best Focus**: The plane with the highest focus quality

EZStitcher provides several options for handling Z-stacks:

- Maximum intensity projection
- Mean projection
- Best focus selection
- Per-plane stitching

For detailed information about Z-stack processing, see the :doc:`../user_guide/zstack_handling` guide.

Tiled Images
~~~~~~~~~~

Tiled images are multiple overlapping images that cover a larger area:

.. image:: ../_static/tiling_diagram.png
   :width: 400
   :alt: Tiling Diagram

Key concepts:

- **Tile**: A single image captured at a specific position
- **Overlap**: The region where adjacent tiles overlap
- **Grid Size**: The number of tiles in X and Y directions
- **Position**: The coordinates of a tile in the final stitched image

EZStitcher handles tiled images by:

1. Determining the relative positions of tiles
2. Aligning tiles with subpixel precision
3. Blending overlapping regions
4. Assembling the final stitched image

EZStitcher Architecture
---------------------

Pipeline Architecture
~~~~~~~~~~~~~~~~~~~

EZStitcher uses a modular pipeline architecture with several key components:

.. image:: ../_static/pipeline_architecture.png
   :width: 600
   :alt: Pipeline Architecture

Core Components
~~~~~~~~~~~~~

- **PipelineOrchestrator**: Coordinates the entire workflow
- **MicroscopeHandler**: Handles microscope-specific functionality
- **Stitcher**: Performs image stitching
- **FocusAnalyzer**: Detects focus quality
- **ImagePreprocessor**: Processes images
- **FileSystemManager**: Manages file operations
- **ImageLocator**: Locates images in various directory structures

These components work together to process microscopy images in a flexible and extensible way.

Processing Workflow
~~~~~~~~~~~~~~~~

The processing workflow follows a clear, linear flow:

1. **Load and organize images**:
   - Detect microscope type
   - Find image directory
   - Organize Z-stack folders
   - Pad filenames for consistent sorting

2. **Process reference images** (for position generation):
   - Apply preprocessing functions to reference channels
   - Create composites if needed
   - Save processed reference images

3. **Generate stitching positions**:
   - Calculate relative positions of tiles
   - Save positions to CSV

4. **Process final images** (for stitching):
   - Apply preprocessing functions to all channels
   - Flatten Z-stacks if present
   - Save processed images

5. **Stitch images**:
   - Load processed images
   - Apply positions
   - Blend overlapping regions
   - Save stitched images

This workflow is implemented in the ``PipelineOrchestrator`` class, which coordinates all the components.

Input/Output Organization
---------------------

EZStitcher organizes input and output files in a specific way:

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

Each output directory serves a specific purpose:

- **processed**: Contains processed individual tiles
- **post_processed**: Contains post-processed images (after channel selection/composition)
- **positions**: Contains CSV files with stitching positions
- **stitched**: Contains final stitched images

For detailed information about file organization, see the :doc:`../user_guide/file_organization` guide.

Microscope Support
-----------------

Microscope Handlers
~~~~~~~~~~~~~~~~

Microscope handlers are responsible for handling microscope-specific functionality:

.. image:: ../_static/microscope_handler.png
   :width: 500
   :alt: Microscope Handler

A ``MicroscopeHandler`` is composed of:

- **FilenameParser**: Parses microscope-specific filenames
- **MetadataHandler**: Extracts metadata from microscope-specific files

EZStitcher includes handlers for:

- **ImageXpress**: Molecular Devices ImageXpress microscopes
- **Opera Phenix**: PerkinElmer Opera Phenix microscopes

The microscope handler is automatically created based on the detected microscope type.

Supported Microscope Formats
~~~~~~~~~~~~~~~~~~~~~~~~~

EZStitcher supports multiple microscope formats:

ImageXpress
^^^^^^^^^

- **File Naming**: ``A01_s1_w1.tif`` (Well A01, Site 1, Channel 1)
- **Directory Structure**:

  .. code-block:: text

      plate_folder/
      ├── TimePoint_1/
      │   ├── A01_s1_w1.tif
      │   ├── A01_s1_w2.tif
      │   ├── A01_s2_w1.tif
      │   └── ...
      └── ...

Opera Phenix
^^^^^^^^^^

- **File Naming**: ``r01c01f001p01-ch1sk1fk1fl1.tiff`` (Well A01, Channel 1, Field 1, Plane 1)
- **Directory Structure**:

  .. code-block:: text

      plate_folder/
      ├── Images/
      │   ├── r01c01f001p01-ch1sk1fk1fl1.tiff
      │   ├── r01c01f001p02-ch1sk1fk1fl1.tiff
      │   ├── r01c01f002p01-ch1sk1fk1fl1.tiff
      │   └── ...
      ├── Index.xml
      └── ...

For detailed information about supported microscope formats, see the :doc:`../appendices/microscope_formats` appendix.

Auto-Detection
^^^^^^^^^^^

EZStitcher can automatically detect the microscope type based on the file naming and directory structure. This makes it easy to use without having to specify the microscope type explicitly.
