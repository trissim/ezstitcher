Core Concepts
=============

This page explains the core concepts of EZStitcher's architecture and workflow.

Pipeline Architecture
---------------------

EZStitcher uses a modular pipeline architecture with several key components:

.. image:: ../_static/pipeline_architecture.png
   :width: 600
   :alt: Pipeline Architecture

Key components:

- **PipelineOrchestrator**: Coordinates the entire workflow
- **MicroscopeHandler**: Handles microscope-specific functionality
- **Stitcher**: Performs image stitching
- **FocusAnalyzer**: Detects focus quality
- **ImagePreprocessor**: Processes images
- **FileSystemManager**: Manages file operations
- **ImageLocator**: Locates images in various directory structures

These components work together to process microscopy images in a flexible and extensible way.

Processing Workflow
-------------------

The processing workflow follows a clear, linear flow:

1. **Load and organize images**:
   - Detect microscope type
   - Find image directory
   - Organize Z-stack folders
   - Pad filenames for consistent sorting

2. **Process tiles** (per well, per site, per channel):
   - Apply preprocessing functions
   - Save processed images

3. **Select or compose channels**:
   - Select specific channels
   - Create composite images
   - Save post-processed images

4. **Flatten Z-stacks** (if present):
   - Apply flattening function (max projection, best focus, etc.)
   - Save flattened images

5. **Generate stitching positions**:
   - Calculate relative positions of tiles
   - Save positions to CSV

6. **Stitch images**:
   - Load images
   - Apply positions
   - Blend overlapping regions
   - Save stitched images

This workflow is implemented in the ``PipelineOrchestrator`` class, which coordinates all the components.

Input/Output Organization
-------------------------

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

Microscope Handlers
-------------------

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

File Naming Conventions
-----------------------

EZStitcher uses specific file naming conventions:

ImageXpress
~~~~~~~~~~~

.. code-block:: text

    A01_s1_w1.tif

- **A01**: Well identifier (row A, column 01)
- **s1**: Site identifier (site 1)
- **w1**: Channel identifier (wavelength 1)

Opera Phenix
~~~~~~~~~~~~

.. code-block:: text

    0101K1F1P1R1.tiff

- **0101**: Well identifier (row 01, column 01, equivalent to A01)
- **K1**: Channel identifier (channel 1)
- **F1**: Field identifier (field 1)
- **P1**: Plane identifier (plane 1)
- **R1**: Round identifier (round 1)

These naming conventions are used to extract components from filenames and to generate patterns for finding files.
