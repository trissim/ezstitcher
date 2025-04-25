===================
PipelineOrchestrator
===================

Role and Responsibilities
------------------------

The ``PipelineOrchestrator`` is the top-level component that manages all plate-specific operations and coordinates the execution of pipelines. It serves as an abstraction layer between the plate-specific details and the pipeline steps.

Key responsibilities:

* **Plate Management**:
  - Plate and well detection
  - Microscope handler initialization (specific to each plate type)
  - Image locator configuration

* **Workspace Initialization**:
  - Creates a workspace by mirroring the plate folder path structure
  - Creates symlinks to the original images in this workspace
  - Ensures that modifications happen on workspace copies, not original data
  - Provides this workspace as the input for pipelines

* **Pipeline Execution**:
  - Multithreaded execution across wells
  - Error handling and logging

* **Specialized Services**:
  - Provides stitching objects with the right configuration for the plate
  - Manages position generation specific to the plate format
  - Abstracts plate-specific operations that depend on the microscope handler

The orchestrator acts as a "plate manager" that knows how to handle the specific details of different plate formats, allowing the pipeline steps to focus on their image processing tasks without needing to know about the underlying plate structure.

Creating an Orchestrator
-----------------------

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration
    config = PipelineConfig(
        num_workers=2  # Use 2 worker threads
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="path/to/plate"
    )

Plate-Specific Services
----------------------

The orchestrator provides several plate-specific services that abstract away the details of different plate formats:

1. **Workspace and Original Data Protection**:

   The orchestrator creates a workspace to protect original data:

   .. code-block:: python

       # Create an orchestrator with a plate path
       orchestrator = PipelineOrchestrator(
           config=config,
           plate_path="path/to/plate"  # Original plate path
       )

       # Access the workspace path (contains symlinks to original images)
       workspace_path = orchestrator.workspace_path

   **Used by**: Pipelines and steps use this workspace path as their input directory, ensuring that original data is protected from modification.

2. **Microscope Handler**: Understands the specific plate format and how to parse filenames

   .. code-block:: python

       # The microscope handler knows how to interpret filenames for the specific plate type
       microscope_handler = orchestrator.microscope_handler

       # Parse a filename to extract components (channel, z-index, site, etc.)
       components = microscope_handler.parser.parse_filename("image_c1_z3_s2.tif")

       # Generate patterns for finding images
       patterns = microscope_handler.auto_detect_patterns(input_dir)

   **Used by**: The `get_stitcher()` method uses the microscope handler's parser to configure the stitcher. The `stitch_images()` and `generate_positions()` methods use it to understand the plate format and parse filenames.

3. **Position Generation**: Generates position files for stitching

   .. code-block:: python

       # Generate positions for a specific well
       positions_file, _ = orchestrator.generate_positions(
           well="A01",
           input_dir=input_dir,
           positions_dir=positions_dir
       )

   **Used by**: The `PositionGenerationStep` calls this method to generate position files for stitching. Internally, this method uses the microscope handler and a stitcher instance obtained via `get_stitcher()`.

4. **Image Stitching**: Stitches images using position files

   .. code-block:: python

       # Stitch images for a specific well
       orchestrator.stitch_images(
           well="A01",
           input_dir=input_dir,
           output_dir=output_dir,
           positions_file=positions_file
       )

   **Used by**: The `ImageStitchingStep` calls this method to stitch images. Internally, this method uses the microscope handler and a stitcher instance obtained via `get_stitcher()`.

5. **Thread-Safe Stitcher Creation**:

   The `get_stitcher()` method creates a new `Stitcher` instance configured for the plate:

   .. code-block:: python

       # Get a thread-safe stitcher instance
       stitcher = orchestrator.get_stitcher()

   **Used by**: The `stitch_images()` and `generate_positions()` methods call this internally to get a thread-safe stitcher instance. Steps don't need to call this directly.

Running Pipelines
----------------

The orchestrator can run one or more pipelines:

.. code-block:: python

    # Run a single pipeline
    orchestrator.run(pipelines=[pipeline])

    # Run multiple pipelines in sequence
    orchestrator.run(pipelines=[pipeline1, pipeline2, pipeline3])

When multiple pipelines are provided, they are executed in sequence for each well. If ``num_workers`` is greater than 1, multiple wells are processed in parallel.
