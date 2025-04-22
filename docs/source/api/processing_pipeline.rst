Processing Pipeline
==================

.. module:: ezstitcher.core.processing_pipeline

This module contains the core processing pipeline for EZStitcher.

PipelineOrchestrator
-------------------

.. py:class:: PipelineOrchestrator(config=None, fs_manager=None, image_preprocessor=None, focus_analyzer=None)

   A robust pipeline orchestrator for microscopy image processing.

   The pipeline follows a clear, linear flow:

   1. Load and organize images (detect microscope type, organize Z-stacks, pad filenames)
   2. Process reference images (flatten Z-stacks, apply channel-specific processing, create composites)
   3. Generate stitching positions (calculate tile positions using reference images)
   4. Process final images (apply channel-specific processing, flatten Z-stacks)
   5. Stitch images (apply positions, blend overlapping regions)

   :param config: Configuration for the pipeline (optional)
   :type config: :class:`~ezstitcher.core.config.PipelineConfig`
   :param fs_manager: File system manager (optional)
   :type fs_manager: :class:`~ezstitcher.core.file_system_manager.FileSystemManager`
   :param image_preprocessor: Image preprocessor (optional)
   :type image_preprocessor: :class:`~ezstitcher.core.image_preprocessor.ImagePreprocessor`
   :param focus_analyzer: Focus analyzer (optional)
   :type focus_analyzer: :class:`~ezstitcher.core.focus_analyzer.FocusAnalyzer`

   .. py:method:: run(plate_folder)

      Process a plate through the complete pipeline.

      :param plate_folder: Path to the plate folder
      :type plate_folder: str or Path
      :return: True if successful, False otherwise
      :rtype: bool

   .. py:method:: process_well(well, dirs)

      Process a single well through the pipeline.

      :param well: Well identifier
      :type well: str
      :param dirs: Dictionary of directories
      :type dirs: dict

   .. py:method:: process_reference_images(well, dirs)

      Process reference images for position generation. This includes flattening Z-stacks using the
      reference_flatten method, applying channel-specific processing functions, and creating composite
      images if needed.

      :param well: Well identifier
      :type well: str
      :param dirs: Dictionary of directories
      :type dirs: dict

   .. py:method:: process_final_images(well, dirs)

      Process images for final stitching. This includes applying channel-specific processing functions
      to all channels and flattening Z-stacks using the stitch_flatten method if needed.

      :param well: Well identifier
      :type well: str
      :param dirs: Dictionary of directories
      :type dirs: dict

   .. py:method:: process_patterns_with_variable_components(input_dir, output_dir, well_filter=None, variable_components=None, group_by=None, processing_funcs=None, processing_args=None)

      Detect patterns with variable components and process them flexibly.

      :param input_dir: Input directory containing images
      :type input_dir: str or Path
      :param output_dir: Output directory for processed images
      :type output_dir: str or Path
      :param well_filter: List of wells to include
      :type well_filter: list, optional
      :param variable_components: Components to make variable (e.g., ['site', 'z_index'])
      :type variable_components: list, optional
      :param group_by: How to group patterns (e.g., 'channel', 'z_index', 'well')
      :type group_by: str, optional
      :param processing_funcs: Processing functions to apply
      :type processing_funcs: callable, list, dict, optional
      :param processing_args: Additional arguments to pass to processing functions
      :type processing_args: dict, optional
      :return: Dictionary mapping wells to processed file paths
      :rtype: dict

   .. py:method:: process_tiles(input_dir, output_dir, patterns, processing_funcs=None, **kwargs)

      Unified processing for image tiles.

      :param input_dir: Input directory
      :type input_dir: str or Path
      :param output_dir: Output directory
      :type output_dir: str or Path
      :param patterns: List of file patterns
      :type patterns: list
      :param processing_funcs: Processing functions to apply
      :type processing_funcs: callable, list, optional
      :param kwargs: Additional arguments to pass to processing functions
      :return: List of output file paths
      :rtype: list

   .. py:method:: generate_positions(well, dirs)

      Generate stitching positions for a well using the reference images. This calculates the relative
      positions of tiles and saves them to a CSV file for later use in the stitching step.

      :param well: Well identifier
      :type well: str
      :param dirs: Dictionary of directories
      :type dirs: dict
      :return: Tuple of (positions_file, stitch_pattern)
      :rtype: tuple

   .. py:method:: stitch_images(well, dirs, positions_file)

      Stitch images for a well. This loads the processed images, applies the positions from the
      positions file, blends overlapping regions, and saves the final stitched images.

      :param well: Well identifier
      :type well: str
      :param dirs: Dictionary of directories
      :type dirs: dict
      :param positions_file: Path to positions file
      :type positions_file: str or Path

   .. py:method:: _setup_directories(plate_path, input_dir)

      Set up directory structure for processing.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :param input_dir: Path to the input directory
      :type input_dir: str or Path
      :return: Dictionary of directories
      :rtype: dict

   .. py:method:: _prepare_images(plate_path)

      Prepare images by padding filenames and organizing Z-stack folders.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path
      :return: Path to the image directory
      :rtype: Path
