Processing Pipeline
==================

.. module:: ezstitcher.core.processing_pipeline

This module contains the core processing pipeline for EZStitcher.

PipelineOrchestrator
-------------------

.. py:class:: PipelineOrchestrator(config)

   A robust pipeline orchestrator for microscopy image processing.

   The pipeline follows a clear, linear flow:
   
   1. Load and organize images
   2. Process tiles (per well, per site, per channel)
   3. Select or compose channels
   4. Flatten Z-stacks (if present)
   5. Generate stitching positions
   6. Stitch images

   :param config: Configuration for the pipeline
   :type config: :class:`~ezstitcher.core.config.PipelineConfig`

   .. py:method:: run(plate_folder)

      Process a plate through the complete pipeline.

      :param plate_folder: Path to the plate folder
      :type plate_folder: str or Path
      :return: True if successful, False otherwise
      :rtype: bool

   .. py:method:: process_well(well, wavelength_patterns, wavelength_patterns_z, dirs)

      Process a single well through the pipeline.

      :param well: Well identifier
      :type well: str
      :param wavelength_patterns: Dictionary mapping wavelengths to varying site patterns
      :type wavelength_patterns: dict
      :param wavelength_patterns_z: Dictionary mapping wavelengths to varying z_index patterns
      :type wavelength_patterns_z: dict
      :param dirs: Dictionary of directories
      :type dirs: dict

   .. py:method:: process_reference_images(well, wavelength_patterns, wavelength_patterns_z, dirs)

      Process reference images for position generation.

      :param well: Well identifier
      :type well: str
      :param wavelength_patterns: Dictionary mapping wavelengths to varying site patterns
      :type wavelength_patterns: dict
      :param wavelength_patterns_z: Dictionary mapping wavelengths to varying z_index patterns
      :type wavelength_patterns_z: dict
      :param dirs: Dictionary of directories
      :type dirs: dict

   .. py:method:: process_final_images(well, wavelength_patterns, wavelength_patterns_z, dirs)

      Process images for final stitching.

      :param well: Well identifier
      :type well: str
      :param wavelength_patterns: Dictionary mapping wavelengths to varying site patterns
      :type wavelength_patterns: dict
      :param wavelength_patterns_z: Dictionary mapping wavelengths to varying z_index patterns
      :type wavelength_patterns_z: dict
      :param dirs: Dictionary of directories
      :type dirs: dict

   .. py:method:: process_tiles(input_dir, output_dir, patterns, channel)

      Process tiles using the specified patterns and channel.

      :param input_dir: Input directory
      :type input_dir: str or Path
      :param output_dir: Output directory
      :type output_dir: str or Path
      :param patterns: List of file patterns
      :type patterns: list
      :param channel: Channel identifier
      :type channel: str
      :return: List of output file paths
      :rtype: list

   .. py:method:: create_composite(well, input_dir, channel_patterns, weights=None)

      Create a composite image from multiple channels for each site and z-index.

      :param well: Well identifier
      :type well: str
      :param input_dir: Input directory
      :type input_dir: str or Path
      :param channel_patterns: Dictionary mapping channels to patterns
      :type channel_patterns: dict
      :param weights: Dictionary mapping channels to weights, or None to use first channel as reference
      :type weights: dict or None
      :return: List of paths to created composite images
      :rtype: list

   .. py:method:: flatten_zstacks(input_dir, output_dir, patterns, method="max")

      Finds planes of the same tile and flattens them into a single image.

      :param input_dir: Input directory
      :type input_dir: str or Path
      :param output_dir: Output directory
      :type output_dir: str or Path
      :param patterns: List of file patterns
      :type patterns: list
      :param method: Method to use for flattening ('max', 'mean', 'best_focus', etc.)
      :type method: str or callable
      :return: List of paths to created images
      :rtype: list

   .. py:method:: generate_positions(well, dirs)

      Generate stitching positions for a well.

      :param well: Well identifier
      :type well: str
      :param dirs: Dictionary of directories
      :type dirs: dict
      :return: Tuple of (positions_file, stitch_pattern)
      :rtype: tuple

   .. py:method:: stitch_images(well, dirs, positions_file)

      Stitch images for a well.

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
