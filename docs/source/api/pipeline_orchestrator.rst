Pipeline Architecture
==================

.. module:: ezstitcher.core.pipeline_orchestrator

This module contains the core processing pipeline architecture for EZStitcher, which consists of three main components:

1. **PipelineOrchestrator**: Coordinates the execution of multiple pipelines across wells
2. **Pipeline**: A sequence of processing steps
3. **Step**: A single processing operation (with specialized subclasses)

For a detailed overview of the pipeline architecture, see :doc:`../development/pipeline_architecture`.

PipelineOrchestrator
-------------------

.. py:class:: PipelineOrchestrator(plate_path=None, workspace_path=None, config=None, fs_manager=None, image_preprocessor=None, focus_analyzer=None)

   The central coordinator that manages the execution of multiple pipelines across wells.

   :param plate_path: Path to the plate folder (optional, can be provided later in run())
   :type plate_path: str or Path
   :param workspace_path: Path to the workspace folder (optional, defaults to plate_path.parent/plate_path.name_workspace)
   :type workspace_path: str or Path
   :param config: Configuration for the pipeline orchestrator
   :type config: :class:`~ezstitcher.core.config.PipelineConfig`
   :param fs_manager: File system manager (optional, a new instance will be created if not provided)
   :type fs_manager: :class:`~ezstitcher.core.file_system_manager.FileSystemManager`
   :param image_preprocessor: Image processor (optional, a new instance will be created if not provided)
   :type image_preprocessor: :class:`~ezstitcher.core.image_processor.ImageProcessor`
   :param focus_analyzer: Focus analyzer (optional, a new instance will be created if not provided)
   :type focus_analyzer: :class:`~ezstitcher.core.focus_analyzer.FocusAnalyzer`

   .. py:method:: run(plate_path=None, pipelines=None)

      Run the pipeline orchestrator with the specified pipelines.

      :param plate_path: Path to the plate folder (optional if provided in __init__)
      :type plate_path: str or Path
      :param pipelines: List of pipelines to run
      :type pipelines: list of :class:`~ezstitcher.core.pipeline.Pipeline`
      :return: True if successful, False otherwise
      :rtype: bool

   .. py:method:: process_well(well, pipelines)

      Process a single well with the specified pipelines.

      :param well: Well identifier
      :type well: str
      :param pipelines: List of pipelines to run
      :type pipelines: list of :class:`~ezstitcher.core.pipeline.Pipeline`
      :return: True if successful, False otherwise
      :rtype: bool

   .. note::

      The ``setup_directories()`` method has been removed. Directory paths are now automatically resolved between steps.
      See :doc:`../concepts/directory_structure` for details on how EZStitcher manages directories.

   .. py:method:: detect_plate_structure(plate_path)

      Detect the plate structure and available wells.

      :param plate_path: Path to the plate folder
      :type plate_path: str or Path

   .. py:method:: generate_positions(well, input_dir, positions_dir)

      Generate stitching positions for a well.

      :param well: Well identifier
      :type well: str
      :param input_dir: Input directory containing reference images
      :type input_dir: str or Path
      :param positions_dir: Output directory for positions files
      :type positions_dir: str or Path
      :return: Tuple of (positions_dir, reference_pattern)
      :rtype: tuple

   .. py:method:: stitch_images(well, input_dir, output_dir, positions_path)

      Stitch images for a well.

      :param well: Well identifier
      :type well: str
      :param input_dir: Input directory containing processed images
      :type input_dir: str or Path
      :param output_dir: Output directory for stitched images
      :type output_dir: str or Path
      :param positions_path: Path to positions file
      :type positions_path: str or Path

Pipeline and Step Classes
---------------------

For documentation on the Pipeline and Step classes, see:

- :doc:`pipeline` - Documentation for the Pipeline class and ProcessingContext
- :doc:`steps` - Documentation for the Step class and its specialized subclasses
