Pipeline Architecture
==================

.. module:: ezstitcher.core.processing_pipeline

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
   :param image_preprocessor: Image preprocessor (optional, a new instance will be created if not provided)
   :type image_preprocessor: :class:`~ezstitcher.core.image_preprocessor.ImagePreprocessor`
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

   .. py:method:: setup_directories()

      Set up directory structure for processing.

      :return: Dictionary of directories
      :rtype: dict

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

.. module:: ezstitcher.core.pipeline

Pipeline
-------

.. py:class:: Pipeline(steps=None, name=None)

   A sequence of processing steps that are executed in order.

   :param steps: Initial list of steps
   :type steps: list of :class:`~ezstitcher.core.steps.Step`
   :param name: Human-readable name for the pipeline
   :type name: str

   .. py:method:: add_step(step)

      Add a step to the pipeline.

      :param step: The step to add
      :type step: :class:`~ezstitcher.core.steps.Step`
      :return: Self, for method chaining
      :rtype: :class:`Pipeline`

   .. py:method:: run(input_dir=None, output_dir=None, well_filter=None, microscope_handler=None, orchestrator=None, positions_file=None)

      Execute the pipeline.

      This method can either:

      1. Take individual parameters and create a ProcessingContext internally, or
      2. Take a pre-configured ProcessingContext object (when called from PipelineOrchestrator)

      The orchestrator parameter is required as it provides access to the microscope handler and other components.

      :param input_dir: Optional input directory override
      :type input_dir: str or Path
      :param output_dir: Optional output directory override
      :type output_dir: str or Path
      :param well_filter: Optional well filter override
      :type well_filter: list
      :param microscope_handler: Optional microscope handler override
      :type microscope_handler: :class:`~ezstitcher.core.microscope_interfaces.MicroscopeHandler`
      :param orchestrator: Optional PipelineOrchestrator instance (required)
      :type orchestrator: :class:`PipelineOrchestrator`
      :param positions_file: Optional positions file to use for stitching
      :type positions_file: str or Path
      :return: The results of the pipeline execution
      :rtype: dict

   .. py:attribute:: input_dir

      Get or set the input directory for the pipeline.

      :type: Path or None

   .. py:attribute:: output_dir

      Get or set the output directory for the pipeline.

      :type: Path or None

.. module:: ezstitcher.core.steps

Step
----

.. py:class:: Step(func, variable_components=None, group_by=None, input_dir=None, output_dir=None, well_filter=None, processing_args=None, name=None)

   A processing step in a pipeline.

   :param func: The processing function(s) to apply
   :type func: callable, list, or dict
   :param variable_components: Components that vary across files (e.g., 'z_index', 'channel')
   :type variable_components: list
   :param group_by: How to group files for processing (e.g., 'channel', 'site')
   :type group_by: str
   :param input_dir: The input directory
   :type input_dir: str or Path
   :param output_dir: The output directory
   :type output_dir: str or Path
   :param well_filter: Wells to process
   :type well_filter: list
   :param processing_args: Additional arguments to pass to the processing function
   :type processing_args: dict
   :param name: Human-readable name for the step
   :type name: str

   .. py:method:: process(context)

      Process the step with the given context.

      :param context: The processing context
      :type context: :class:`~ezstitcher.core.pipeline.ProcessingContext`
      :return: The updated processing context
      :rtype: :class:`~ezstitcher.core.pipeline.ProcessingContext`

PositionGenerationStep
---------------------

.. py:class:: PositionGenerationStep(name="Position Generation", input_dir=None, output_dir=None, processing_args=None)

   A specialized Step for generating positions.

   :param name: Name of the step
   :type name: str
   :param input_dir: Input directory
   :type input_dir: str or Path
   :param output_dir: Output directory (for positions files)
   :type output_dir: str or Path
   :param processing_args: Additional arguments for the processing function
   :type processing_args: dict

   .. py:method:: process(context)

      Generate positions for stitching and store them in the context.

      :param context: The processing context
      :type context: :class:`~ezstitcher.core.pipeline.ProcessingContext`
      :return: The updated processing context
      :rtype: :class:`~ezstitcher.core.pipeline.ProcessingContext`

.. module:: ezstitcher.core.pipeline

ProcessingContext
---------------

.. py:class:: ProcessingContext(input_dir=None, output_dir=None, well_filter=None, config=None, **kwargs)

   Maintains state during pipeline execution.

   The ProcessingContext holds input/output directories, well filter, configuration,
   and results during pipeline execution. It serves as a communication mechanism
   between steps in a pipeline, allowing each step to access and modify shared state.

   :param input_dir: The input directory
   :type input_dir: str or Path
   :param output_dir: The output directory
   :type output_dir: str or Path
   :param well_filter: Wells to process
   :type well_filter: list
   :param config: Configuration parameters
   :type config: dict
   :param **kwargs: Additional context attributes that will be added to the context

   .. py:attribute:: input_dir

      The input directory for processing.

      :type: Path or None

   .. py:attribute:: output_dir

      The output directory for processing results.

      :type: Path or None

   .. py:attribute:: well_filter

      List of wells to process.

      :type: list or None

   .. py:attribute:: config

      Configuration parameters.

      :type: dict

   .. py:attribute:: results

      Processing results.

      :type: dict

ImageStitchingStep
----------------

.. py:class:: ImageStitchingStep(name="Image Stitching", input_dir=None, positions_dir=None, output_dir=None, processing_args=None)

   A specialized Step for stitching images.

   :param name: Name of the step
   :type name: str
   :param input_dir: Input directory
   :type input_dir: str or Path
   :param positions_dir: Directory containing position files
   :type positions_dir: str or Path
   :param output_dir: Output directory
   :type output_dir: str or Path
   :param processing_args: Additional arguments for the processing function
   :type processing_args: dict

   .. py:method:: process(context)

      Stitch images using the positions file from the context.

      :param context: The processing context
      :type context: :class:`~ezstitcher.core.pipeline.ProcessingContext`
      :return: The updated processing context
      :rtype: :class:`~ezstitcher.core.pipeline.ProcessingContext`
