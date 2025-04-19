# Diff 6: Update Documentation

Status: Not Started
Progress: 0%
Last Updated: 2023-05-15
Dependencies: [plans/refactor/07-diff-05-update-tests.md]

## Overview

This diff updates the documentation to reflect the changes made in the refactoring. It includes updates to the API documentation, examples, and user guide.

## Files to Modify

### 1. `docs/source/api/processing_pipeline.rst`

```diff
Processing Pipeline
==================

.. module:: ezstitcher.core.processing_pipeline

This module contains the core processing pipeline for EZStitcher.

PipelineOrchestrator
-------------------

.. py:class:: PipelineOrchestrator(config=None, fs_manager=None, image_preprocessor=None, focus_analyzer=None)

   A robust pipeline orchestrator for microscopy image processing.

   The pipeline follows a clear, linear flow:

   1. Load and organize images
   2. Process patterns with variable components
   3. Generate stitching positions
   4. Stitch images

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

-   .. py:method:: process_reference_images(well, dirs)
+   .. py:method:: process_reference_images(well, dirs) -> ProcessingContext

      Process images for position generation.

      :param well: Well identifier
      :type well: str
      :param dirs: Dictionary of directories
      :type dirs: dict
+     :return: Processing context with results
+     :rtype: :class:`~ezstitcher.core.processing.context.ProcessingContext`

-   .. py:method:: process_final_images(well, dirs)
+   .. py:method:: process_final_images(well, dirs) -> ProcessingContext

      Process images for final stitching.

      :param well: Well identifier
      :type well: str
      :param dirs: Dictionary of directories
      :type dirs: dict
+     :return: Processing context with results
+     :rtype: :class:`~ezstitcher.core.processing.context.ProcessingContext`

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

      Unified processing using zstack_processor.

      :param input_dir: Input directory
      :type input_dir: str or Path
      :param output_dir: Output directory
      :type output_dir: str or Path
      :param patterns: List of file patterns
      :type patterns: list
      :param processing_funcs: Processing functions to apply (optional)
      :type processing_funcs: callable, list, optional
      :param kwargs: Additional arguments to pass to processing functions
      :type kwargs: dict
      :return: Paths to created images
      :rtype: list
```

### 2. `docs/source/api/processing/index.rst`

```rst
Processing
=========

.. module:: ezstitcher.core.processing

This module contains classes for processing images in a pipeline.

.. toctree::
   :maxdepth: 2

   context
   pipeline
   step
   strategy
   steps
   strategies
```

### 3. `docs/source/api/processing/context.rst`

```rst
Processing Context
================

.. module:: ezstitcher.core.processing.context

This module contains the ProcessingContext class for storing processing context.

ProcessingContext
--------------

.. py:class:: ProcessingContext(pipeline_orchestrator, well, dirs, channels=None)

   Context object for processing operations.

   :param pipeline_orchestrator: Pipeline orchestrator
   :type pipeline_orchestrator: :class:`~ezstitcher.core.processing_pipeline.PipelineOrchestrator`
   :param well: Well identifier
   :type well: str
   :param dirs: Dictionary of directories
   :type dirs: dict
   :param channels: List of channels to process
   :type channels: list, optional

   .. py:attribute:: pipeline_orchestrator
      :type: PipelineOrchestrator

      Pipeline orchestrator.

   .. py:attribute:: well
      :type: str

      Well identifier.

   .. py:attribute:: dirs
      :type: dict

      Dictionary of directories.

   .. py:attribute:: channels
      :type: list

      List of channels to process.

   .. py:attribute:: results
      :type: dict

      Dictionary of results.

   .. py:property:: input_dir
      :type: Path

      Input directory.

   .. py:property:: output_dir
      :type: Path

      Output directory.

   .. py:property:: post_processed_dir
      :type: Path

      Post-processed directory.

   .. py:property:: positions_dir
      :type: Path

      Positions directory.

   .. py:property:: stitched_dir
      :type: Path

      Stitched directory.

   .. py:method:: set_result(key, value)

      Set a result value.

      :param key: Result key
      :type key: str
      :param value: Result value
      :type value: Any

   .. py:method:: get_result(key, default=None)

      Get a result value.

      :param key: Result key
      :type key: str
      :param default: Default value if key is not found
      :type default: Any
      :return: Result value or default
      :rtype: Any
```

### 4. `docs/source/api/processing/pipeline.rst`

```rst
Processing Pipeline
================

.. module:: ezstitcher.core.processing.pipeline

This module contains the ProcessingPipeline class for chaining processing steps.

ProcessingPipeline
--------------

.. py:class:: ProcessingPipeline

   Pipeline for processing images.

   .. py:method:: add_step(step)

      Add a processing step to the pipeline.

      :param step: Processing step to add
      :type step: :class:`~ezstitcher.core.processing.step.ProcessingStep`
      :return: Self for chaining
      :rtype: :class:`~ezstitcher.core.processing.pipeline.ProcessingPipeline`

   .. py:method:: process(context)

      Process the input context through all steps in the pipeline.

      :param context: Processing context object
      :type context: :class:`~ezstitcher.core.processing.context.ProcessingContext`
      :return: Updated context object
      :rtype: :class:`~ezstitcher.core.processing.context.ProcessingContext`
```

### 5. `docs/source/api/processing/step.rst`

```rst
Processing Step
============

.. module:: ezstitcher.core.processing.step

This module contains the ProcessingStep abstract base class for processing steps.

ProcessingStep
----------

.. py:class:: ProcessingStep

   Abstract base class for processing steps in the pipeline.

   .. py:method:: process(context)

      Process the input context and return the updated context.

      :param context: Processing context object
      :type context: :class:`~ezstitcher.core.processing.context.ProcessingContext`
      :return: Updated context object
      :rtype: :class:`~ezstitcher.core.processing.context.ProcessingContext`
```

### 6. `docs/source/api/processing/strategy.rst`

```rst
Processing Strategy
===============

.. module:: ezstitcher.core.processing.strategy

This module contains the ProcessingStrategy protocol for processing strategies.

ProcessingStrategy
--------------

.. py:class:: ProcessingStrategy

   Protocol for processing strategies.

   .. py:method:: create_pipeline(context)

      Create a processing pipeline for the given context.

      :param context: Processing context object
      :type context: :class:`~ezstitcher.core.processing.context.ProcessingContext`
      :return: Processing pipeline
      :rtype: :class:`~ezstitcher.core.processing.pipeline.ProcessingPipeline`
```

### 7. `docs/source/api/processing/steps.rst`

```rst
Processing Steps
=============

.. module:: ezstitcher.core.processing.steps

This module contains concrete implementations of processing steps.

ZStackFlatteningStep
----------------

.. py:class:: ZStackFlatteningStep(method, focus_analyzer=None)

   Flattens Z-stacks using the specified method.

   :param method: Flattening method (max_projection, mean_projection, best_focus)
   :type method: str
   :param focus_analyzer: Focus analyzer for best_focus method
   :type focus_analyzer: :class:`~ezstitcher.core.focus_analyzer.FocusAnalyzer`, optional

   .. py:method:: process(context)

      Process the input context and return the updated context.

      :param context: Processing context object
      :type context: :class:`~ezstitcher.core.processing.context.ProcessingContext`
      :return: Updated context object
      :rtype: :class:`~ezstitcher.core.processing.context.ProcessingContext`

ChannelProcessingStep
-----------------

.. py:class:: ChannelProcessingStep(processing_funcs=None)

   Processes images for each channel using the specified functions.

   :param processing_funcs: Dictionary mapping channels to processing functions
   :type processing_funcs: dict, optional

   .. py:method:: process(context)

      Process the input context and return the updated context.

      :param context: Processing context object
      :type context: :class:`~ezstitcher.core.processing.context.ProcessingContext`
      :return: Updated context object
      :rtype: :class:`~ezstitcher.core.processing.context.ProcessingContext`

CompositeCreationStep
-----------------

.. py:class:: CompositeCreationStep(weights=None)

   Creates composite images from multiple channels.

   :param weights: Dictionary mapping channels to weights
   :type weights: dict, optional

   .. py:method:: process(context)

      Process the input context and return the updated context.

      :param context: Processing context object
      :type context: :class:`~ezstitcher.core.processing.context.ProcessingContext`
      :return: Updated context object
      :rtype: :class:`~ezstitcher.core.processing.context.ProcessingContext`
```

### 8. `docs/source/api/processing/strategies.rst`

```rst
Processing Strategies
=================

.. module:: ezstitcher.core.processing.strategies

This module contains concrete implementations of processing strategies.

ReferenceProcessingStrategy
-----------------------

.. py:class:: ReferenceProcessingStrategy(config, image_preprocessor, focus_analyzer)

   Strategy for processing reference images.

   :param config: Pipeline configuration
   :type config: :class:`~ezstitcher.core.config.PipelineConfig`
   :param image_preprocessor: Image preprocessor
   :type image_preprocessor: :class:`~ezstitcher.core.image_preprocessor.ImagePreprocessor`
   :param focus_analyzer: Focus analyzer
   :type focus_analyzer: :class:`~ezstitcher.core.focus_analyzer.FocusAnalyzer`

   .. py:method:: create_pipeline(context)

      Create a processing pipeline for the given context.

      :param context: Processing context object
      :type context: :class:`~ezstitcher.core.processing.context.ProcessingContext`
      :return: Processing pipeline
      :rtype: :class:`~ezstitcher.core.processing.pipeline.ProcessingPipeline`

FinalProcessingStrategy
------------------

.. py:class:: FinalProcessingStrategy(config, image_preprocessor, focus_analyzer)

   Strategy for processing final images.

   :param config: Pipeline configuration
   :type config: :class:`~ezstitcher.core.config.PipelineConfig`
   :param image_preprocessor: Image preprocessor
   :type image_preprocessor: :class:`~ezstitcher.core.image_preprocessor.ImagePreprocessor`
   :param focus_analyzer: Focus analyzer
   :type focus_analyzer: :class:`~ezstitcher.core.focus_analyzer.FocusAnalyzer`

   .. py:method:: create_pipeline(context)

      Create a processing pipeline for the given context.

      :param context: Processing context object
      :type context: :class:`~ezstitcher.core.processing.context.ProcessingContext`
      :return: Processing pipeline
      :rtype: :class:`~ezstitcher.core.processing.pipeline.ProcessingPipeline`
```

### 9. `docs/source/user_guide/image_processing.rst`

```diff
Image Processing
==============

EZStitcher provides a flexible image processing pipeline for microscopy images.

Basic Processing
-------------

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_method="combined"
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

+ Advanced Processing with Pipeline and Strategy
+ ------------------------------------------
+
+ EZStitcher provides a flexible pipeline and strategy system for customizing the processing pipeline.
+
+ .. code-block:: python
+
+     from ezstitcher.core.config import PipelineConfig
+     from ezstitcher.core.processing_pipeline import PipelineOrchestrator
+     from ezstitcher.core.processing.context import ProcessingContext
+     from ezstitcher.core.processing.pipeline import ProcessingPipeline
+     from ezstitcher.core.processing.steps import ZStackFlatteningStep, ChannelProcessingStep, CompositeCreationStep
+     from ezstitcher.core.processing.strategies import ReferenceProcessingStrategy, FinalProcessingStrategy
+
+     # Create configuration
+     config = PipelineConfig(
+         reference_channels=["1", "2"],
+         reference_flatten="max_projection",
+         stitch_flatten="best_focus",
+         focus_method="combined"
+     )
+
+     # Create pipeline orchestrator
+     orchestrator = PipelineOrchestrator(config)
+
+     # Create context
+     context = ProcessingContext(
+         pipeline_orchestrator=orchestrator,
+         well="A01",
+         dirs={
+             'input': "path/to/input",
+             'processed': "path/to/processed",
+             'post_processed': "path/to/post_processed",
+             'positions': "path/to/positions",
+             'stitched': "path/to/stitched"
+         },
+         channels=["1", "2"]
+     )
+
+     # Create strategy
+     strategy = ReferenceProcessingStrategy(
+         config=config,
+         image_preprocessor=orchestrator.image_preprocessor,
+         focus_analyzer=orchestrator.focus_analyzer
+     )
+
+     # Create and run pipeline
+     pipeline = strategy.create_pipeline(context)
+     result = pipeline.process(context)
+
+     # Access results
+     flattened_files = result.get_result('flattened_files')
+     processed_files = result.get_result('processed_files')
+     composite_files = result.get_result('composite_files')
+
+ Custom Processing Steps
+ --------------------
+
+ You can create custom processing steps by subclassing the `ProcessingStep` class.
+
+ .. code-block:: python
+
+     from ezstitcher.core.processing.context import ProcessingContext
+     from ezstitcher.core.processing.step import ProcessingStep
+
+     class CustomProcessingStep(ProcessingStep):
+         """Custom processing step."""
+
+         def __init__(self, param1, param2):
+             """Initialize with parameters."""
+             self.param1 = param1
+             self.param2 = param2
+
+         def process(self, context):
+             """Process the input context and return the updated context."""
+             # Process images
+             result = context.pipeline_orchestrator.process_patterns_with_variable_components(
+                 input_dir=context.input_dir,
+                 output_dir=context.output_dir,
+                 well_filter=[context.well],
+                 variable_components=['site'],
+                 processing_funcs=lambda images: [self.custom_process(img) for img in images]
+             ).get(context.well, [])
+
+             # Update context with result
+             context.set_result('custom_files', result)
+             return context
+
+         def custom_process(self, image):
+             """Custom processing function."""
+             # Implement custom processing logic
+             return image
+
+ Custom Processing Strategies
+ ------------------------
+
+ You can create custom processing strategies by implementing the `ProcessingStrategy` protocol.
+
+ .. code-block:: python
+
+     from ezstitcher.core.processing.context import ProcessingContext
+     from ezstitcher.core.processing.pipeline import ProcessingPipeline
+     from ezstitcher.core.processing.steps import ZStackFlatteningStep, ChannelProcessingStep
+
+     class CustomProcessingStrategy:
+         """Custom processing strategy."""
+
+         def __init__(self, config, image_preprocessor, focus_analyzer):
+             """Initialize with parameters."""
+             self.config = config
+             self.image_preprocessor = image_preprocessor
+             self.focus_analyzer = focus_analyzer
+
+         def create_pipeline(self, context):
+             """Create a processing pipeline for the given context."""
+             # Create pipeline
+             pipeline = ProcessingPipeline()
+
+             # Add custom steps
+             pipeline.add_step(ZStackFlatteningStep(
+                 self.config.reference_flatten,
+                 self.focus_analyzer
+             ))
+             pipeline.add_step(CustomProcessingStep(
+                 param1=self.config.param1,
+                 param2=self.config.param2
+             ))
+
+             return pipeline
```

## Implementation Notes

1. The API documentation is updated to reflect the changes made in the refactoring.

2. New documentation is added for the new classes and modules.

3. The user guide is updated to include examples of using the new pipeline and strategy classes.

4. The documentation follows the same style and format as the existing documentation.

## Testing Plan

1. Build the documentation to ensure that it is formatted correctly.

2. Review the documentation to ensure that it accurately reflects the changes made in the refactoring.

3. Test the examples in the user guide to ensure that they work correctly.

## Validation Criteria

1. The documentation builds without errors
2. The documentation accurately reflects the changes made in the refactoring
3. The examples in the user guide work correctly
