Configuration
=============

.. module:: ezstitcher.core.config

This module contains configuration classes for ezstitcher.

PipelineConfig
-------------

.. py:class:: PipelineConfig

   Configuration for the pipeline orchestrator.

   .. py:attribute:: processed_dir_suffix
      :type: str
      :value: "_processed"

      Suffix for the processed directory.

   .. py:attribute:: post_processed_dir_suffix
      :type: str
      :value: "_post_processed"

      Suffix for the post-processed directory.

   .. py:attribute:: positions_dir_suffix
      :type: str
      :value: "_positions"

      Suffix for the positions directory.

   .. py:attribute:: stitched_dir_suffix
      :type: str
      :value: "_stitched"

      Suffix for the stitched directory.

   .. py:attribute:: well_filter
      :type: list or None
      :value: None

      Optional list of wells to process.

   .. py:attribute:: reference_channels
      :type: list
      :value: ["1"]

      List of channels to use for position generation.

   .. py:attribute:: reference_processing
      :type: callable, list, dict, or None
      :value: None

      Preprocessing functions for reference channels.

   .. py:attribute:: reference_composite_weights
      :type: dict or None
      :value: None

      Weights for creating composite images from reference channels.

   .. py:attribute:: final_processing
      :type: dict or None
      :value: None

      Preprocessing functions for final stitching.

   .. py:attribute:: stitcher
      :type: StitcherConfig
      :value: StitcherConfig()

      Configuration for the Stitcher class.

   .. py:attribute:: reference_flatten
      :type: str or callable
      :value: "max_projection"

      Method for flattening Z-stacks for position generation.

   .. py:attribute:: stitch_flatten
      :type: str, callable, or None
      :value: None

      Method for flattening Z-stacks for final stitching.

   .. py:attribute:: save_reference
      :type: bool
      :value: True

      Whether to save reference images.

   .. py:attribute:: additional_projections
      :type: list or None
      :value: None

      List of additional projections to create.

   .. py:attribute:: focus_method
      :type: str
      :value: "combined"

      Focus detection method.

   .. py:attribute:: focus_config
      :type: FocusAnalyzerConfig
      :value: FocusAnalyzerConfig()

      Configuration for the FocusAnalyzer class.

StitcherConfig
------------

.. py:class:: StitcherConfig

   Configuration for the Stitcher class.

   .. py:attribute:: tile_overlap
      :type: float
      :value: 10.0

      Percentage overlap between tiles.

   .. py:attribute:: tile_overlap_x
      :type: float or None
      :value: None

      Percentage overlap between tiles in the x direction. If None, tile_overlap is used.

   .. py:attribute:: tile_overlap_y
      :type: float or None
      :value: None

      Percentage overlap between tiles in the y direction. If None, tile_overlap is used.

   .. py:attribute:: max_shift
      :type: int
      :value: 50

      Maximum allowed shift in pixels.

   .. py:attribute:: margin_ratio
      :type: float
      :value: 0.1

      Ratio of image size to use as margin for blending.

   .. py:attribute:: pixel_size
      :type: float
      :value: 1.0

      Pixel size in micrometers.

FocusAnalyzerConfig
-----------------

.. py:class:: FocusAnalyzerConfig

   Configuration for the FocusAnalyzer class.

   .. py:attribute:: method
      :type: str
      :value: "combined"

      Focus detection method. Options: "combined", "nvar", "normalized_variance", "lap", "laplacian", "ten", "tenengrad", "fft".

   .. py:attribute:: roi
      :type: tuple or None
      :value: None

      Optional region of interest as (x, y, width, height).

   .. py:attribute:: weights
      :type: dict or None
      :value: None

      Optional dictionary with weights for each metric in combined focus measure.

ImagePreprocessorConfig
---------------------

.. py:class:: ImagePreprocessorConfig

   Configuration for the ImagePreprocessor class.

   .. py:attribute:: preprocessing_funcs
      :type: dict
      :value: {}

      Dictionary mapping channels to preprocessing functions.

   .. py:attribute:: composite_weights
      :type: dict or None
      :value: None

      Optional dictionary with weights for each channel in composite images.

Legacy Configuration Classes
--------------------------

The following classes are maintained for backward compatibility:

.. py:class:: PlateProcessorConfig

   Configuration for the PlateProcessor class.

   .. py:attribute:: reference_channels
      :type: list
      :value: ["1"]

      List of channels to use for position generation.

   .. py:attribute:: well_filter
      :type: list or None
      :value: None

      Optional list of wells to process.

   .. py:attribute:: use_reference_positions
      :type: bool
      :value: False

      Whether to use reference positions.

   .. py:attribute:: microscope_type
      :type: str
      :value: "auto"

      Type of microscope ('auto', 'ImageXpress', 'OperaPhenix', etc.).

   .. py:attribute:: rename_files
      :type: bool
      :value: True

      Whether to rename files with consistent padding.

   .. py:attribute:: padding_width
      :type: int
      :value: 3

      Width to pad site numbers to.

   .. py:attribute:: dry_run
      :type: bool
      :value: False

      Whether to perform a dry run.

   .. py:attribute:: output_dir_suffix
      :type: str
      :value: "_processed"

      Suffix for the output directory.

   .. py:attribute:: positions_dir_suffix
      :type: str
      :value: "_positions"

      Suffix for the positions directory.

   .. py:attribute:: stitched_dir_suffix
      :type: str
      :value: "_stitched"

      Suffix for the stitched directory.

   .. py:attribute:: best_focus_dir_suffix
      :type: str
      :value: "_best_focus"

      Suffix for the best focus directory.

   .. py:attribute:: projections_dir_suffix
      :type: str
      :value: "_Projections"

      Suffix for the projections directory.

   .. py:attribute:: timepoint_dir_name
      :type: str
      :value: "TimePoint_1"

      Name of the timepoint directory.

   .. py:attribute:: preprocessing_funcs
      :type: dict or None
      :value: None

      Dictionary mapping channels to preprocessing functions.

   .. py:attribute:: composite_weights
      :type: dict or None
      :value: None

      Weights for creating composite images.

   .. py:attribute:: stitcher
      :type: StitcherConfig
      :value: StitcherConfig()

      Configuration for the Stitcher class.

   .. py:attribute:: focus_analyzer
      :type: FocusAnalyzerConfig
      :value: FocusAnalyzerConfig()

      Configuration for the FocusAnalyzer class.

   .. py:attribute:: image_preprocessor
      :type: ImagePreprocessorConfig
      :value: ImagePreprocessorConfig()

      Configuration for the ImagePreprocessor class.

   .. py:attribute:: reference_flatten
      :type: str or callable
      :value: "max_projection"

      Method for flattening Z-stacks for position generation.

   .. py:attribute:: stitch_flatten
      :type: str, callable, or None
      :value: None

      Method for flattening Z-stacks for final stitching.

   .. py:attribute:: save_reference
      :type: bool
      :value: True

      Whether to save reference images.

   .. py:attribute:: additional_projections
      :type: list or None
      :value: None

      List of additional projections to create.

   .. py:attribute:: focus_method
      :type: str
      :value: "combined"

      Focus detection method.

.. py:class:: StitchingConfig

   Legacy configuration for stitching.

   .. py:attribute:: reference_channels
      :type: list
      :value: ["1"]

      List of channels to use for position generation.

   .. py:attribute:: tile_overlap
      :type: float
      :value: 10.0

      Percentage overlap between tiles.

   .. py:attribute:: max_shift
      :type: int
      :value: 50

      Maximum allowed shift in pixels.

   .. py:attribute:: focus_detect
      :type: bool
      :value: False

      Whether to detect focus.

   .. py:attribute:: focus_method
      :type: str
      :value: "combined"

      Focus detection method.

   .. py:attribute:: create_projections
      :type: bool
      :value: False

      Whether to create projections.

   .. py:attribute:: stitch_z_reference
      :type: str
      :value: "best_focus"

      Z-stack reference for stitching.

   .. py:attribute:: save_projections
      :type: bool
      :value: True

      Whether to save projections.

   .. py:attribute:: stitch_all_z_planes
      :type: bool
      :value: False

      Whether to stitch all Z-planes.

   .. py:attribute:: well_filter
      :type: list or None
      :value: None

      Optional list of wells to process.

   .. py:attribute:: composite_weights
      :type: dict or None
      :value: None

      Weights for creating composite images.

   .. py:attribute:: preprocessing_funcs
      :type: dict or None
      :value: None

      Dictionary mapping channels to preprocessing functions.

   .. py:attribute:: margin_ratio
      :type: float
      :value: 0.1

      Ratio of image size to use as margin for blending.

.. py:class:: FocusConfig

   Legacy configuration for focus detection.

   .. py:attribute:: method
      :type: str
      :value: "combined"

      Focus detection method.

   .. py:attribute:: roi
      :type: list or None
      :value: None

      Optional region of interest as [x, y, width, height].

.. py:class:: PlateConfig

   Legacy configuration for plate processing.

   .. py:attribute:: plate_folder
      :type: str
      :value: ""

      Path to the plate folder.

   .. py:attribute:: stitching
      :type: StitchingConfig
      :value: StitchingConfig()

      Configuration for stitching.

   .. py:attribute:: reference_flatten
      :type: str or callable
      :value: "max_projection"

      Method for flattening Z-stacks for position generation.

   .. py:attribute:: stitch_flatten
      :type: str, callable, or None
      :value: None

      Method for flattening Z-stacks for final stitching.

   .. py:attribute:: focus
      :type: FocusConfig
      :value: FocusConfig()

      Configuration for focus detection.
