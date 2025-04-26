Pipeline Factories
=================

.. module:: ezstitcher.core.pipeline_factories

This module contains factory functions that create pre-configured pipelines
for common workflows, leveraging specialized steps to reduce boilerplate code.
These factories provide a higher-level interface for creating pipelines,
making it easier to construct pipelines for common use cases.

Basic Stitching Pipeline
-----------------------

.. py:function:: create_basic_stitching_pipeline(input_dir, output_dir=None, normalize=True, normalization_params=None, preprocessing_steps=None, well_filter=None)

   Create a basic stitching pipeline for single-channel, single-Z data.

   This factory creates a pipeline with the following steps:

   1. Optional preprocessing steps
   2. Optional normalization step
   3. Position generation step
   4. Image stitching step

   :param input_dir: Input directory containing images
   :type input_dir: str or Path
   :param output_dir: Output directory for stitched images (default: auto-generated)
   :type output_dir: str or Path, optional
   :param normalize: Whether to include a normalization step (default: True)
   :type normalize: bool
   :param normalization_params: Parameters for normalization (default: {'low_percentile': 0.1, 'high_percentile': 99.9})
   :type normalization_params: dict, optional
   :param preprocessing_steps: Additional preprocessing steps to include before position generation
   :type preprocessing_steps: list, optional
   :param well_filter: Wells to process
   :type well_filter: list, optional
   :return: A list of Pipeline objects ready for execution: [position_pipeline, stitching_pipeline]
   :rtype: list

   **Example:**

   .. code-block:: python

      from ezstitcher.core.pipeline_factories import create_basic_stitching_pipeline

      # Create a basic stitching pipeline
      pipelines = create_basic_stitching_pipeline(
          input_dir="path/to/images",
          output_dir="path/to/output"
      )

      # Run the pipelines
      orchestrator.run(pipelines=pipelines)

Multi-Channel Stitching Pipeline
-----------------------------

.. py:function:: create_multichannel_stitching_pipeline(input_dir, output_dir=None, normalize=True, normalization_params=None, composite_weights=None, preprocessing_steps=None, well_filter=None, stitch_channels_separately=False)

   Create a stitching pipeline for multi-channel data.

   This factory creates a pipeline with the following steps:

   1. Optional preprocessing steps
   2. Optional normalization step
   3. Channel compositing step (for position generation)
   4. Position generation step
   5. Image stitching step (either on composite or individual channels)

   :param input_dir: Input directory containing images
   :type input_dir: str or Path
   :param output_dir: Output directory for stitched images (default: auto-generated)
   :type output_dir: str or Path, optional
   :param normalize: Whether to include a normalization step (default: True)
   :type normalize: bool
   :param normalization_params: Parameters for normalization (default: {'low_percentile': 0.1, 'high_percentile': 99.9})
   :type normalization_params: dict, optional
   :param composite_weights: Weights for channel compositing (default: equal weights)
   :type composite_weights: list, optional
   :param preprocessing_steps: Additional preprocessing steps to include before position generation
   :type preprocessing_steps: list, optional
   :param well_filter: Wells to process
   :type well_filter: list, optional
   :param stitch_channels_separately: Whether to stitch each channel separately (default: False)
   :type stitch_channels_separately: bool
   :return: A list of Pipeline objects ready for execution
   :rtype: list

   **Example:**

   .. code-block:: python

      from ezstitcher.core.pipeline_factories import create_multichannel_stitching_pipeline

      # Create a multi-channel stitching pipeline
      pipelines = create_multichannel_stitching_pipeline(
          input_dir="path/to/images",
          composite_weights=[0.7, 0.3],
          stitch_channels_separately=True
      )

      # Run the pipelines
      orchestrator.run(pipelines=pipelines)

Z-Stack Stitching Pipeline
-----------------------

.. py:function:: create_zstack_stitching_pipeline(input_dir, output_dir=None, z_processing_method="projection", z_processing_options=None, normalize=True, normalization_params=None, preprocessing_steps=None, well_filter=None, stitch_original_zstack=False)

   Create a stitching pipeline for Z-stack data.

   This factory creates a pipeline with the following steps:

   1. Optional preprocessing steps
   2. Z-stack processing step (projection or focus)
   3. Optional normalization step
   4. Position generation step
   5. Image stitching step (either on processed or original Z-stack)

   :param input_dir: Input directory containing images
   :type input_dir: str or Path
   :param output_dir: Output directory for stitched images (default: auto-generated)
   :type output_dir: str or Path, optional
   :param z_processing_method: Method for Z-stack processing ("projection" or "focus", default: "projection")
   :type z_processing_method: str
   :param z_processing_options: Options for Z-stack processing:
                               - For projection: {'method': 'max'} (default)
                               - For focus: {'metric': 'combined'} (default)
   :type z_processing_options: dict, optional
   :param normalize: Whether to include a normalization step (default: True)
   :type normalize: bool
   :param normalization_params: Parameters for normalization (default: {'low_percentile': 0.1, 'high_percentile': 99.9})
   :type normalization_params: dict, optional
   :param preprocessing_steps: Additional preprocessing steps to include before Z-stack processing
   :type preprocessing_steps: list, optional
   :param well_filter: Wells to process
   :type well_filter: list, optional
   :param stitch_original_zstack: Whether to stitch the original Z-stack (default: False)
   :type stitch_original_zstack: bool
   :return: A list of Pipeline objects ready for execution
   :rtype: list

   **Example:**

   .. code-block:: python

      from ezstitcher.core.pipeline_factories import create_zstack_stitching_pipeline

      # Create a Z-stack stitching pipeline with maximum intensity projection
      pipelines = create_zstack_stitching_pipeline(
          input_dir="path/to/images",
          z_processing_method="projection",
          z_processing_options={'method': 'max'}
      )

      # Run the pipelines
      orchestrator.run(pipelines=pipelines)

      # Create a Z-stack stitching pipeline with focus selection
      pipelines = create_zstack_stitching_pipeline(
          input_dir="path/to/images",
          z_processing_method="focus",
          z_processing_options={'metric': 'laplacian'}
      )

      # Run the pipelines
      orchestrator.run(pipelines=pipelines)

Focus Stitching Pipeline
-----------------------

.. py:function:: create_focus_stitching_pipeline(input_dir, output_dir=None, focus_metric="combined", focus_roi=None, focus_weights=None, normalize=True, normalization_params=None, preprocessing_steps=None, well_filter=None, stitch_original_zstack=False)

   Create a stitching pipeline for Z-stack data with focus selection.

   This factory creates a pipeline with the following steps:

   1. Optional preprocessing steps
   2. Focus selection step
   3. Optional normalization step
   4. Position generation step
   5. Image stitching step (either on focused or original Z-stack)

   :param input_dir: Input directory containing images
   :type input_dir: str or Path
   :param output_dir: Output directory for stitched images (default: auto-generated)
   :type output_dir: str or Path, optional
   :param focus_metric: Focus metric to use ("laplacian", "sobel", "variance", "combined", default: "combined")
   :type focus_metric: str
   :param focus_roi: Region of interest for focus selection (default: None, uses entire image)
   :type focus_roi: dict, optional
   :param focus_weights: Weights for combined focus metric (default: None, uses equal weights)
   :type focus_weights: dict, optional
   :param normalize: Whether to include a normalization step (default: True)
   :type normalize: bool
   :param normalization_params: Parameters for normalization (default: {'low_percentile': 0.1, 'high_percentile': 99.9})
   :type normalization_params: dict, optional
   :param preprocessing_steps: Additional preprocessing steps to include before focus selection
   :type preprocessing_steps: list, optional
   :param well_filter: Wells to process
   :type well_filter: list, optional
   :param stitch_original_zstack: Whether to stitch the original Z-stack (default: False)
   :type stitch_original_zstack: bool
   :return: A list of Pipeline objects ready for execution
   :rtype: list

   **Example:**

   .. code-block:: python

      from ezstitcher.core.pipeline_factories import create_focus_stitching_pipeline

      # Create a focus stitching pipeline
      pipelines = create_focus_stitching_pipeline(
          input_dir="path/to/images",
          focus_metric="laplacian"
      )

      # Run the pipelines
      orchestrator.run(pipelines=pipelines)