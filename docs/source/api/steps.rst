Steps
=====

.. module:: ezstitcher.core.steps

This module contains the Step class and its specialized subclasses for the EZStitcher pipeline architecture.

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

ImageStitchingStep
----------------

.. py:class:: ImageStitchingStep(name="Image Stitching", input_dir=None, positions_dir=None, output_dir=None, processing_args=None)

   A specialized Step for stitching images using position files.

   :param name: Name of the step
   :type name: str
   :param input_dir: Input directory containing images to stitch
   :type input_dir: str or Path
   :param positions_dir: Directory containing position files (optional, can be provided in context)
   :type positions_dir: str or Path
   :param output_dir: Output directory for stitched images
   :type output_dir: str or Path
   :param processing_args: Additional arguments for the stitching function
   :type processing_args: dict

   .. py:method:: process(context)

      Stitch images using the positions file from the context.

      This step:
      1. Locates the positions file for the current well
      2. Loads images according to the positions file
      3. Stitches the images together
      4. Saves the stitched image to the output directory

      :param context: The processing context
      :type context: :class:`~ezstitcher.core.pipeline.ProcessingContext`
      :return: The updated processing context
      :rtype: :class:`~ezstitcher.core.pipeline.ProcessingContext`
