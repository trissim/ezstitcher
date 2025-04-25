Steps
=====

.. module:: ezstitcher.core.steps

This module contains the Step class and its specialized subclasses for the EZStitcher pipeline architecture.

For comprehensive information about steps, including:

* Step parameters and their usage
* Variable components and group_by
* Function handling patterns
* Directory resolution
* Best practices

See :doc:`../concepts/step` documentation.

For detailed information about specialized steps like PositionGenerationStep and ImageStitchingStep, see :ref:`specialized-steps` in the :doc:`../concepts/specialized_steps` documentation.

Step
----

.. py:class:: Step(func, variable_components=None, group_by=None, input_dir=None, output_dir=None, well_filter=None, name=None)

   A processing step in a pipeline.

   For detailed information about step parameters and their usage, see :ref:`step-parameters` in the :doc:`../concepts/step` documentation.

   For information about variable components, see :ref:`variable-components` in the :doc:`../concepts/step` documentation.

   For information about the group_by parameter, see :ref:`group-by` in the :doc:`../concepts/step` documentation.

   For best practices when using steps, see :ref:`best-practices-specialized-steps` in the :doc:`../user_guide/best_practices` documentation.

   :param func: The processing function(s) to apply. Can be a single callable, a tuple of (function, kwargs), a list of functions or function tuples, or a dictionary mapping component values to functions or function tuples.
   :type func: callable, tuple, list, or dict
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

.. py:class:: PositionGenerationStep(name="Position Generation", input_dir=None, output_dir=None)

   A specialized Step for generating positions.

   For detailed information about how this step works, see :ref:`position-generation-step` in the :doc:`../concepts/specialized_steps` documentation.

   For information about typical stitching workflows using this step, see :ref:`typical-stitching-workflows` in the :doc:`../concepts/specialized_steps` documentation.

   For best practices when using specialized steps, see :ref:`specialized-steps-best-practices` in the :doc:`../concepts/specialized_steps` documentation.

   :param name: Name of the step (optional)
   :type name: str
   :param input_dir: Input directory (optional)
   :type input_dir: str or Path
   :param output_dir: Output directory for positions files (optional)
   :type output_dir: str or Path

   .. py:method:: process(context)

      Generate positions for stitching and store them in the context.

      :param context: The processing context
      :type context: :class:`~ezstitcher.core.pipeline.ProcessingContext`
      :return: The updated processing context
      :rtype: :class:`~ezstitcher.core.pipeline.ProcessingContext`

ImageStitchingStep
----------------

.. py:class:: ImageStitchingStep(name="Image Stitching", input_dir=None, positions_dir=None, output_dir=None)

   A specialized Step for stitching images using position files.

   For detailed information about how this step works, see :ref:`image-stitching-step` in the :doc:`../concepts/specialized_steps` documentation.

   For information about typical stitching workflows using this step, see :ref:`typical-stitching-workflows` in the :doc:`../concepts/specialized_steps` documentation.

   For best practices when using specialized steps, see :ref:`specialized-steps-best-practices` in the :doc:`../concepts/specialized_steps` documentation.

   :param name: Name of the step (optional)
   :type name: str
   :param input_dir: Input directory containing images to stitch (optional)
   :type input_dir: str or Path
   :param positions_dir: Directory containing position files (optional, can be provided in context)
   :type positions_dir: str or Path
   :param output_dir: Output directory for stitched images (optional)
   :type output_dir: str or Path

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
