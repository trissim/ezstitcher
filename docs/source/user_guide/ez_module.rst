==========
EZ Module
==========

The EZ module provides a simplified interface for common stitching workflows,
making it easier for non-coders to use EZStitcher.

Quick Start
----------

The simplest way to stitch a plate of microscopy images is to use the ``stitch_plate`` function:

.. code-block:: python

   from ezstitcher import stitch_plate
   
   # Stitch a plate with default settings
   stitch_plate("path/to/plate")

This will:

1. Automatically detect the plate format
2. Apply normalization
3. Handle Z-stacks if present
4. Generate positions and stitch images
5. Save the output to a new directory

Configuration Options
-------------------

You can customize the stitching process with various options:

.. code-block:: python

   from ezstitcher import stitch_plate
   
   # Stitch with custom options
   stitch_plate(
       "path/to/plate",
       output_path="path/to/output",
       normalize=True,
       flatten_z=True,
       z_method="focus",
       channel_weights=[0.7, 0.3, 0]
   )

More Control with EZStitcher Class
--------------------------------

For more control, you can use the ``EZStitcher`` class:

.. code-block:: python

   from ezstitcher import EZStitcher
   
   # Create stitcher
   stitcher = EZStitcher("path/to/plate")
   
   # Customize options
   stitcher.set_options(
       normalize=True,
       z_method="focus",
       channel_weights=[0.7, 0.3, 0]
   )
   
   # Run stitching
   stitcher.stitch()

Common Use Cases
===============

Single-Channel Stitching
-----------------------

.. code-block:: python

   from ezstitcher import stitch_plate
   
   # Stitch a single-channel plate
   stitch_plate("path/to/single_channel_plate")

Multi-Channel Stitching
---------------------

.. code-block:: python

   from ezstitcher import stitch_plate
   
   # Stitch a multi-channel plate
   # Channel weights determine how channels are combined for position generation
   stitch_plate(
       "path/to/multi_channel_plate",
       channel_weights=[0.7, 0.3, 0]  # 70% channel 1, 30% channel 2, 0% channel 3
   )

Z-Stack Stitching
---------------

.. code-block:: python

   from ezstitcher import stitch_plate
   
   # Stitch a Z-stack plate with maximum intensity projection
   stitch_plate(
       "path/to/z_stack_plate",
       flatten_z=True,
       z_method="max"
   )
   
   # Stitch a Z-stack plate with focus-based projection
   stitch_plate(
       "path/to/z_stack_plate",
       flatten_z=True,
       z_method="focus"
   )

Processing Specific Wells
-----------------------

.. code-block:: python

   from ezstitcher import stitch_plate
   
   # Process only specific wells
   stitch_plate(
       "path/to/plate",
       well_filter=["A01", "B02", "C03"]
   )

Troubleshooting
==============

Common Issues
-----------

**No output generated**

- Check that the input path exists and contains microscopy images
- Verify that the microscope format is supported
- Check for error messages in the console output

**Z-stacks not detected**

- Explicitly set ``flatten_z=True`` if auto-detection fails
- Check that Z-stack images follow the expected naming convention

**Poor stitching quality**

- Try different normalization settings
- Adjust channel weights to emphasize channels with more features
- Try different Z-flattening methods

API Reference
============

EZStitcher Class
--------------

.. py:class:: EZStitcher(input_path, output_path=None, normalize=True, flatten_z=None, z_method="max", channel_weights=None, well_filter=None)

   Simplified interface for microscopy image stitching.
   
   This class provides an easy-to-use interface for common stitching workflows,
   hiding the complexity of pipelines and orchestrators.
   
   :param input_path: Path to the plate folder
   :type input_path: str or Path
   :param output_path: Path for output (default: input_path + "_stitched")
   :type output_path: str or Path, optional
   :param normalize: Whether to apply normalization
   :type normalize: bool, default=True
   :param flatten_z: Whether to flatten Z-stacks (auto-detected if None)
   :type flatten_z: bool or None, optional
   :param z_method: Method for Z-flattening ("max", "mean", "focus", etc.)
   :type z_method: str, default="max"
   :param channel_weights: Weights for channel compositing (auto-detected if None)
   :type channel_weights: list of float or None, optional
   :param well_filter: List of wells to process (processes all if None)
   :type well_filter: list of str or None, optional
   
   .. py:method:: set_options(**kwargs)
   
      Update configuration options.
      
      :param kwargs: Configuration options to update
      :return: self for method chaining
      
   .. py:method:: stitch()
   
      Run the complete stitching process with current settings.
      
      :return: Path to the output directory
      :rtype: Path

stitch_plate Function
------------------

.. py:function:: stitch_plate(input_path, output_path=None, **kwargs)

   One-liner function to stitch a plate of microscopy images.
   
   :param input_path: Path to the plate folder
   :type input_path: str or Path
   :param output_path: Path for output (default: input_path + "_stitched")
   :type output_path: str or Path, optional
   :param kwargs: Additional options passed to EZStitcher
   :return: Path to the stitched output
   :rtype: Path
