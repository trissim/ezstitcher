==========
EZ Module
==========

.. note::
   **Complexity Level: Beginner**

   This section is designed for beginners and non-coders who want to process and stitch images with minimal code.

The EZ module provides a simplified interface for stitching microscopy images with a single function call. It handles all the complexity behind the scenes, making it perfect for beginners and users who want quick results.

.. code-block:: python

   from ezstitcher import stitch_plate

   # Stitch a plate with a single function call
   stitch_plate("path/to/plate")

That's it! This single line will:

1. Automatically detect the plate format
2. Process all channels and Z-stacks appropriately
3. Generate positions and stitch images
4. Save the output to a new directory

Key Parameters
------------

While the default settings work well for most cases, you can customize the behavior with a few key parameters:

.. code-block:: python

   stitch_plate(
       "path/to/plate",                    # Input directory with microscopy images
       output_path="path/to/output",       # Where to save results (optional)
       normalize=True,                     # Apply intensity normalization (default: True)
       flatten_z=True,                     # Flatten Z-stacks to 2D (auto-detected if None)
       z_method="max",                     # How to flatten Z-stacks: "max", "mean", "focus"
       channel_weights=[0.7, 0.3, 0],      # Weights for position finding (auto-detected if None)
       well_filter=["A01", "B02"]          # Process only specific wells (optional)
   )

Z-Stack Processing
---------------

For plates with Z-stacks, you can control how they're flattened:

.. code-block:: python

   # Maximum intensity projection (brightest pixel from each Z-stack)
   stitch_plate("path/to/plate", flatten_z=True, z_method="max")

   # Focus-based projection (selects best-focused plane)
   stitch_plate("path/to/plate", flatten_z=True, z_method="focus")

   # Mean projection (average across Z-planes)
   stitch_plate("path/to/plate", flatten_z=True, z_method="mean")

More Control
---------

For slightly more control while keeping things simple, use the ``EZStitcher`` class:

.. code-block:: python

   from ezstitcher import EZStitcher

   # Create a stitcher
   stitcher = EZStitcher("path/to/plate")

   # Set options
   stitcher.set_options(
       normalize=True,
       z_method="focus"
   )

   # Run stitching
   stitcher.stitch()

Troubleshooting
------------

**Common issues:**

- **No output**: Check that the input path exists and contains microscopy images
- **Z-stacks not detected**: Explicitly set ``flatten_z=True``
- **Poor quality**: Try different ``z_method`` values or adjust ``channel_weights``

When You Need More Control
-----------------------

If you need more flexibility than the EZ module provides:

1. First, explore all the options available in the EZ module (see the Key Parameters section above)
2. If you still need more control, see :doc:`transitioning_from_ez` to learn how to bridge the gap to custom pipelines
3. For even more advanced usage, see :doc:`intermediate_usage` for creating custom pipelines with wrapped steps

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
