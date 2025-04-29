===========
Basic Usage
===========

.. note::
   **Complexity Level: Beginner**

   This section is designed for beginners who want to get started with EZStitcher quickly.

This page provides an overview of how to use EZStitcher for basic image stitching tasks. If you're looking for a quick start guide, see :doc:`../getting_started/quick_start`.

Getting Started with EZStitcher
-----------------------------

The simplest way to use EZStitcher is through the EZ module, which provides a one-liner function for stitching microscopy images:

.. code-block:: python

   from ezstitcher import stitch_plate

   # Stitch a plate with default settings
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

Understanding Key Concepts
-----------------------

**Plates and Wells**

EZStitcher processes microscopy data organized in plates and wells. A plate contains multiple wells, and each well contains multiple images.

**Images and Channels**

Microscopy images can have multiple channels (e.g., DAPI, GFP, RFP) and Z-stacks (multiple focal planes).

**Processing Steps**

Behind the scenes, EZStitcher processes images through a series of steps, such as:

- Normalization: Adjusting image intensity for consistent visualization
- Z-flattening: Converting 3D Z-stacks into 2D images
- Channel compositing: Combining multiple channels into a single image
- Position generation: Finding the relative positions of tiles
- Image stitching: Combining tiles into a complete image

The EZ module handles all these steps automatically, so you don't need to worry about them unless you need more control.

When You Need More Control
-----------------------

If you need more flexibility than the EZ module provides:

1. First, explore all the options available in the EZ module (see the Key Parameters section above)
2. If you still need more control, see :doc:`transitioning_from_ez` to learn how to bridge the gap to custom pipelines
3. For even more advanced usage, see :doc:`intermediate_usage` for creating custom pipelines with wrapped steps

For detailed API documentation of the EZ module, see :doc:`../api/ez`.
