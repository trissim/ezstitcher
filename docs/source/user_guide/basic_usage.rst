===========
Basic Usage
===========

.. note::
   **Complexity Level: Beginner**

   This section is designed for beginners who want to understand the basic concepts of EZStitcher.

This page provides an overview of the basic concepts and usage patterns in EZStitcher. If you're looking for a quick start guide, see :doc:`../getting_started/quick_start`.

.. important::
   For most users, especially beginners, we recommend using the :doc:`ez_module` which provides a simplified interface with minimal code.

Understanding the Three-Tier Approach
-----------------------------------

As described in the :ref:`three-tier-approach` section of the introduction, EZStitcher offers three main approaches for creating stitching pipelines:

1. **EZ Module (Beginner Level)**: A simplified, one-liner interface for beginners and non-coders
2. **Custom Pipelines with Wrapped Steps (Intermediate Level)**: More flexibility and control using wrapped steps (NormStep, ZFlatStep, etc.)
3. **Library Extension with Base Step (Advanced Level)**: For advanced users who need to understand implementation details

This page focuses on the basic concepts that apply to all three approaches.

Key Concepts
----------

**Plates and Wells**

EZStitcher processes microscopy data organized in plates and wells. A plate contains multiple wells, and each well contains multiple images.

**Images and Channels**

Microscopy images can have multiple channels (e.g., DAPI, GFP, RFP) and Z-stacks (multiple focal planes).

**Processing Steps**

EZStitcher processes images through a series of steps, such as:

- Normalization: Adjusting image intensity for consistent visualization
- Z-flattening: Converting 3D Z-stacks into 2D images
- Channel compositing: Combining multiple channels into a single image
- Position generation: Finding the relative positions of tiles
- Image stitching: Combining tiles into a complete image

**Pipelines**

A pipeline is a sequence of processing steps that are executed in order. EZStitcher typically uses two pipelines:

1. Position generation pipeline: Processes images and generates position information
2. Assembly pipeline: Uses the position information to stitch images together

Getting Started with the EZ Module
--------------------------------

The simplest way to use EZStitcher is through the EZ module:

.. code-block:: python

   from ezstitcher import stitch_plate

   # Stitch a plate with default settings
   stitch_plate("path/to/plate")

For more information on the EZ module, see :doc:`ez_module`.

Moving Beyond the EZ Module
-------------------------

As your needs become more specialized, you may want more control over the processing steps. The :doc:`transitioning_from_ez` guide helps you bridge the gap between the EZ module and custom pipelines.

Learning Path
-----------

Based on your experience level and needs, here's where to go next:

**For Beginners:**
* For a simplified interface with minimal code, see the :doc:`ez_module` guide
* When you're ready to move beyond the EZ module, see :doc:`transitioning_from_ez`

**For Intermediate Users:**
* To learn how to create custom pipelines with wrapped steps, see :doc:`intermediate_usage`
* For best practices at all levels, see :doc:`best_practices`

**For Advanced Users:**
* To understand how wrapped steps are implemented using the base Step class, see :doc:`advanced_usage`
* For detailed information about the architecture, see :doc:`../concepts/architecture_overview`

**For All Users:**
* For more information on the three-tier approach, see the :ref:`three-tier-approach` section in the introduction
