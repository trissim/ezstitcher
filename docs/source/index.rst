Welcome to EZStitcher's Documentation
=====================================

EZStitcher is a Python package for stitching microscopy images with support for Z-stacks, multi-channel fluorescence, and advanced focus detection.

.. image:: _static/ezstitcher_logo.png
   :width: 400
   :alt: EZStitcher Logo

Getting Started Quickly
--------------------

The fastest way to get started with EZStitcher is to use the ``EZ module``:

.. code-block:: python

    from ezstitcher import stitch_plate

    # Stitch a plate with a single function call
    stitch_plate("path/to/microscopy/data")

For a complete quick start guide, see :doc:`getting_started/quick_start`.

Key Features
------------

- **Simplified Interface for Non-Coders**: Process and stitch images with minimal code using the EZ module
- **Multi-channel fluorescence support**: Process and stitch multiple fluorescence channels
- **Z-stack handling**: Process 3D image stacks with various projection methods
- **Advanced focus detection**: Find the best focused plane in Z-stacks
- **Flexible preprocessing**: Apply custom preprocessing to images
- **Multiple microscope support**: Works with ImageXpress and Opera Phenix microscopes
- **Automatic detection**: Automatically detect microscope type and image organization
- **Object-oriented API**: Clean, modular design for easy customization

Key Resources
-----------

* :doc:`getting_started/quick_start` - Get started in minutes with a minimal example
* :doc:`user_guide/ez_module` - Learn about the simplified interface for non-coders (recommended for beginners)
* :doc:`user_guide/transitioning_from_ez` - Bridge the gap between the EZ module and custom pipelines
* :doc:`user_guide/introduction` - Learn about ezstitcher's architecture and concepts

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/quick_start
   getting_started/installation

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/introduction
   user_guide/ez_module
   user_guide/transitioning_from_ez
   user_guide/intermediate_usage
   user_guide/advanced_usage
   user_guide/best_practices
   user_guide/integration

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   concepts/overview
   concepts/basic_microscopy
   concepts/architecture_overview
   concepts/pipeline_orchestrator
   concepts/pipeline
   concepts/pipeline_factory
   concepts/step
   concepts/function_handling
   concepts/processing_context
   concepts/directory_structure

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/index

.. toctree::
   :maxdepth: 2
   :caption: Appendices

   appendices/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
