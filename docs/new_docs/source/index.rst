Welcome to EZStitcher's Documentation
==================================

EZStitcher is a Python package for stitching microscopy images with support for Z-stacks, multi-channel fluorescence, and advanced focus detection.

.. image:: _static/ezstitcher_logo.png
   :width: 400
   :alt: EZStitcher Logo

Key Features
-----------

- **Multi-channel fluorescence support**: Process and stitch multiple fluorescence channels
- **Z-stack handling**: Process 3D image stacks with various projection methods
- **Advanced focus detection**: Find the best focused plane in Z-stacks
- **Flexible preprocessing**: Apply custom preprocessing to images
- **Multiple microscope support**: Works with ImageXpress and Opera Phenix microscopes
- **Automatic detection**: Automatically detect microscope type and image organization
- **Object-oriented API**: Clean, modular design for easy customization
- **Command-line interface**: Easy to use from the command line

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/basic_concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/core_concepts
   user_guide/file_organization
   user_guide/configuration
   user_guide/microscope_support
   user_guide/image_processing
   user_guide/zstack_handling
   user_guide/focus_detection
   user_guide/stitching

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/processing_pipeline
   api/stitcher
   api/focus_analyzer
   api/image_preprocessor
   api/file_system_manager
   api/image_locator
   api/microscope_interfaces
   api/config
   api/microscopes

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_stitching
   examples/zstack_processing
   examples/custom_preprocessing
   examples/custom_focus_detection
   examples/advanced_configuration
   examples/opera_phenix
   examples/imagexpress

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/architecture
   development/contributing
   development/testing
   development/release_process

.. toctree::
   :maxdepth: 2
   :caption: Troubleshooting

   troubleshooting/common_issues
   troubleshooting/error_messages
   troubleshooting/performance

.. toctree::
   :maxdepth: 2
   :caption: Appendices

   appendices/glossary
   appendices/references
   appendices/changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
