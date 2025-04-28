Welcome to EZStitcher's Documentation
=====================================

EZStitcher is a Python package for stitching microscopy images with support for Z-stacks, multi-channel fluorescence, and advanced focus detection.

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Create orchestrator
    orchestrator = PipelineOrchestrator(plate_path="path/to/plate")

    # Create a factory with default settings
    factory = AutoPipelineFactory(input_dir=orchestrator.workspace_path)

    # Create and run pipelines
    pipelines = factory.create_pipelines()
    orchestrator.run(pipelines=pipelines)

.. image:: _static/ezstitcher_logo.png
   :width: 400
   :alt: EZStitcher Logo

Key Features
------------

- **Pipeline Factory**: Create complete stitching workflows with minimal code
- **Multi-channel fluorescence support**: Process and stitch multiple fluorescence channels
- **Z-stack handling**: Process 3D image stacks with various projection methods
- **Advanced focus detection**: Find the best focused plane in Z-stacks
- **Flexible preprocessing**: Apply custom preprocessing to images
- **Multiple microscope support**: Works with ImageXpress and Opera Phenix microscopes
- **Automatic detection**: Automatically detect microscope type and image organization
- **Object-oriented API**: Clean, modular design for easy customization

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   concepts/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

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
