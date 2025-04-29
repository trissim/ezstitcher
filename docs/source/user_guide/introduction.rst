============
Introduction
============

What is EZStitcher?
------------------

EZStitcher is a Python library designed to simplify the processing and stitching of microscopy images. It provides a flexible pipeline architecture that allows researchers to easily process large microscopy datasets, create composite images, flatten Z-stacks, and stitch tiled images together.

**Key Features:**

* **Simplified Interface for Non-Coders**: Process and stitch images with minimal code using the EZ module
* **Pipeline Architecture**: Organize processing steps in a logical sequence
* **Automatic Directory Management**: Protect original data while maintaining organized outputs
* **Flexible Function Handling**: Apply various processing functions in different patterns
* **Support for Various Microscope Formats**: Work with data from different microscope types
* **Multithreaded Processing**: Process multiple wells in parallel for faster results

**Use Cases:**

* Processing large microscopy datasets with consistent operations
* Stitching tiled microscopy images into complete well or plate views
* Creating composite images from multiple fluorescence channels
* Flattening Z-stacks into 2D projections
* Applying custom image processing algorithms to microscopy data

EZStitcher is designed for researchers working with microscopy data who need a flexible, code-based approach to image processing and stitching.

Supported Microscope Types
------------------------

EZStitcher currently supports multiple microscope types, including ImageXpress and Opera Phenix.

For detailed information about supported microscope types, including file formats, naming conventions, and directory structures, see :ref:`microscope-formats` and :ref:`microscope-comparison`.

Support for additional microscope types can be added by implementing the appropriate interfaces. See the :doc:`../development/extending` guide for details.

Core Architecture Overview
------------------------

EZStitcher uses a pipeline architecture that organizes processing into a logical sequence of steps. For detailed information about the architecture, see :doc:`../concepts/architecture_overview`.

The architecture consists of three main components:

* **PipelineOrchestrator**: Coordinates the execution of pipelines across wells
* **Pipeline**: A sequence of processing steps that are executed in order
* **Step**: A single processing operation that can be applied to images

.. figure:: ../_static/architecture_overview.png
   :alt: EZStitcher Architecture Overview
   :width: 80%
   :align: center

   EZStitcher's pipeline architecture showing the relationship between Orchestrator, Pipeline, and Steps.

For comprehensive information about EZStitcher's architecture, including:

* Detailed component descriptions
* Processing workflow and modularity
* Component interactions
* Typical processing flow

See :doc:`../concepts/architecture_overview`.

For details about specific components, see:

* :doc:`../concepts/pipeline_orchestrator` - Details about the Orchestrator
* :doc:`../concepts/pipeline` - Details about Pipelines
* :doc:`../concepts/step` - Details about Steps

Installation
--------------------

EZStitcher requires Python 3.11 or higher. For installation instructions, see the :doc:`../getting_started/installation` guide.

Getting Started
---------------------

.. _three-tier-approach:

EZStitcher offers three main approaches for creating stitching pipelines, each designed for a different level of user experience and need for control:

1. **EZ Module (Beginner Level)**: A simplified, one-liner interface for beginners and non-coders
2. **Custom Pipelines with Steps (Intermediate Level)**: More flexibility and control using pre-defined steps
3. **Library Extension with Base Step (Advanced Level)**: For advanced users who need to understand implementation details

For detailed information about each approach and when to use them, see :doc:`basic_usage`.

When to Use Which Approach
-------------------------

For guidance on when to use each approach, see :doc:`basic_usage`.

For a quick introduction with a minimal working example, see the :doc:`../getting_started/quick_start` guide.

Key Concepts
-----------

EZStitcher is built around several key concepts that work together to provide a flexible and powerful image processing framework. For detailed information about each concept, see the corresponding documentation:

* :doc:`../concepts/architecture_overview` - Overview of EZStitcher's architecture
* :doc:`../concepts/pipeline_orchestrator` - Details about the Orchestrator
* :doc:`../concepts/pipeline` - Details about Pipelines
* :doc:`../concepts/step` - Details about Steps
* :doc:`../concepts/function_handling` - Details about function handling
* :doc:`../concepts/directory_structure` - Details about directory structure
* :doc:`best_practices` - Best practices for using EZStitcher

How to Use This Guide
-------------------

This user guide is organized by complexity level to provide a clear learning path:

* **Basic Usage (Beginner Level)**: Get started quickly with the EZ module
* **Intermediate Usage (Custom Pipelines)**: Learn how to create custom pipelines with steps
* **Advanced Usage (Implementation Details)**: Master advanced features and understand how steps are implemented
* **Best Practices**: Learn recommended practices for all levels
* **Integration**: Integrate EZStitcher with other tools (advanced level)

Each section is clearly marked with its complexity level to help you navigate the documentation based on your experience. Start with the sections that match your current level and progress through the guide as you become more familiar with EZStitcher.

For a comprehensive understanding of EZStitcher's architecture and concepts, please refer to the :doc:`../concepts/index` section.


