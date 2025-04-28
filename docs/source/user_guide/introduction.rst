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

EZStitcher offers three main approaches for creating stitching pipelines:

1. Using the ``EZ module`` for a simplified, one-liner interface (recommended for beginners)
2. Creating custom pipelines for maximum flexibility and control (for advanced users)
3. Extending the library for organization-wide standardization (for contributors)

This three-tier approach allows users to choose the right level of abstraction for their needs:

* **EZ Module**: For beginners and non-coders who want minimal code and default settings
* **Custom Pipelines**: For advanced users who need more control and flexibility
* **Library Extension**: For contributors who want to extend the core library

Most users should start with the EZ module and move to custom pipelines as their needs become more specialized.

When to Use Which Approach
-------------------------

| Use **EZ Module** when… | Use **Custom Pipelines** when… | Use **Library Extension** when… |
|------------------------|--------------------------------|--------------------------------|
| • You want minimal code | • You need bespoke processing  | • You're contributing to EZStitcher |
| • You're new to EZStitcher | • You want per‑channel logic | • You need organization-wide standards |
| • Default settings are sufficient | • You need maximum flexibility | • You're extending core functionality |
| • You want auto-detection | • You want full transparency | • You're adding new microscope types |

For a quick introduction with a minimal working example, see the :doc:`../getting_started/quick_start` guide.

For detailed examples of all approaches, including common use cases and customization options, see the :doc:`ez_module` and :doc:`basic_usage` guides.

Key Concepts
-----------

EZStitcher is built around several key concepts that work together to provide a flexible and powerful image processing framework:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Concept
     - Documentation
   * - **Architecture Overview**
     - :doc:`../concepts/architecture_overview`
   * - **Pipeline Orchestrator**
     - :doc:`../concepts/pipeline_orchestrator`
   * - **Pipeline**
     - :doc:`../concepts/pipeline`
   * - **Steps**
     - :doc:`../concepts/step`
   * - **Specialized Steps**
     - :doc:`../concepts/specialized_steps`
   * - **Function Handling**
     - :doc:`../concepts/function_handling`
   * - **Directory Structure**
     - :doc:`../concepts/directory_structure`
   * - **Best Practices**
     - :doc:`best_practices`

Understanding these concepts will help you create effective image processing workflows tailored to your specific needs.

How to Use This Guide
-------------------

This user guide is organized into several sections:

* **EZ Module**: Learn about the simplified interface for non-coders (recommended for most users)
* **Basic Usage**: Explore custom pipelines for more flexibility
* **Intermediate Usage**: Discover more complex workflows and customization options
* **Advanced Usage**: Master advanced features and techniques
* **Integration**: Integrate EZStitcher with other tools

For a comprehensive understanding of EZStitcher's architecture and concepts, please refer to the :doc:`../concepts/index` section.

.. _learning-path:

Learning Path
---------

EZStitcher provides a flexible framework for processing and stitching microscopy images. Here's a recommended learning path based on your experience level:

**Getting Started:**

* Start with the :doc:`ez_module` guide for the simplest approach
* Try the examples above to get hands-on experience
* Review the :doc:`../concepts/pipeline` to understand the pipeline architecture

**Intermediate Usage:**

* Learn more complex workflows in :doc:`intermediate_usage`
* Study :doc:`../concepts/specialized_steps` to understand specialized steps
* Review best practices in :doc:`best_practices`

**Advanced Usage:**

* Explore advanced features in :doc:`advanced_usage`
* Learn about custom functions and multithreading
* Study :doc:`../concepts/function_handling` to understand function handling patterns

**Advanced Topics:**

* Create custom processing functions as shown in :doc:`advanced_usage`
* Optimize performance with multithreaded processing in :doc:`advanced_usage`
* Extend EZStitcher to support new microscope types using :doc:`../development/extending`
* Integrate with other tools as described in :doc:`integration`

**Mastering EZStitcher:**

* Study :doc:`../concepts/step` to understand step parameters in detail
* Explore :doc:`../concepts/function_handling` to learn about advanced function patterns
* Learn about :doc:`../concepts/directory_structure` to understand how directories are managed
* Dive into the API reference for detailed information about all classes and methods

**Getting Help:**

* Consult the documentation for detailed information
* Check the GitHub repository for issues and updates
* Join the community for support and discussions
