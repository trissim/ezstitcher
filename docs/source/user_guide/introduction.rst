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
2. **Custom Pipelines with Wrapped Steps (Intermediate Level)**: More flexibility and control using wrapped steps (NormStep, ZFlatStep, etc.)
3. **Library Extension with Base Step (Advanced Level)**: For advanced users who need to understand implementation details

This three-tier approach allows users to choose the right level of abstraction for their needs:

* **EZ Module**: "I just want to stitch my images quickly"
  - For beginners and non-coders who want minimal code
  - Uses sensible defaults and auto-detection
  - Handles common use cases with a single function call
  - Example: ``stitch_plate("path/to/plate")``

* **Custom Pipelines with Wrapped Steps**: "I need more control over the processing steps"
  - For intermediate users who need more flexibility
  - Uses wrapped steps (NormStep, ZFlatStep, etc.) that provide a clean interface for common operations
  - Allows customization of processing steps and parameters
  - Example: Creating pipelines with ``ZFlatStep()``, ``NormStep()``, etc.

* **Library Extension with Base Step**: "I need to understand how the steps work under the hood"
  - For advanced users who need to understand implementation details
  - Uses the base Step class to create custom processing functions
  - Provides maximum flexibility and control
  - Example: ``Step(func=custom_function, variable_components=['z_index'])``

Most users should start with the EZ module and move to custom pipelines with wrapped steps as their needs become more specialized. Only advanced users who need to understand implementation details should use the base Step class directly.

When to Use Which Approach
-------------------------

| Use **EZ Module** when… | Use **Custom Pipelines with Wrapped Steps** when… | Use **Library Extension with Base Step** when… |
|------------------------|--------------------------------|--------------------------------|
| • You want minimal code | • You need bespoke processing  | • You need to understand implementation details |
| • You're new to EZStitcher | • You want per‑channel logic | • You're creating custom processing functions |
| • Default settings are sufficient | • You need more flexibility | • You're extending core functionality |
| • You want auto-detection | • You want to customize processing steps | • You're implementing new microscope handlers |
| • You want a one-liner solution | • You need multiple output types | • You're contributing to EZStitcher |

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
   * - **Steps and Step Types**
     - :doc:`../concepts/step`
   * - **Function Handling**
     - :doc:`../concepts/function_handling`
   * - **Directory Structure**
     - :doc:`../concepts/directory_structure`
   * - **Best Practices**
     - :doc:`best_practices`

Understanding these concepts will help you create effective image processing workflows tailored to your specific needs.

How to Use This Guide
-------------------

This user guide is organized by complexity level to provide a clear learning path:

* **EZ Module (Beginner Level)**: Learn about the simplified interface for non-coders (recommended for most users)
* **Transitioning from EZ Module**: Bridge the gap between the EZ module and custom pipelines
* **Intermediate Usage (Custom Pipelines with Wrapped Steps)**: Create custom pipelines with wrapped steps (NormStep, ZFlatStep, etc.)
* **Advanced Usage (Library Extension with Base Step)**: Master advanced features and understand how wrapped steps are implemented
* **Best Practices**: Learn recommended practices for all levels
* **Integration**: Integrate EZStitcher with other tools (advanced level)

Each section is clearly marked with its complexity level to help you navigate the documentation based on your experience. Start with the sections that match your current level and progress through the guide as you become more familiar with EZStitcher.

For a comprehensive understanding of EZStitcher's architecture and concepts, please refer to the :doc:`../concepts/index` section.

.. _learning-path:

Learning Path
---------

EZStitcher provides a flexible framework for processing and stitching microscopy images. Here's a recommended learning path based on your experience level:

**Beginner Level (EZ Module):**

* Start with the :doc:`ez_module` guide for the simplest approach
* Try the examples to get hands-on experience
* Use the EZ module for quick results with minimal code

**Transitioning to Intermediate Level:**

* Read the :doc:`transitioning_from_ez` guide to understand how to bridge the gap between the EZ module and custom pipelines
* Learn about the pipeline architecture in :doc:`../concepts/pipeline`
* Understand the basic concepts in :doc:`basic_usage`

**Intermediate Level (Custom Pipelines with Wrapped Steps):**

* Learn how to create custom pipelines with wrapped steps in :doc:`intermediate_usage`
* Understand how to use wrapped steps (NormStep, ZFlatStep, etc.) for common operations
* Review best practices in :doc:`best_practices`

**Advanced Level (Library Extension with Base Step):**

* Explore advanced features in :doc:`advanced_usage`
* Learn how wrapped steps are implemented using the base Step class
* Study :doc:`../concepts/step` to understand step parameters in detail
* Explore :doc:`../concepts/function_handling` to learn about advanced function patterns

**Expert Level:**

* Create custom processing functions as shown in :doc:`advanced_usage`
* Optimize performance with multithreaded processing
* Extend EZStitcher to support new microscope types using :doc:`../development/extending`
* Integrate with other tools as described in :doc:`integration`

**Getting Help:**

* Consult the documentation for detailed information
* Check the GitHub repository for issues and updates
* Join the community for support and discussions
