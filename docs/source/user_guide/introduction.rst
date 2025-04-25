============
Introduction
============

What is EZStitcher?
------------------

EZStitcher is a Python library designed to simplify the processing and stitching of microscopy images. It provides a flexible pipeline architecture that allows researchers to easily process large microscopy datasets, create composite images, flatten Z-stacks, and stitch tiled images together.

**Key Features:**

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

EZStitcher uses a pipeline architecture that organizes processing into a logical sequence of steps. The architecture consists of three main components:

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

Installation and Setup
--------------------

EZStitcher requires Python 3.11 or higher. The simplest way to install EZStitcher is directly from the Git repository using pyenv and pip.

### Quick Installation

1. **Set up a Python environment with pyenv**:

```bash
# Install Python 3.11 with pyenv
pyenv install 3.11.0

# Create a virtual environment
pyenv virtualenv 3.11.0 ezstitcher-env

# Activate the environment
pyenv local ezstitcher-env
```

2. **Install EZStitcher from the Git repository**:

```bash
# Clone the repository
git clone https://github.com/your-org/ezstitcher.git
cd ezstitcher

# Install the package and dependencies
pip install -e .
```

All dependencies will be automatically installed from the requirements.txt file included in the repository.

Quick Start Example
-----------------

Here's a simple example that demonstrates how to create and run a pipeline for processing and stitching microscopy images:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create configuration with 2 worker threads
    config = PipelineConfig(num_workers=2)

    # Create orchestrator with path to microscopy data
    orchestrator = PipelineOrchestrator(config=config, plate_path="path/to/plate")

    # Create pipeline with processing, position generation, and stitching steps
    pipeline = Pipeline(
        steps=[
            # Step 1: Basic processing - normalize image intensities
            Step(
                name="Basic Processing",
                func=IP.stack_percentile_normalize,
                input_dir=orchestrator.workspace_path
            ),
            PositionGenerationStep(),

            # By default, uses previous step's output directory (position files)
            ImageStitchingStep(
                # input_dir=orchestrator.workspace_path  # Uncomment to use original images for stitching
            )
        ],
        name="Simple Pipeline"
    )

    # Run the pipeline
    orchestrator.run(pipelines=[pipeline])

**Step-by-Step Explanation:**

1. We create a configuration with 2 worker threads for parallel processing
2. We create an orchestrator that points to our microscopy data
3. We define a pipeline with three steps:
   - A basic processing step that normalizes image intensities
   - A position generation step that calculates tile positions
   - An image stitching step that combines the processed tiles
4. We run the pipeline using the orchestrator

**Expected Output:**

* Processed images will be saved in the workspace directory with the suffix `_out` (e.g., `plate_workspace_out`)
* Position files will be saved in the workspace directory with the suffix `_positions` (e.g., `plate_workspace_positions`)
* Stitched images will be saved in the workspace directory with the suffix `_stitched` (e.g., `plate_workspace_stitched`)

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

* **Intermediate Usage**: Provides detailed examples of common EZStitcher workflows
* **Advanced Usage**: Explores custom functions, multithreading, and extensions
* **Integration**: Shows how to integrate EZStitcher with other tools

For a comprehensive understanding of EZStitcher's architecture and concepts, please refer to the :doc:`../concepts/index` section.

.. _learning-path:

Learning Path
---------

EZStitcher provides a flexible framework for processing and stitching microscopy images. By understanding its core concepts and architecture, you can create powerful pipelines tailored to your specific needs.

Here's a recommended learning path based on your experience level:

**Getting Started:**

* Read the :doc:`basic_usage` guide to learn the fundamentals
* Try the Quick Start example above to get hands-on experience
* Review the :doc:`../concepts/architecture_overview` to understand the big picture

**Building Basic Pipelines:**

* Learn about Z-stack processing in :doc:`intermediate_usage`
* Explore channel-specific processing in :doc:`intermediate_usage`
* Understand position generation and stitching in :doc:`intermediate_usage`
* Review best practices in :doc:`best_practices`

**Advanced Topics:**

* Create custom processing functions as shown in :doc:`advanced_usage`
* Optimize performance with multithreaded processing in :doc:`advanced_usage`
* Extend EZStitcher to support new microscope types using :doc:`../development/extending`
* Integrate with other tools as described in :doc:`integration`

**Mastering EZStitcher:**

* Dive into :doc:`../concepts/pipeline` to create custom pipelines
* Study :doc:`../concepts/step` to understand step parameters in detail
* Explore :doc:`../concepts/function_handling` to learn about advanced function patterns
* Learn about :doc:`../concepts/directory_structure` to understand how directories are managed

**Getting Help:**

* Consult the documentation for detailed information
* Check the GitHub repository for issues and updates
* Join the community for support and discussions