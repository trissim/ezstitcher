============
Introduction
============

What is EZStitcher?
------------------

EZStitcher is a Python library designed to simplify the processing and stitching of microscopy images. It provides a flexible pipeline architecture that allows researchers to easily process large microscopy datasets, create composite images, flatten Z-stacks, and stitch tiled images together.

**Key Features:**

* **Pipeline Factory**: Create complete stitching workflows with minimal code
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

Getting Started
---------------------

EZStitcher offers two main approaches for creating stitching pipelines:

1. Using ``AutoPipelineFactory`` for convenient, pre-configured pipelines
2. Building custom pipelines for maximum flexibility and control

Both approaches are valid and powerful, with different strengths depending on your needs.

Using AutoPipelineFactory:
^^^^^^^^^^^^^^^^^^^^^^^

The ``AutoPipelineFactory`` creates pre-configured pipelines for common workflows:

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    from pathlib import Path

    # Path to your plate folder
    plate_path = Path("/path/to/your/plate")

    # Create orchestrator
    orchestrator = PipelineOrchestrator(plate_path=plate_path)

    # Create a factory with default settings
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        output_dir=plate_path.parent / f"{plate_path.name}_stitched",
        normalize=True  # Apply normalization (default)
    )

    # Create the pipelines
    pipelines = factory.create_pipelines()

    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

Building Custom Pipelines:
^^^^^^^^^^^^^^^^^^^^^

For maximum flexibility, you can build custom pipelines by directly specifying each step:

.. code-block:: python

    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.step_factories import ZFlatStep, CompositeStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create position generation pipeline
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Create composite for position generation
            CompositeStep(),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    positions_dir = position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Normalize image intensities
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(positions_dir=positions_dir)
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

The ``AutoPipelineFactory`` automatically creates two pipelines:

1. A pipeline for generating position files
2. A pipeline for stitching images using those position files

This approach handles all common stitching scenarios, including:

* Single-channel and multi-channel data
* Single Z-plane and Z-stack data
* Various projection methods for Z-stacks (max, mean, focus detection, etc.)
* Normalization and preprocessing
* Automatic directory management

For detailed examples, see :doc:`basic_usage`.

Custom pipelines offer maximum flexibility and control:

* Complete control over each step in the pipeline
* Ability to create highly customized workflows
* Terse and elegant code for specific use cases
* Direct access to all pipeline features
* Flexibility to combine any processing functions

Both approaches are valid and powerful, with different strengths depending on your needs. The choice between them depends on your specific requirements and preferences. For detailed examples of custom pipelines, see :doc:`intermediate_usage` and :doc:`advanced_usage`.

**Expected Output:**

* Processed images will be saved in the workspace directory with the suffix `_out` (e.g., `plate_workspace_out`)
* Position files will be saved in the workspace directory with the suffix `_positions` (e.g., `plate_workspace_positions`)
* Stitched images will be saved in the output directory specified (e.g., `plate_stitched`)

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
   * - **Pipeline Factory**
     - :doc:`../concepts/pipeline_factory`
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

* **Basic Usage**: Shows how to use AutoPipelineFactory and create simple custom pipelines
* **Intermediate Usage**: Demonstrates more complex pipelines using both approaches
* **Advanced Usage**: Explores custom functions, multithreading, and other advanced topics
* **Integration**: Shows how to integrate EZStitcher with other tools

For a comprehensive understanding of EZStitcher's architecture and concepts, please refer to the :doc:`../concepts/index` section.

.. _learning-path:

Learning Path
---------

EZStitcher provides a flexible framework for processing and stitching microscopy images. Here's a recommended learning path based on your experience level:

**Getting Started:**

* Start with the :doc:`basic_usage` guide to learn both approaches
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