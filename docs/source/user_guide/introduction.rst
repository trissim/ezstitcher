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

EZStitcher currently supports the following microscope types:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Microscope
     - File Format
     - Naming Convention
   * - ImageXpress
     - TIFF + HTD/XML metadata
     - ``[Well]_s[Site]_w[Channel].tif``
   * - Opera Phenix
     - TIFF + Index.xml metadata
     - ``r[Row]c[Col]f[Field]p[Plane]-ch[Channel]sk[Skip]fk[Fk]fl[Fl].tiff``

Support for additional microscope types can be added by implementing the appropriate interfaces. See the :doc:`../development/extending` guide for details.

Core Architecture Overview
------------------------

EZStitcher uses a pipeline architecture that organizes processing into a logical sequence of steps.

.. figure:: ../_static/architecture_overview.png
   :alt: EZStitcher Architecture Overview
   :width: 80%
   :align: center

   EZStitcher's pipeline architecture showing the relationship between Orchestrator, Pipeline, and Steps.

The architecture consists of three main components:

* **Pipeline**: A sequence of processing steps that can be applied to microscopy images
* **Orchestrator**: Manages plate-level operations and provides services to steps
* **Step**: A single processing operation that can be applied to images

For detailed information about EZStitcher's architecture, see:

* :doc:`../concepts/architecture_overview` - Overview of the architecture
* :doc:`../concepts/pipeline_orchestrator` - Details about the Orchestrator
* :doc:`../concepts/pipeline` - Details about Pipelines
* :doc:`../concepts/step` - Details about Steps

Installation and Setup
--------------------

Requirements
^^^^^^^^^^^^^^^^^

* Python 3.11 or higher
* NumPy, SciPy, scikit-image, pandas, tqdm
* OpenCV (for image processing)

Installation with pip
^^^^^^^^^^^^^^^^^

The recommended way to install EZStitcher is using pip with a virtual environment:

.. code-block:: bash

    # Create a virtual environment with pyenv (recommended)
    pyenv install 3.11.0
    pyenv virtualenv 3.11.0 ezstitcher-env
    pyenv activate ezstitcher-env

    # Install EZStitcher
    pip install ezstitcher

Installation from source
^^^^^^^^^^^^^^^^^

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/trissim/ezstitcher.git
    cd ezstitcher
    pip install -e .

Verifying Installation
^^^^^^^^^^^^^^^^^^^

To verify that EZStitcher is installed correctly:

.. code-block:: python

    import ezstitcher
    print(ezstitcher.__version__)

Quick Start Example
-----------------

Here's a simple example that demonstrates how to create and run a pipeline for processing and stitching microscopy images:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP

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

**Common Issues:**

* **Directory Permissions**: Ensure you have write permissions for the output directories
* **Missing Dependencies**: Make sure all required dependencies are installed
* **Image Format Issues**: Verify that your images are in a supported format

Key Concepts Preview
------------------

EZStitcher includes several key concepts that are important to understand for effective use:

* **Pipeline Architecture**: The overall structure of EZStitcher's processing framework
* **Function Handling Patterns**: Different ways to specify processing functions
* **Directory Structure and Resolution**: How EZStitcher manages directories
* **Step Parameters**: How to configure processing steps
* **Specialized Steps**: Pre-built steps for common tasks

For detailed explanations of these concepts, see the following documentation:

* :doc:`../concepts/architecture_overview` - Overview of the architecture
* :doc:`../concepts/function_handling` - Function handling patterns
* :doc:`../concepts/directory_structure` - Directory structure and resolution
* :doc:`../concepts/step` - Step parameters and configuration
* :doc:`../concepts/specialized_steps` - Specialized steps for common tasks

How to Use This Guide
-------------------

This user guide is organized into several sections:

* **Intermediate Usage**: Provides detailed examples of common EZStitcher workflows
* **Advanced Usage**: Explores custom functions, multithreading, and extensions
* **Integration**: Shows how to integrate EZStitcher with other tools

For a comprehensive understanding of EZStitcher's architecture and concepts, please refer to the :doc:`../concepts/index` section.

Next Steps
---------

Now that you have a basic understanding of EZStitcher, here are some recommendations for next steps:

**For All Users:**

* Read the :doc:`intermediate_usage` guide for detailed examples
* Explore the concepts documentation to understand the core architecture

**For Intermediate Users:**

* Explore :doc:`../concepts/function_handling` to learn about advanced function patterns
* Learn about :doc:`../concepts/directory_structure` to understand how directories are managed

**For Advanced Users:**

* Dive into :doc:`../concepts/pipeline` to create custom pipelines
* Study :doc:`../concepts/step` to understand step parameters in detail

**Getting Help:**

* Consult the documentation for detailed information
* Check the GitHub repository for issues and updates
* Join the community for support and discussions

EZStitcher provides a flexible framework for processing and stitching microscopy images. By understanding its core concepts and architecture, you can create powerful pipelines tailored to your specific needs.
