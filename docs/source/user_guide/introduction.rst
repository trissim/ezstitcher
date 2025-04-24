============
Introduction
============

What is EZStitcher?
------------------

EZStitcher is a Python library for processing and stitching high-throughput microscopy images. It provides a flexible, pipeline-based architecture that allows researchers to create custom image processing workflows for various microscopy data formats.

Unlike monolithic applications, EZStitcher embraces a modular approach where processing pipelines are built programmatically as Python scripts, giving you complete control over the processing workflow while maintaining simplicity and reusability.

Key Features
-----------

* **Pipeline Architecture**: Build custom processing workflows from reusable components
* **Microscope Support**: Process images from ImageXpress, Opera Phenix, and other microscopes
* **Z-Stack Processing**: Handle 3D image stacks with various projection methods
* **Focus Detection**: Automatically find the best-focused plane in Z-stacks
* **Channel-Specific Processing**: Apply different processing to different fluorescence channels
* **Multithreaded Processing**: Process multiple wells in parallel for performance
* **Extensible Design**: Add custom processing functions and microscope support

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

Installation
-----------

Requirements
^^^^^^^^^^^

* Python 3.11 or higher
* NumPy, SciPy, scikit-image, pandas, tqdm
* OpenCV (for image processing)

Installation with pip
^^^^^^^^^^^^^^^^^^^^

The recommended way to install EZStitcher is using pip with a virtual environment:

.. code-block:: bash

    # Create a virtual environment with pyenv (recommended)
    pyenv install 3.11.0
    pyenv virtualenv 3.11.0 ezstitcher-env
    pyenv activate ezstitcher-env

    # Install EZStitcher
    pip install ezstitcher

Installation from source
^^^^^^^^^^^^^^^^^^^^^^

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/trissim/ezstitcher.git
    cd ezstitcher
    pip install -e .

Verifying Installation
^^^^^^^^^^^^^^^^^^^^

To verify that EZStitcher is installed correctly:

.. code-block:: python

    import ezstitcher
    print(ezstitcher.__version__)

Quick Start
----------

Here's a minimal example to get started with EZStitcher:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP

    # Create configuration
    config = PipelineConfig(
        num_workers=1  # Single-threaded for simplicity
    )

    # Create orchestrator with plate path
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="path/to/plate"
    )

    # Set up directory structure
    dirs = orchestrator.setup_directories()

    # Create a simple pipeline
    pipeline = Pipeline(
        steps=[
            # Step 1: Process images
            Step(name="Image Processing",
                 func=IP.stack_percentile_normalize,
                 variable_components=['channel']),

            # Step 2: Generate positions
            PositionGenerationStep(
                name="Generate Positions",
                output_dir=dirs['positions']),

            # Step 3: Stitch images
            ImageStitchingStep(
                name="Stitch Images",
                output_dir=dirs['stitched'])
        ],
        name="Basic Pipeline"
    )

    # Run the pipeline
    success = orchestrator.run(pipelines=[pipeline])

For more detailed examples, see the :doc:`basic_usage` and :doc:`../examples/pipeline_examples` sections.

How to Use This Guide
-------------------

This user guide is organized into several sections:

* **Core Concepts**: Explains the fundamental concepts of EZStitcher's pipeline architecture
* **Basic Usage**: Provides simple examples to get started with EZStitcher
* **Intermediate Usage**: Covers more advanced topics like Z-stack processing and stitching
* **Advanced Usage**: Explores custom functions, multithreading, and extensions
* **Practical Examples**: Shows complete workflows for common use cases

If you're new to EZStitcher, we recommend starting with the :doc:`core_concepts` section to understand the pipeline architecture, then moving on to the :doc:`basic_usage` section for practical examples.

For API reference documentation, see the :doc:`../api/index` section.
