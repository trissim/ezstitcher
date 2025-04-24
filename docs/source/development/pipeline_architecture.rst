Pipeline Architecture
====================

Overview
--------

EZStitcher's pipeline architecture has been redesigned to provide a more flexible, modular, and extensible framework for processing microscopy images. The new architecture is composed of three main components:

1. **PipelineOrchestrator**: Coordinates the execution of multiple pipelines across wells
2. **Pipeline**: A sequence of processing steps
3. **Step**: A single processing operation (with specialized subclasses)

This hierarchical design allows for complex workflows to be built from simple, reusable components.

Architecture Diagram
-------------------

.. code-block:: text

    ┌─────────────────────────────────────────┐
    │            PipelineOrchestrator         │
    │                                         │
    │  ┌─────────┐    ┌─────────┐             │
    │  │ Pipeline│    │ Pipeline│    ...      │
    │  │         │    │         │             │
    │  │ ┌─────┐ │    │ ┌─────┐ │             │
    │  │ │Step │ │    │ │Step │ │             │
    │  │ └─────┘ │    │ └─────┘ │             │
    │  │ ┌─────┐ │    │ ┌─────┐ │             │
    │  │ │Step │ │    │ │Step │ │             │
    │  │ └─────┘ │    │ └─────┘ │             │
    │  │   ...   │    │   ...   │             │
    │  └─────────┘    └─────────┘             │
    └─────────────────────────────────────────┘

Key Components
-------------

PipelineOrchestrator
^^^^^^^^^^^^^^^^^^^

The ``PipelineOrchestrator`` is the central coordinator that manages the execution of multiple pipelines. It handles:

- Plate and well detection
- Directory structure management
- Multithreaded execution of pipelines
- Error handling and logging

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1"],
        num_workers=2  # Use 2 worker threads
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(config=config, plate_path="path/to/plate")

    # Run the orchestrator with pipelines
    success = orchestrator.run(pipelines=[pipeline1, pipeline2])

Pipeline
^^^^^^^

A ``Pipeline`` is a sequence of processing steps that are executed in order. It provides:

- Step management (adding, removing, reordering)
- Context passing between steps
- Input/output directory management

.. code-block:: python

    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step

    # Create a pipeline with steps
    pipeline = Pipeline(
        steps=[step1, step2, step3],
        name="My Processing Pipeline"
    )

    # Add a step to the pipeline
    pipeline.add_step(step4)

    # Run the pipeline with a context
    result_context = pipeline.run(context)

Step
^^^^

A ``Step`` is a single processing operation that can be applied to images. The base ``Step`` class provides:

- Image loading and saving
- Processing function application
- Variable component handling (e.g., channels, z-indices)
- Group-by functionality for processing related images together

.. code-block:: python

    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP

    # Create a basic processing step
    step = Step(
        name="Image Enhancement",
        func=IP.stack_percentile_normalize,
        variable_components=['channel'],
        group_by='channel',
        input_dir="path/to/input",
        output_dir="path/to/output"
    )

Specialized step classes include:

- **PositionGenerationStep**: Generates position files for stitching
- **ImageStitchingStep**: Stitches images using position files

Function Handling
---------------

The pipeline architecture supports three patterns for processing functions:

1. **Single Function**: A callable that takes a list of images and returns a list of processed images

   .. code-block:: python

       # Single function
       step = Step(func=IP.stack_percentile_normalize)

2. **List of Functions**: A sequence of functions applied one after another to the images

   .. code-block:: python

       # List of functions
       step = Step(func=[stack(IP.sharpen), IP.stack_percentile_normalize])

3. **Dictionary of Functions**: A mapping from component values (like channel numbers) to functions or lists of functions

   .. code-block:: python

       # Dictionary of functions
       step = Step(
           func={"1": process_dapi, "2": process_calcein},
           variable_components=['channel'],
           group_by='channel'
       )

This flexibility allows for complex processing workflows to be built from simple, reusable components.

Relationship Between Components
-----------------------------

The relationship between the components is hierarchical:

1. The ``PipelineOrchestrator`` manages multiple ``Pipeline`` instances
2. Each ``Pipeline`` contains multiple ``Step`` instances
3. Each ``Step`` applies processing functions to images

The ``PipelineOrchestrator`` handles the high-level coordination, such as well detection and multithreading, while the ``Pipeline`` and ``Step`` classes handle the actual image processing.

Example Workflow
--------------

A typical workflow using the pipeline architecture might look like this:

1. Create a ``PipelineConfig`` with desired settings
2. Create a ``PipelineOrchestrator`` with the config
3. Create one or more ``Pipeline`` instances with appropriate ``Step`` instances
4. Run the orchestrator with the pipelines

For detailed examples, see the :doc:`../examples/pipeline_examples` documentation and the integration tests in the ``tests/integration`` directory.
