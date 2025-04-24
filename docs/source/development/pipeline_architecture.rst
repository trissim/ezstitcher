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
    from ezstitcher.core.file_system_manager import FileSystemManager
    from ezstitcher.core.image_preprocessor import ImagePreprocessor
    from ezstitcher.core.focus_analyzer import FocusAnalyzer

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1"],  # Use channel 1 as reference
        num_workers=2              # Use 2 worker threads
    )

    # Create orchestrator with full parameter set
    # All parameters except config are optional and will be created if not provided
    orchestrator = PipelineOrchestrator(
        plate_path="path/to/plate",                # Path to the plate folder
        workspace_path="path/to/workspace",        # Path to the workspace folder (optional)
        config=config,                             # Pipeline configuration
        fs_manager=FileSystemManager(),            # File system manager (optional)
        image_preprocessor=ImagePreprocessor(),    # Image preprocessor (optional)
        focus_analyzer=FocusAnalyzer()             # Focus analyzer (optional)
    )

    # Run the orchestrator with pipelines
    success = orchestrator.run(pipelines=[pipeline1, pipeline2])

Pipeline
^^^^^^^

A ``Pipeline`` is a sequence of processing steps that are executed in order. It provides:

- Step management (adding, removing, reordering)
- Context passing between steps
- Input/output directory management

.. code-block:: python

    from ezstitcher.core.pipeline import Pipeline, ProcessingContext
    from ezstitcher.core.steps import Step

    # Create a pipeline with steps
    pipeline = Pipeline(
        steps=[step1, step2, step3],
        name="My Processing Pipeline"
    )

    # Add a step to the pipeline
    pipeline.add_step(step4)

    # Method 1: Run the pipeline with individual parameters
    # The pipeline will create a ProcessingContext internally
    results = pipeline.run(
        input_dir="path/to/input",
        output_dir="path/to/output",
        well_filter=["A01", "B02"],
        orchestrator=orchestrator,  # Required - provides access to microscope_handler
        positions_file="path/to/positions.csv"  # Optional
    )

    # Method 2: Run the pipeline with a pre-configured context
    # This is typically used when the pipeline is run from the PipelineOrchestrator
    context = ProcessingContext(
        input_dir="path/to/input",
        output_dir="path/to/output",
        well_filter=["A01", "B02"],
        orchestrator=orchestrator,
        positions_file="path/to/positions.csv"
    )
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
       # This applies the same function to all images
       step = Step(
           name="Normalize Images",
           func=IP.stack_percentile_normalize,  # Single function to apply
           variable_components=['channel'],     # Process each channel separately
           processing_args={                    # Additional arguments for the function
               'low_percentile': 0.1,
               'high_percentile': 99.9
           }
       )

2. **List of Functions**: A sequence of functions applied one after another to the images

   .. code-block:: python

       # List of functions
       # This applies multiple functions in sequence
       step = Step(
           name="Enhance Images",
           func=[
               stack(IP.sharpen),              # First sharpen the images
               IP.stack_percentile_normalize   # Then normalize the intensities
           ],
           variable_components=['channel'],    # Process each channel separately
           group_by='site'                     # Group by site for processing
       )

3. **Dictionary of Functions**: A mapping from component values (like channel numbers) to functions or lists of functions

   .. code-block:: python

       # Dictionary of functions
       # This applies different functions to different channels
       step = Step(
           name="Channel-Specific Processing",
           func={
               "1": process_dapi,      # Apply process_dapi to channel 1
               "2": process_calcein    # Apply process_calcein to channel 2
           },
           variable_components=['channel'],  # Process each channel separately
           group_by='channel'               # Group by channel for channel-specific processing
       )

This flexibility allows for complex processing workflows to be built from simple, reusable components.

Relationship Between Components
-----------------------------

The relationship between the components is hierarchical:

1. The ``PipelineOrchestrator`` manages multiple ``Pipeline`` instances
2. Each ``Pipeline`` contains multiple ``Step`` instances
3. Each ``Step`` applies processing functions to images
4. The ``ProcessingContext`` facilitates communication between steps

The ``PipelineOrchestrator`` handles the high-level coordination, such as well detection and multithreading, while the ``Pipeline`` and ``Step`` classes handle the actual image processing.

ProcessingContext
^^^^^^^^^^^^^^^

The ``ProcessingContext`` is a crucial component that maintains state during pipeline execution. It:

- Holds input/output directories, well filter, and configuration
- Stores processing results
- Serves as a communication mechanism between steps
- Can be extended with additional attributes via kwargs

.. code-block:: python

    from ezstitcher.core.pipeline import ProcessingContext

    # Create a processing context
    context = ProcessingContext(
        input_dir="path/to/input",
        output_dir="path/to/output",
        well_filter=["A01", "B02"],
        orchestrator=orchestrator,  # Reference to the PipelineOrchestrator
        # Additional attributes can be added as kwargs
        positions_file="path/to/positions.csv",
        custom_parameter=42
    )

    # Access attributes
    print(context.input_dir)
    print(context.custom_parameter)  # Custom attributes are accessible directly

    # Steps can add results to the context
    context.results["step1"] = {"processed_files": 10}

When a pipeline runs, it creates a ProcessingContext (or uses one provided) and passes it from step to step. Each step can read from and write to the context, allowing for flexible data flow through the pipeline.

Example Workflow
--------------

A typical workflow using the pipeline architecture might look like this:

1. Create a ``PipelineConfig`` with desired settings
2. Create a ``PipelineOrchestrator`` with the config
3. Create one or more ``Pipeline`` instances with appropriate ``Step`` instances
4. Run the orchestrator with the pipelines

For detailed examples, see the :doc:`../examples/pipeline_examples` documentation and the integration tests in the ``tests/integration`` directory.
