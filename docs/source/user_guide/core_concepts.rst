=============
Core Concepts
=============

Pipeline Architecture Overview
----------------------------

EZStitcher is built around a flexible pipeline architecture that allows you to create custom image processing workflows. The architecture consists of three main components:

1. **PipelineOrchestrator**: Coordinates the execution of pipelines across wells
2. **Pipeline**: A sequence of processing steps
3. **Step**: A single processing operation

This hierarchical design allows complex workflows to be built from simple, reusable components:

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

When you run a pipeline, data flows through the steps in sequence. Each step processes the images and passes the results to the next step through a shared context object.

PipelineOrchestrator
------------------

The ``PipelineOrchestrator`` is the top-level component that coordinates the execution of pipelines. It handles:

* Plate and well detection
* Directory structure management
* Multithreaded execution
* Error handling and logging

Creating an Orchestrator
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration
    config = PipelineConfig(
        num_workers=2  # Use 2 worker threads
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="path/to/plate"
    )

Directory Structure
^^^^^^^^^^^^^^^^

The orchestrator creates a standard directory structure for processing:

.. code-block:: python

    # Set up directory structure
    dirs = orchestrator.setup_directories()

    # This creates:
    # - dirs['input']: Original images
    # - dirs['processed']: Processed individual tiles
    # - dirs['post_processed']: Post-processed images
    # - dirs['positions']: CSV files with stitching positions
    # - dirs['stitched']: Final stitched images

Running Pipelines
^^^^^^^^^^^^^^

The orchestrator can run one or more pipelines:

.. code-block:: python

    # Run a single pipeline
    orchestrator.run(pipelines=[pipeline])

    # Run multiple pipelines in sequence
    orchestrator.run(pipelines=[pipeline1, pipeline2, pipeline3])

When multiple pipelines are provided, they are executed in sequence for each well. If ``num_workers`` is greater than 1, multiple wells are processed in parallel.

Pipeline
-------

A ``Pipeline`` is a sequence of processing steps that are executed in order. It provides:

* Step management (adding, removing, reordering)
* Context passing between steps
* Input/output directory management

Creating a Pipeline
^^^^^^^^^^^^^^^

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

Running a Pipeline
^^^^^^^^^^^^^^

A pipeline can be run directly, but it's typically run through the orchestrator:

.. code-block:: python

    # Run through the orchestrator (recommended)
    orchestrator.run(pipelines=[pipeline])

    # Run directly (advanced usage)
    results = pipeline.run(
        input_dir="path/to/input",
        output_dir="path/to/output",
        well_filter=["A01", "B02"],
        orchestrator=orchestrator  # Required for microscope handler access
    )

Pipeline Context
^^^^^^^^^^^^^

When a pipeline runs, it creates a ``ProcessingContext`` that is passed from step to step. This context holds:

* Input/output directories
* Well filter
* Configuration
* Results from previous steps

This allows steps to communicate and build on each other's results.

Step
----

A ``Step`` is a single processing operation that can be applied to images. The base ``Step`` class provides:

* Image loading and saving
* Processing function application
* Variable component handling (e.g., channels, z-indices)
* Group-by functionality for processing related images together

Creating a Basic Step
^^^^^^^^^^^^^^^^^

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

Step Parameters
^^^^^^^^^^^^

* ``name``: Human-readable name for the step
* ``func``: The processing function(s) to apply (see Function Handling below)
* ``variable_components``: Components that vary across files (e.g., 'z_index', 'channel')
* ``group_by``: How to group files for processing (e.g., 'channel', 'site')
* ``input_dir``: The input directory (optional, can inherit from pipeline)
* ``output_dir``: The output directory (optional, can inherit from pipeline)
* ``processing_args``: Additional arguments to pass to the processing function

Variable Components
^^^^^^^^^^^^^^^^

The ``variable_components`` parameter specifies which components vary across files. In most cases, you don't need to set this explicitly as it defaults to 'site', but there are specific cases where you should change it:

.. code-block:: python

    # When flattening Z-stacks, set variable_components to 'z_index'
    step = Step(
        name="Z-Stack Flattening",
        func=IP.create_projection,
        variable_components=['z_index'],  # Process each z-index separately
        processing_args={'method': 'max_projection'}
    )

    # When creating composite images, set variable_components to 'channel'
    step = Step(
        name="Create Composite",
        func=IP.create_composite,
        variable_components=['channel'],  # Process each channel separately
        group_by='site'  # Group by site to combine channels for each site
    )

    # For most other operations, the default 'site' is appropriate
    step = Step(
        name="Enhance Images",
        func=stack(IP.sharpen)
        # variable_components defaults to ['site']
    )

Group By
^^^^^^^

The ``group_by`` parameter specifies what the keys in a dictionary of functions correspond to. It determines how the keys in your function dictionary are mapped to components in the file names:

.. code-block:: python

    # When using a dictionary of channel-specific functions
    step = Step(
        name="Channel-Specific Processing",
        func={"1": process_dapi, "2": process_calcein},
        # variable_components defaults to ['site']
        group_by='channel'  # Keys "1" and "2" correspond to channel values
    )

In this example:
- ``group_by='channel'`` means the keys in the function dictionary ("1" and "2") correspond to channel values
- Files with channel="1" will be processed by ``process_dapi``
- Files with channel="2" will be processed by ``process_calcein``

The ``group_by`` parameter should never be the same as ``variable_components``:

.. code-block:: python

    # When creating composite images
    step = Step(
        name="Create Composite",
        func=IP.create_composite,
        variable_components=['channel'],  # Process each channel separately
        group_by='site'  # Group files by site for processing
    )

When using a dictionary of functions, ``group_by`` is required to tell EZStitcher what component the dictionary keys refer to. This allows for component-specific processing, such as applying different functions to different channels.

Specialized Step Classes
---------------------

EZStitcher includes specialized Step subclasses for common tasks:

PositionGenerationStep
^^^^^^^^^^^^^^^^^^^

The ``PositionGenerationStep`` generates position files for stitching:

.. code-block:: python

    from ezstitcher.core.steps import PositionGenerationStep

    # Create a position generation step
    step = PositionGenerationStep(
        name="Generate Positions",
        output_dir=dirs['positions']
    )

This step:
1. Analyzes the images to find overlapping regions
2. Calculates the relative positions of tiles
3. Saves the positions to CSV files

ImageStitchingStep
^^^^^^^^^^^^^^^

The ``ImageStitchingStep`` stitches images using position files:

.. code-block:: python

    from ezstitcher.core.steps import ImageStitchingStep

    # Create an image stitching step
    step = ImageStitchingStep(
        name="Stitch Images",
        positions_dir=dirs['positions'],
        output_dir=dirs['stitched']
    )

This step:
1. Loads the position files
2. Loads the images according to the positions
3. Stitches the images together
4. Saves the stitched images

When to Use Specialized Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use specialized steps when you need the specific functionality they provide. For general image processing tasks, use the base ``Step`` class.

Function Handling
--------------

The Step class supports three patterns for processing functions:

Single Function
^^^^^^^^^^^^

A callable that takes a list of images and returns a list of processed images:

.. code-block:: python

    # Single function
    step = Step(
        name="Normalize Images",
        func=IP.stack_percentile_normalize,
        variable_components=['channel']
    )

List of Functions
^^^^^^^^^^^^^

A sequence of functions applied one after another:

.. code-block:: python

    from ezstitcher.core.utils import stack

    # List of functions
    step = Step(
        name="Enhance Images",
        func=[
            stack(IP.sharpen),              # First sharpen the images
            IP.stack_percentile_normalize   # Then normalize the intensities
        ],
        variable_components=['channel']
    )

The ``stack()`` utility function applies a single-image function to each image in a stack.

Dictionary of Functions
^^^^^^^^^^^^^^^^^^^

A mapping from component values to functions, allowing different processing for different components:

.. code-block:: python

    # Define channel-specific processing functions
    def process_dapi(stack):
        """Process DAPI channel images."""
        stack = IP.stack_percentile_normalize(stack)
        return [IP.tophat(img) for img in stack]

    def process_calcein(stack):
        """Process Calcein channel images."""
        return [IP.tophat(img) for img in stack]

    # Dictionary of functions
    step = Step(
        name="Channel-Specific Processing",
        func={
            "1": process_dapi,      # Apply process_dapi to channel 1
            "2": process_calcein    # Apply process_calcein to channel 2
        },
        # variable_components defaults to ['site']
        group_by='channel'  # Specifies that keys "1" and "2" refer to channel values
    )

When using a dictionary of functions:
- The `group_by` parameter is required to specify what component the dictionary keys refer to
- Each key in the dictionary corresponds to a specific value of that component
- Files are processed by the function that matches their component value
- For example, with `group_by='channel'`, files with channel="1" are processed by the function at key "1"

Matching Processing Args with Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using a list of functions, you can provide a matching list of processing_args:

.. code-block:: python

    # List of functions with matching processing_args
    step = Step(
        name="Multi-Step Processing",
        func=[
            stack(IP.tophat),             # Background removal
            stack(IP.sharpen),            # Sharpening
            IP.stack_percentile_normalize  # Normalization
        ],
        # variable_components defaults to ['site']
        processing_args=[
            {'size': 15},                 # Arguments for tophat
            {'sigma': 1.0, 'amount': 2.0},  # Arguments for sharpen
            {'low_percentile': 1.0, 'high_percentile': 99.0}  # Arguments for normalize
        ]
    )

Each dictionary in the processing_args list is matched with the corresponding function in the func list.

You can also use this pattern with a dictionary of functions, where each function can be a list with a matching list of processing_args:

.. code-block:: python

    # Dictionary of functions with matching processing_args
    step = Step(
        name="Advanced Channel Processing",
        func={
            "1": [stack(IP.tophat), stack(IP.sharpen)],      # Process channel 1
            "2": [IP.stack_percentile_normalize]             # Process channel 2
        },
        # variable_components defaults to ['site']
        group_by='channel',  # Group by channel to apply different functions
        processing_args={
            "1": [
                {'size': 15},                 # Arguments for tophat
                {'sigma': 1.0, 'amount': 2.0}  # Arguments for sharpen
            ],
            "2": [
                {'low_percentile': 1.0, 'high_percentile': 99.0}  # Arguments for normalize
            ]
        }
    )

When to Use Each Pattern
^^^^^^^^^^^^^^^^^^^^^

* **Single Function**: When you need to apply the same processing to all images
* **List of Functions**: When you need to apply multiple processing steps in sequence
* **Dictionary of Functions**: When you need to apply different processing to different components

ProcessingContext
--------------

The ``ProcessingContext`` is a crucial component that maintains state during pipeline execution. It:

* Holds input/output directories, well filter, and configuration
* Stores processing results
* Serves as a communication mechanism between steps

Creating a Context
^^^^^^^^^^^^^^

The context is typically created by the pipeline, but you can create it manually for advanced usage:

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

Accessing Context Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^

Context attributes can be accessed directly:

.. code-block:: python

    # Access standard attributes
    print(context.input_dir)
    print(context.well_filter)

    # Access custom attributes
    print(context.positions_file)
    print(context.custom_parameter)

Storing Results
^^^^^^^^^^^^

Steps can store results in the context:

.. code-block:: python

    # Store results
    context.results["step1"] = {"processed_files": 10}

    # Access results from another step
    processed_files = context.results.get("step1", {}).get("processed_files", 0)

Communication Between Steps
^^^^^^^^^^^^^^^^^^^^^^^^

The context allows steps to communicate and build on each other's results:

.. code-block:: python

    # Step 1: Generate positions and store in context
    def process(self, context):
        # ... generate positions ...
        context.positions_file = "path/to/positions.csv"
        return context

    # Step 2: Use positions from context
    def process(self, context):
        positions_file = context.positions_file
        # ... use positions_file ...
        return context

Directory Structure
----------------

EZStitcher uses a dynamic directory resolution system that automatically manages directories based on the pipeline structure.

Dynamic Directory Resolution
^^^^^^^^^^^^^^^^^^^^^

Unlike traditional approaches that require explicit directory management, EZStitcher's pipeline architecture automatically resolves directories based on the sequence of steps:

1. Input/output directories are defined only on the first step
2. Each step's output directory becomes the next step's input directory
3. Directories are created automatically when needed during pipeline execution
4. Each well gets its own subfolder during processing

This approach eliminates the need to manually specify directories for each step and reduces the chance of errors.

.. code-block:: python

    # The orchestrator provides the initial input directory
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="path/to/plate"  # This becomes the initial input directory
    )

    # The setup_directories method helps with directory resolution
    # but doesn't create a fixed set of directories
    dirs = orchestrator.setup_directories()

    # You only need to specify directories for the first step
    # Subsequent steps will automatically use appropriate directories
    pipeline = Pipeline(
        steps=[
            # First step specifies input_dir explicitly
            Step(name="Step 1",
                 func=IP.stack_percentile_normalize,
                 input_dir=dirs['input'],
                 output_dir="path/to/output"),

            # Second step automatically uses the output of the first step
            Step(name="Step 2",
                 func=stack(IP.sharpen))
                 # input_dir is automatically set to "path/to/output"
                 # output_dir is automatically determined
        ],
        name="My Pipeline"
    )

Specialized Step Directory Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specialized steps like ``PositionGenerationStep`` and ``ImageStitchingStep`` have intelligent directory handling:

.. code-block:: python

    # Create a pipeline with specialized steps
    pipeline = Pipeline(
        steps=[
            # Regular processing step
            Step(name="Process Images",
                 func=IP.stack_percentile_normalize,
                 input_dir=dirs['input'],
                 output_dir="path/to/processed"),

            # PositionGenerationStep automatically creates a positions directory
            PositionGenerationStep(
                name="Generate Positions"),
                # input_dir is automatically set to "path/to/processed"
                # output_dir is automatically set to a "positions" directory

            # ImageStitchingStep automatically uses the positions directory
            ImageStitchingStep(
                name="Stitch Images")
                # input_dir is automatically set to "path/to/processed"
                # positions_dir is automatically determined from the previous step
                # output_dir is automatically set to a "stitched" directory
        ],
        name="Complete Pipeline"
    )

Customizing Directories
^^^^^^^^^^^^^^^^^^^

While automatic directory resolution works for most cases, you can customize directories when needed:

.. code-block:: python

    # Create custom directories
    from pathlib import Path

    # Specify custom directories for specific steps
    step1 = Step(
        name="Custom Directory Step",
        func=IP.stack_percentile_normalize,
        input_dir=Path("path/to/custom/input"),
        output_dir=Path("path/to/custom/output")
    )

    # Specialized steps can also use custom directories
    position_step = PositionGenerationStep(
        name="Custom Positions Step",
        input_dir=Path("path/to/custom/input"),
        output_dir=Path("path/to/custom/positions")
    )

    stitch_step = ImageStitchingStep(
        name="Custom Stitch Step",
        input_dir=Path("path/to/custom/input"),
        positions_dir=Path("path/to/custom/positions"),
        output_dir=Path("path/to/custom/stitched")
    )

Directory Creation and Well Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the pipeline runs, directories are created automatically as needed:

1. Each well gets its own subfolder during processing
2. The pipeline creates output directories if they don't exist
3. Directory structure mirrors the plate structure
4. Original directory structure is preserved in output directories

This approach ensures that:
- Image loading is based on folder state
- Well data remains organized and separate
- Original file organization is maintained

Putting It All Together
--------------------

Let's see how all these concepts work together in a complete example:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
    from ezstitcher.core.utils import stack

    # 1. Create configuration
    config = PipelineConfig(
        num_workers=2  # Use 2 worker threads
    )

    # 2. Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path="path/to/plate"
    )

    # 3. Set up directory structure
    dirs = orchestrator.setup_directories()

    # 4. Create position generation pipeline
    position_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks
            Step(name="Z-Stack Flattening",
                 func=IP.create_projection,
                 variable_components=['z_index'],
                 processing_args={'method': 'max_projection'},
                 input_dir=dirs['input'],
                 output_dir=dirs['processed']),

            # Step 2: Enhance images
            Step(name="Image Enhancement",
                 func=[stack(IP.sharpen),
                      IP.stack_percentile_normalize],
                 variable_components=['channel']),

            # Step 3: Generate positions
            PositionGenerationStep(
                name="Generate Positions",
                output_dir=dirs['positions'])
        ],
        name="Position Generation Pipeline"
    )

    # 5. Create image assembly pipeline
    assembly_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks
            Step(name="Z-Stack Flattening",
                 func=IP.create_projection,
                 variable_components=['z_index'],
                 processing_args={'method': 'max_projection'},
                 input_dir=dirs['input'],
                 output_dir=dirs['post_processed']),

            # Step 2: Process channels
            Step(name="Channel Processing",
                 func=IP.stack_percentile_normalize,
                 variable_components=['channel']),

            # Step 3: Stitch images
            ImageStitchingStep(
                name="Stitch Images",
                positions_dir=dirs['positions'],
                output_dir=dirs['stitched'])
        ],
        name="Image Assembly Pipeline"
    )

    # 6. Run the orchestrator with both pipelines
    success = orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

Data Flow
^^^^^^^

Here's how data flows through this example:

1. The orchestrator detects wells in the plate folder
2. For each well, it runs the position_pipeline:
   a. Z-Stack Flattening step processes each z-index separately
   b. Image Enhancement step applies sharpening and normalization
   c. Generate Positions step calculates tile positions
3. Then it runs the assembly_pipeline:
   a. Z-Stack Flattening step processes each z-index separately
   b. Channel Processing step normalizes each channel
   c. Stitch Images step stitches the images using the positions

The result is a set of stitched images for each well and channel.
