===================
Directory Structure
===================

Overview
--------

EZStitcher uses a structured approach to directory management that balances automation with flexibility. This document explains how directories are managed, resolved, and customized in EZStitcher.

Basic Directory Concepts
-----------------------

In EZStitcher, several key directories are used during processing:

* **Plate Path**: The original directory containing microscopy images
* **Workspace Path**: A copy of the plate path with symlinks to protect original data
* **Input Directory**: Where a step reads images from
* **Output Directory**: Where a step saves processed images
* **Positions Directory**: Where position files for stitching are saved
* **Stitched Directory**: Where final stitched images are saved

Default Directory Structure
-------------------------

When you run a pipeline, EZStitcher creates a directory structure as steps are executed:

.. code-block:: text

    /path/to/plate/                  # Original plate path
    /path/to/plate_workspace/        # Workspace with symlinks to original images
    /path/to/plate_workspace/_out/   # Processed images (configurable suffix)
    /path/to/plate_workspace/_positions/  # Position files for stitching (configurable suffix)
    /path/to/plate_workspace/_stitched/   # Stitched images (configurable suffix)

This structure ensures that:

1. Original data is protected (via the workspace)
2. Processed images are kept separate from original images
3. Position files are stored in a dedicated directory
4. Stitched images are stored separately from individual processed tiles

Directory Resolution
------------------

EZStitcher automatically resolves directories for steps in a pipeline, minimizing the need for manual directory management. Here's how it works:

1. **Basic Resolution Logic**:

   .. code-block:: text

       Pipeline Input Dir → Step 1 → Step 2 → Step 3 → ... → Pipeline Output Dir
                            |         |         |
                            v         v         v
                         Output 1  Output 2  Output 3

   - Each step's output directory becomes the next step's input directory
   - If a step doesn't specify an output directory, it's automatically generated
   - The pipeline's output directory is used for the last step if not specified

2. **First Step Special Handling**:
   - If the first step doesn't specify an input directory, the pipeline's input directory is used
   - Typically, you should set the first step's input directory to ``orchestrator.workspace_path``

3. **Default Directory Generation**:
   - The first step always gets a new output directory (with "_out" suffix) if none is specified
   - This ensures we never modify files in the workspace path
   - Subsequent steps will use their input directory as their output directory (in-place processing) if no output directory is specified
   - This allows for more efficient processing by avoiding unnecessary file copying

4. **ImageStitchingStep Special Handling**:
   - The ``ImageStitchingStep`` has special directory handling to ensure stitched images are saved separately
   - By default, its output directory is set to ``{workspace_path}/_stitched``
   - This ensures stitched images are not mixed with processed individual tiles

Example Directory Flow
--------------------

Here's an example of how directories flow through a pipeline:

.. code-block:: text

    # Starting with a plate path: /data/plates/plate1

    orchestrator.workspace_path = /data/plates/plate1_workspace

    # Pipeline with 3 steps:

    Step 1 (Z-Stack Flattening):
      input_dir = /data/plates/plate1_workspace
      output_dir = /data/plates/plate1_workspace/_out  # New directory to protect workspace

    Step 2 (Channel Processing):
      input_dir = /data/plates/plate1_workspace/_out
      output_dir = /data/plates/plate1_workspace/_out  # In-place processing

    Step 3 (Image Stitching):
      input_dir = /data/plates/plate1_workspace/_out
      positions_dir = /data/plates/plate1_workspace/_positions
      output_dir = /data/plates/plate1_workspace/_stitched  # New directory for stitched images

This automatic directory resolution simplifies pipeline creation and ensures a consistent directory structure.

Step Initialization Best Practices
--------------------------------

When initializing steps, follow these best practices for directory specification:

1. **First Step in a Pipeline**:
   - Always specify ``input_dir`` for the first step, typically using ``orchestrator.workspace_path``
   - This ensures that processing happens on the workspace copies, not the original data
   - Specify ``output_dir`` only if you need a specific directory structure

   .. code-block:: python

       # First step in a pipeline
       first_step = Step(
           name="First Step",
           func=IP.stack_percentile_normalize,
           input_dir=orchestrator.workspace_path,  # Always specify for first step
           # output_dir is automatically determined
       )

2. **Subsequent Steps**:
   - Don't specify ``input_dir`` for subsequent steps
   - Each step's output directory automatically becomes the next step's input directory
   - Specify ``output_dir`` only if you need a specific directory structure

   .. code-block:: python

       # Subsequent step in a pipeline
       subsequent_step = Step(
           name="Subsequent Step",
           func=stack(IP.sharpen),
           # input_dir is automatically set to previous step's output_dir
           # output_dir is automatically determined
       )

3. **Specialized Steps**:
   - For ``PositionGenerationStep``, don't specify ``input_dir`` or ``output_dir`` unless needed
   - For ``ImageStitchingStep``, don't specify ``input_dir``, ``positions_dir``, or ``output_dir`` unless needed

   .. code-block:: python

       # Position generation step
       position_step = PositionGenerationStep(
           name="Generate Positions"
           # input_dir is automatically set to previous step's output_dir
           # output_dir is automatically determined
       )

       # Image stitching step
       stitch_step = ImageStitchingStep(
           name="Stitch Images"
           # input_dir is automatically set
           # positions_dir is automatically determined from previous steps
           # output_dir is automatically determined
       )

4. **Common Mistakes to Avoid**:
   - Specifying unnecessary directories, making the code more verbose
   - Forgetting to use ``orchestrator.workspace_path`` for the first step
   - Manually managing directories that could be automatically resolved

Following these best practices will make your code more concise and less error-prone, while taking full advantage of EZStitcher's automatic directory resolution.

Custom Directory Structures
-------------------------

While EZStitcher's automatic directory resolution works well for most cases, you may sometimes need more control over where files are saved.

You can create custom directory structures by explicitly specifying output directories:

.. code-block:: python

    # Create a pipeline with custom directory structure
    pipeline = Pipeline(
        steps=[
            # First step: Save to a specific directory
            Step(
                name="Z-Stack Flattening",
                func=(IP.create_projection, {'method': 'max_projection'}),
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path,
                output_dir=Path("/custom/output/path/flattened")
            ),

            # Second step: Save to another specific directory
            Step(
                name="Channel Processing",
                func=IP.stack_percentile_normalize,
                variable_components=['channel'],
                group_by='channel',
                # input_dir is automatically set to the previous step's output_dir
                output_dir=Path("/custom/output/path/processed")
            ),

            # Image stitching step: Save to a specific directory
            ImageStitchingStep(
                name="Stitch Images",
                # input_dir is automatically set to the previous step's output_dir
                # positions_dir is automatically determined
                output_dir=Path("/custom/output/path/stitched")
            )
        ],
        name="Custom Directory Pipeline"
    )

Customizing ImageStitchingStep Directories
----------------------------------------

For more control over the ImageStitchingStep directories:

.. code-block:: python

    pipeline = Pipeline(
        steps=[
            # Processing steps...

            # Custom position generation step
            PositionGenerationStep(
                name="Generate Positions",
                # input_dir is automatically set
                output_dir=Path("/custom/positions")  # Custom positions directory
            ),

            # Custom image stitching step
            ImageStitchingStep(
                name="Stitch Images",
                input_dir=Path("/custom/input"),  # Custom input directory
                positions_dir=Path("/custom/positions"),  # Custom positions directory
                output_dir=Path("/custom/stitched")  # Custom output directory
            )
        ],
        name="Custom Stitching Pipeline"
    )

When to Specify Directories Explicitly
------------------------------------

1. **Always specify input_dir for the first step**:
   - Use `orchestrator.workspace_path` to ensure processing happens on workspace copies
   - This protects original data from modification

2. **Specify output_dir only when you need a specific directory structure**:
   - For example, when you need to save results in a specific location
   - When you need to reference the output directory from outside the pipeline

3. **Don't specify input_dir for subsequent steps**:
   - Each step's output directory automatically becomes the next step's input directory
   - This reduces code verbosity and potential for errors

4. **Don't specify directories for specialized steps unless needed**:
   - `PositionGenerationStep` and `ImageStitchingStep` have intelligent directory handling
   - They automatically find the right directories based on the pipeline context

Configuring Directory Suffixes
-------------------------

EZStitcher allows you to configure the directory suffixes used for different types of steps through the `PipelineConfig` class:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig

    # Create a configuration with custom directory suffixes
    config = PipelineConfig(
        out_dir_suffix="_output",           # For regular processing steps (default: "_out")
        processed_dir_suffix="_proc",       # For intermediate processing steps (default: "_processed")
        positions_dir_suffix="_pos",        # For position generation steps (default: "_positions")
        stitched_dir_suffix="_stitched_images"  # For stitching steps (default: "_stitched")
    )

    # Create an orchestrator with the custom configuration
    orchestrator = PipelineOrchestrator(config=config, plate_path=plate_path)

    # Now all pipelines run with this orchestrator will use the custom suffixes
    pipeline = Pipeline(
        steps=[
            # Steps will use the custom suffixes for their output directories
            Step(name="First Step", func=IP.stack_percentile_normalize, input_dir=orchestrator.workspace_path),
            PositionGenerationStep(name="Generate Positions"),
            ImageStitchingStep(name="Stitch Images")
        ]
    )

    # Run the pipeline
    orchestrator.run(pipelines=[pipeline])

This allows you to customize the directory structure to match your organization's naming conventions or to integrate with existing workflows.

Directory Structure Best Practices
--------------------------------

1. **Use the workspace path for the first step**:
   - Always use `orchestrator.workspace_path` as the input directory for the first step
   - This ensures that original data is protected from modification

2. **Minimize directory specification**:
   - Only specify directories when necessary
   - Let EZStitcher handle directory resolution automatically when possible
   - This makes your code more concise and less error-prone

3. **Use consistent directory naming**:
   - Follow the default naming conventions when possible
   - Or configure custom suffixes through PipelineConfig for consistent naming
   - This makes it easier to understand the directory structure

4. **Consider performance**:
   - In-place processing (using the same directory for input and output) is more efficient
   - This is the default behavior for steps after the first step
   - Only use separate input and output directories when necessary
