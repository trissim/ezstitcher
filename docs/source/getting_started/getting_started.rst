Getting Started with EZStitcher
===========================

This guide will help you install EZStitcher and run your first image stitching pipeline in just a few minutes.

Installation
-----------

**System Requirements**

- Python 3.11 (only supported version)
- 8GB RAM minimum (16GB recommended for large images)
- Multi-core CPU recommended for faster processing

**Quick Install**

.. code-block:: bash

    pip install ezstitcher

**Using pyenv (recommended)**

If you need to install Python 3.11, we recommend using `pyenv <https://github.com/pyenv/pyenv>`_:

.. code-block:: bash

    # Install Python 3.11 with pyenv
    pyenv install 3.11
    pyenv local 3.11

    # Create virtual environment and install ezstitcher
    python -m venv .venv
    source .venv/bin/activate
    pip install ezstitcher

**Verify Installation**

.. code-block:: bash

    python -c "import ezstitcher; print('EZStitcher installed successfully')"

Basic Usage
----------

There are two ways to use EZStitcher, depending on your needs:

Option 1: EZ Module (Recommended for Beginners)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EZ module provides a simplified interface that requires minimal code:

.. code-block:: python

    from ezstitcher import stitch_plate
    from pathlib import Path

    # Path to your microscopy data
    plate_path = Path("path/to/your/microscopy/data")

    # Stitch the plate with a single function call
    stitch_plate(plate_path)

    # That's it! Output will be in a directory named after your input with "_stitched" appended

This single line will:

1. Automatically detect the plate format
2. Process all channels and Z-stacks appropriately
3. Generate positions and stitch images
4. Save the output to a new directory

**Key Parameters**

While the default settings work well for most cases, you can customize the behavior:

.. code-block:: python

    stitch_plate(
        "path/to/plate",                    # Input directory with microscopy images
        output_path="path/to/output",       # Where to save results (optional)
        normalize=True,                     # Apply intensity normalization (default: True)
        flatten_z=True,                     # Flatten Z-stacks to 2D (auto-detected if None)
        z_method="max",                     # How to flatten Z-stacks: "max", "mean", "focus"
        channel_weights=[0.7, 0.3, 0],      # Weights for position finding (auto-detected if None)
        well_filter=["A01", "B02"]          # Process only specific wells (optional)
    )

**Z-Stack Processing**

For plates with Z-stacks, you can control how they're flattened:

.. code-block:: python

    # Maximum intensity projection (brightest pixel from each Z-stack)
    stitch_plate("path/to/plate", flatten_z=True, z_method="max")

    # Focus-based projection (selects best-focused plane)
    stitch_plate("path/to/plate", flatten_z=True, z_method="focus")

    # Mean projection (average across Z-planes)
    stitch_plate("path/to/plate", flatten_z=True, z_method="mean")

Option 2: Custom Pipelines (For Advanced Flexibility)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced users who need more control and flexibility:

.. code-block:: python

    from pathlib import Path
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, NormStep, PositionGenerationStep, ImageStitchingStep, ZFlatStep, CompositeStep

    # Path to your microscopy data
    plate_path = Path("path/to/your/microscopy/data")

    # Create an orchestrator to manage the stitching process
    orchestrator = PipelineOrchestrator(plate_path=plate_path)

    # Position generation pipeline
    pos_pipe = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            ZFlatStep(),                # Flatten Z-stacks
            NormStep(),                 # Normalize to enhance contrast
            CompositeStep(),            # Create composite from channels
            PositionGenerationStep()    # Generate positions
        ],
        name="Position Generation"
    )
    positions_dir = pos_pipe.steps[-1].output_dir

    # Assembly pipeline
    asm_pipe = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            NormStep(),                 # Normalize to enhance contrast
            ImageStitchingStep(positions_dir=positions_dir)  # Stitch images
        ],
        name="Assembly"
    )

    # Run pipelines
    orchestrator.run(pipelines=[pos_pipe, asm_pipe])

    # Note: This follows EZStitcher's standard pipeline pattern:
    # 1. Position Generation: Z-flattening → Normalization → Channel compositing → Position generation
    # 2. Assembly: Normalization → Image stitching
    # This pattern works for all scenarios (single/multi-channel, with/without Z-stacks)

Understanding Key Concepts
-----------------------

Here are the key concepts you need to understand:

**Plates and Wells**

EZStitcher processes microscopy data organized in plates and wells. A plate contains multiple wells, and each well contains multiple images.

**Images and Channels**

Microscopy images can have multiple channels (e.g., DAPI, GFP, RFP) and Z-stacks (multiple focal planes).

**Processing Steps**

Behind the scenes, EZStitcher processes images through a series of steps:

- Z-flattening: Converting 3D Z-stacks into 2D images
- Normalization: Adjusting image intensity for consistent visualization
- Channel compositing: Combining multiple channels into a single image
- Position generation: Finding the relative positions of tiles
- Image stitching: Combining tiles into a complete image

These steps are organized into two standard pipelines:

1. **Position Generation Pipeline**: Z-flattening → Normalization → Channel compositing → Position generation
2. **Assembly Pipeline**: Normalization → Image stitching

The EZ module handles all these steps automatically, so you don't need to worry about them unless you need more control.

Troubleshooting
------------

**Common issues:**

- **No output**: Check that the input path exists and contains microscopy images
- **Z-stacks not detected**: Explicitly set ``flatten_z=True``
- **Poor quality**: Try different ``z_method`` values or adjust ``channel_weights``

Next Steps
---------

Now that you've run your first stitching pipeline, you can:

- See :doc:`../user_guide/basic_usage` for more detailed examples and options with the EZ module
- Explore :doc:`../concepts/architecture_overview` to learn about EZStitcher's architecture
- See :doc:`../user_guide/intermediate_usage` to learn how to create custom pipelines with steps
- Discover advanced features in :doc:`../user_guide/advanced_usage`
- Check out best practices in :doc:`../user_guide/best_practices`
