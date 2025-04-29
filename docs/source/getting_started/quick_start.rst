Quick Start
==========

This guide will help you get started with ezstitcher quickly. In just a few minutes, you'll install the package and run a basic image stitching pipeline.

Prerequisites
-----------

- Python 3.11
- Basic understanding of Python
- Some microscopy images to stitch

Installation
----------

If you haven't installed ezstitcher yet, follow the :doc:`installation` guide.

Basic Examples
-------------

There are two ways to get started with EZStitcher:

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

    # Create position generation pipeline
    pos_pipe = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            ZFlatStep(),
            NormStep(),
            CompositeStep(),
            PositionGenerationStep(),
        ],
        name="Position Generation",
    )
    positions_dir = pos_pipe.steps[-1].output_dir

    # Create assembly pipeline
    asm_pipe = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=plate_path.parent / f"{plate_path.name}_stitched",
        steps=[
            NormStep(),
            ImageStitchingStep(positions_dir=positions_dir),
        ],
        name="Assembly",
    )

    # Run the pipelines
    orchestrator.run(pipelines=[pos_pipe, asm_pipe])

Expected Output
-------------

After running the example:

1. The console will show progress information as the pipelines run
2. Upon successful completion, you'll see a message indicating the pipelines completed
3. Stitched images will be saved in a new directory with "_stitched" appended to the original directory name

What's Next
---------

Now that you've run your first stitching pipeline, you can:

- Learn more about basic usage in the :doc:`../user_guide/basic_usage` guide
- Learn about ezstitcher's architecture in the :doc:`../user_guide/introduction`
- Explore custom pipelines in the :doc:`../user_guide/intermediate_usage` guide
- Discover advanced features in the :doc:`../user_guide/advanced_usage` guide
