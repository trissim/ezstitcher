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

Option 2: AutoPipelineFactory (More Control)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more control over the stitching process:

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    from pathlib import Path

    # Path to your microscopy data
    plate_path = Path("path/to/your/microscopy/data")

    # Create an orchestrator to manage the stitching process
    orchestrator = PipelineOrchestrator(plate_path=plate_path)

    # Create a pipeline factory with default settings
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True  # Apply normalization to improve image quality
    )

    # Create the stitching pipelines
    pipelines = factory.create_pipelines()

    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

    # Output will be in a directory named after your input with "_stitched" appended
    print(f"Stitched images saved to: {plate_path.parent / f'{plate_path.name}_stitched'}")

Expected Output
-------------

After running the example:

1. The console will show progress information as the pipelines run
2. Upon successful completion, you'll see a message indicating the pipelines completed
3. Stitched images will be saved in a new directory with "_stitched" appended to the original directory name

What's Next
---------

Now that you've run your first stitching pipeline, you can:

- Learn about ezstitcher's architecture in the :doc:`../user_guide/introduction`
- Explore more detailed examples in the :doc:`../user_guide/basic_usage` guide
- Learn about the simplified interface in the :doc:`../user_guide/ez_module` guide
- Try different parameters for the :doc:`../concepts/pipeline_factory`
