=======
Pipeline
=======

Overview
-------

A ``Pipeline`` is a sequence of processing steps that are executed in order. It provides:

* Step management (adding, removing, reordering)
* Context passing between steps
* Input/output directory management

Creating a Pipeline
-----------------

The recommended way to create a pipeline is to provide all steps at once during initialization:

.. code-block:: python

    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create a pipeline with all steps at once (recommended approach)
    pipeline = Pipeline(
        steps=[
            Step(name="Z-Stack Flattening",
                 func=(IP.create_projection, {'method': 'max_projection'}),
                 variable_components=['z_index'],
                 input_dir=orchestrator.workspace_path),

            Step(name="Image Enhancement",
                 func=IP.stack_percentile_normalize),

            PositionGenerationStep(name="Generate Positions")
        ],
        name="My Processing Pipeline"
    )

Alternatively, you can add steps one by one using the ``add_step()`` method:

.. code-block:: python

    # Create an empty pipeline
    pipeline = Pipeline(name="My Processing Pipeline")

    # Add steps one by one
    pipeline.add_step(Step(name="Z-Stack Flattening",
                          func=(IP.create_projection, {'method': 'max_projection'}),
                          variable_components=['z_index'],
                          input_dir=orchestrator.workspace_path))

    pipeline.add_step(Step(name="Image Enhancement",
                          func=IP.stack_percentile_normalize))

    pipeline.add_step(PositionGenerationStep(name="Generate Positions"))

The first approach (providing all steps at once) is recommended for most use cases as it's more concise and easier to understand. The second approach (adding steps one by one) is useful for dynamic scenarios where steps need to be added conditionally or configured based on the output of previous steps.

Running a Pipeline
----------------

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
--------------

When a pipeline runs, it creates a ``ProcessingContext`` that is passed from step to step. This context holds:

* Input/output directories
* Well filter
* Configuration
* Results from previous steps

This allows steps to communicate and build on each other's results.

Directory Resolution
------------------

EZStitcher automatically resolves directories for steps in a pipeline, minimizing the need for manual directory management.

For detailed information on directory resolution, directory flow, and best practices, see :doc:`directory_structure`.
