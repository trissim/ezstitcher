.. _pipeline-concept:

=======
Pipeline
=======

.. _pipeline-overview:

Overview
-------

A ``Pipeline`` is a sequence of processing steps that are executed in order. It provides:

* Step management (adding, removing, reordering)
* Context passing between steps
* Input/output directory management
* Automatic directory resolution between steps

.. _pipeline-creation:

Creating a Pipeline
-----------------

The recommended way to create a pipeline is to provide all steps at once during initialization:

.. code-block:: python

    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create a pipeline with all steps at once (recommended approach)
    pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,    # Pipeline input directory
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched", # Pipeline output directory
        steps=[
            Step(
                func=(IP.create_projection, {'method': 'max_projection'}),
                variable_components=['z_index'],
                input_dir=orchestrator.workspace_path
            ),

            Step(
                func=IP.stack_percentile_normalize
            ),

            PositionGenerationStep()
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

.. _pipeline-parameters:

Pipeline Parameters
----------------

A ``Pipeline`` accepts the following parameters:

* **name**: A human-readable name for the pipeline (optional but recommended for logging)
* **steps**: A list of Step objects to execute in sequence
* **input_dir**: The directory containing input images (typically ``orchestrator.workspace_path``)
* **output_dir**: The directory where final output will be saved
* **well_filter**: List of wells to process (optional, can be overridden by the orchestrator)

Each parameter plays an important role:

* **name** helps identify the pipeline in logs and debugging output
* **steps** defines the sequence of operations to perform
* **input_dir** and **output_dir** establish the data flow boundaries
* **well_filter** allows for selective processing of specific wells

.. _pipeline-running:

Running a Pipeline
----------------

A pipeline can be run directly, but it's typically run through the orchestrator:

.. code-block:: python

    # Run through the orchestrator (recommended)
    success = orchestrator.run(pipelines=[pipeline])

    if success:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed. Check logs for details.")

    # Run directly (advanced usage)
    results = pipeline.run(
        input_dir="path/to/input",
        output_dir="path/to/output",
        well_filter=["A01", "B02"],
        orchestrator=orchestrator  # Required for microscope handler access
    )

.. _pipeline-context:

Pipeline Context
--------------

When a pipeline runs, it creates a ``ProcessingContext`` that is passed from step to step. This context holds:

* Input/output directories
* Well filter
* Configuration
* Results from previous steps
* Reference to the orchestrator

This allows steps to communicate and build on each other's results. The context is created at the beginning of pipeline execution and updated by each step as it runs.

.. _pipeline-multithreaded:

Multithreaded Processing
---------------------

Pipelines can be run in a multithreaded environment through the orchestrator:

.. code-block:: python

    # Create configuration with multithreaded processing
    config = PipelineConfig(
        num_workers=4  # Use 4 worker threads
    )

    # Create orchestrator with multithreading
    orchestrator = PipelineOrchestrator(
        config=config,
        plate_path=plate_path
    )

    # Run the pipeline with multithreading
    # Each well will be processed in a separate thread
    orchestrator.run(pipelines=[pipeline])

The number of worker threads determines how many wells can be processed concurrently. This can significantly improve performance when processing multiple wells.

.. _pipeline-directory-resolution:

Directory Resolution
------------------

EZStitcher automatically resolves directories for steps in a pipeline, minimizing the need for manual directory management.

For detailed information on directory resolution, directory flow, and best practices, see :doc:`directory_structure`.

.. _pipeline-saving-loading:

Saving and Loading Pipelines
-------------------------

While EZStitcher doesn't have built-in functions for saving and loading pipelines, you can easily save your pipeline configurations as Python scripts:

.. code-block:: python

    # save_pipeline.py
    def create_basic_pipeline(plate_path, num_workers=1):
        """Create a basic processing pipeline."""
        # Create configuration
        config = PipelineConfig(
            num_workers=num_workers
        )

        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            config=config,
            plate_path=plate_path
        )

        # Create pipeline
        pipeline = Pipeline(
            input_dir=orchestrator.workspace_path,
            output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched",
            steps=[
                # Pipeline steps...
            ],
            name="Basic Processing Pipeline"
        )

        return orchestrator, pipeline

This approach allows you to:
* Parameterize your pipelines
* Reuse pipeline configurations across projects
* Version control your pipeline configurations

.. _pipeline-best-practices:

Best Practices
------------

For comprehensive best practices on using pipelines effectively, see :ref:`best-practices-pipeline` in the :doc:`../user_guide/best_practices` guide.
