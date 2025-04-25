.. _specialized-steps:

=================
Specialized Steps
=================

EZStitcher includes specialized Step subclasses for common tasks that leverage the orchestrator's plate-specific services. These steps are designed to work seamlessly with the orchestrator to handle plate-specific operations.

.. _position-generation-step:

PositionGenerationStep
---------------------

The ``PositionGenerationStep`` generates position files for stitching by leveraging the orchestrator's plate-specific services:

.. code-block:: python

    from ezstitcher.core.steps import PositionGenerationStep

    # Create a position generation step
    step = PositionGenerationStep(
        input_dir=None,  # Optional: Input directory containing images to analyze
        output_dir=None  # Optional: Output directory for position files
    )

This step:
1. Accesses the orchestrator through the context to get plate-specific configuration
2. Uses the orchestrator's microscope handler to understand the plate format
3. Analyzes the images to find overlapping regions using the orchestrator's stitcher
4. Calculates the relative positions of tiles
5. Saves the positions to CSV files

Behind the scenes, the step uses the orchestrator's `generate_positions` method, which is configured with the right parameters for the specific plate type:

.. code-block:: python

    # Inside the PositionGenerationStep's process method
    def process(self, context):
        # Get the orchestrator
        orchestrator = context.orchestrator

        # Use the orchestrator's position generation service
        # This handles all the plate-specific details
        positions_file, _ = orchestrator.generate_positions(
            well=context.well,
            input_dir=context.input_dir,
            positions_dir=context.output_dir
        )

        # Store the positions file in the context for other steps to use
        context.positions_file = positions_file
        return context

.. _image-stitching-step:

ImageStitchingStep
-----------------

The ``ImageStitchingStep`` stitches images using position files and the orchestrator's stitching services:

.. code-block:: python

    from ezstitcher.core.steps import ImageStitchingStep

    # Create an image stitching step
    step = ImageStitchingStep(
        input_dir=None,      # Optional: Directory containing images to stitch
        positions_dir=None,  # Optional: Directory containing position files
        output_dir=None      # Optional: Directory to save stitched images
    )

This step:
1. Accesses the orchestrator through the context to get plate-specific configuration
2. Uses the orchestrator's microscope handler to understand the plate format
3. Loads the position files
4. Uses the orchestrator's stitcher (configured for the specific plate) to stitch the images
5. Saves the stitched images

Behind the scenes, the step uses the orchestrator's `stitch_images` method, which is configured with the right parameters for the specific plate type:

.. code-block:: python

    # Inside the ImageStitchingStep's process method
    def process(self, context):
        # Get the orchestrator
        orchestrator = context.orchestrator

        # Find the positions file
        positions_file = self._find_positions_file(context)

        # Use the orchestrator's stitching service
        # This handles all the plate-specific details
        orchestrator.stitch_images(
            well=context.well,
            input_dir=context.input_dir,
            output_dir=context.output_dir,
            positions_file=positions_file
        )

        return context

.. _orchestrator-step-interaction:

Orchestrator-Step Interaction
---------------------------

The specialized steps leverage the orchestrator's services to handle plate-specific operations:

1. **Plate Format Understanding**: The orchestrator's microscope handler knows how to interpret filenames and folder structures for different plate types.

2. **Stitching Configuration**: The orchestrator provides a stitcher configured with the right parameters (tile overlap, margin ratio, etc.) for the specific plate type.

3. **Position Generation**: The orchestrator handles the details of generating positions based on the plate format.

4. **Image Loading**: The orchestrator uses ImageLocator to find the actual image directory within the plate path.

This abstraction allows the steps to focus on their specific tasks without needing to know the details of different plate formats.

.. _specialized-step-parameters:

Specialized Step Parameters
----------------------

PositionGenerationStep Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``input_dir``: Directory containing images to analyze (optional)
* ``output_dir``: Directory to save position files (optional)

The ``PositionGenerationStep`` doesn't use the ``func``, ``variable_components``, or ``group_by`` parameters since it has a fixed purpose.

ImageStitchingStep Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``input_dir``: Directory containing images to stitch (optional)
* ``positions_dir``: Directory containing position files (optional)
* ``output_dir``: Directory to save stitched images (optional)

The ``ImageStitchingStep`` doesn't use the ``func``, ``variable_components``, or ``group_by`` parameters since it has a fixed purpose.

.. _when-to-use-specialized-steps:

When to Use Specialized Steps
---------------------------

Use specialized steps when you need the specific functionality they provide. For general image processing tasks, use the base ``Step`` class. The specialized steps are designed to work seamlessly with the orchestrator to handle plate-specific operations.

.. _specialized-steps-best-practices:

Specialized Step Best Practices
-----------------------------

Here are some key recommendations for using specialized steps:

.. _specialized-steps-directory-resolution:

1. **Directory Resolution**:
   - Let EZStitcher automatically resolve directories when possible
   - Only specify directories when you need a specific directory structure
   - You can explicitly set ``input_dir=orchestrator.workspace_path`` to use original images for stitching

2. **Step Order**:
   - Place ``PositionGenerationStep`` after image processing steps
   - Place ``ImageStitchingStep`` after ``PositionGenerationStep``
   - This ensures that position generation works with processed images

For comprehensive best practices for specialized steps, see :ref:`best-practices-specialized-steps` in the :doc:`../user_guide/best_practices` guide.

.. _typical-stitching-workflows:

Typical Stitching Workflows
-------------------------

Here are some common workflows that use specialized steps:

Basic Stitching Workflow
^^^^^^^^^^^^^^^^^^^^^

A typical stitching workflow involves these main steps:

1. Process images to enhance features (optional)
2. Generate position files that describe how the tiles fit together
3. Stitch the images using these position files

.. code-block:: python

    from ezstitcher.core.steps import PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create a pipeline for stitching
    stitching_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched",
        steps=[
            # Process images (optional)
            Step(
                func=IP.stack_percentile_normalize,
                input_dir=orchestrator.workspace_path
            ),

            # Generate positions
            PositionGenerationStep(),

            # Stitch images
            ImageStitchingStep()
        ],
        name="Stitching Pipeline"
    )

    # Run the pipeline
    orchestrator.run(pipelines=[stitching_pipeline])

Multi-Channel Stitching
^^^^^^^^^^^^^^^^^^^^

When working with multiple channels, it's important to create a composite image before position generation:

.. code-block:: python

    # Create a pipeline for multi-channel stitching
    multi_channel_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched",
        steps=[
            # Process channels
            Step(
                func=IP.stack_percentile_normalize,
                variable_components=['channel'],
                input_dir=orchestrator.workspace_path
            ),

            # Create composite image for position generation
            Step(
                func=IP.create_composite,  # Equal weighting for all channels
                variable_components=['channel']
            ),

            # Generate positions
            PositionGenerationStep(),

            # Stitch images
            ImageStitchingStep()
        ],
        name="Multi-Channel Stitching Pipeline"
    )

Using Original Images for Stitching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes you want to process images for position generation but use the original images for stitching:

.. code-block:: python

    # Create a pipeline that uses processed images for position generation
    # but original images for stitching
    original_stitching_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched",
        steps=[
            # Process images for position generation
            Step(
                func=IP.stack_percentile_normalize,
                input_dir=orchestrator.workspace_path,
                output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_processed"
            ),

            # Generate positions using processed images
            PositionGenerationStep(),

            # Stitch using original images
            ImageStitchingStep(
                input_dir=orchestrator.workspace_path  # Use original images for stitching
            )
        ],
        name="Original Image Stitching Pipeline"
    )
