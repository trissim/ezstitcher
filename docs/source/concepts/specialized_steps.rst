=================
Specialized Steps
=================

EZStitcher includes specialized Step subclasses for common tasks that leverage the orchestrator's plate-specific services.

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

Orchestrator-Step Interaction
---------------------------

The specialized steps leverage the orchestrator's services to handle plate-specific operations:

1. **Plate Format Understanding**: The orchestrator's microscope handler knows how to interpret filenames and folder structures for different plate types.

2. **Stitching Configuration**: The orchestrator provides a stitcher configured with the right parameters (tile overlap, margin ratio, etc.) for the specific plate type.

3. **Position Generation**: The orchestrator handles the details of generating positions based on the plate format.

4. **Image Loading**: The orchestrator uses ImageLocator to find the actual image directory within the plate path.

This abstraction allows the steps to focus on their specific tasks without needing to know the details of different plate formats.

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

When to Use Specialized Steps
---------------------------

Use specialized steps when you need the specific functionality they provide. For general image processing tasks, use the base ``Step`` class. The specialized steps are designed to work seamlessly with the orchestrator to handle plate-specific operations.

Specialized Step Best Practices
-----------------------------

1. **Directory Resolution**:
   - Let EZStitcher automatically resolve directories when possible
   - Only specify directories when you need a specific directory structure
   - The ``ImageStitchingStep`` follows the standard directory resolution logic, using the previous step's output directory as its input
   - You can explicitly set ``input_dir=orchestrator.workspace_path`` to use original images for stitching instead of processed images
   - The ``positions_dir`` for ``ImageStitchingStep`` is automatically determined if not specified

2. **Step Order**:
   - Place ``PositionGenerationStep`` after image processing steps
   - Place ``ImageStitchingStep`` after ``PositionGenerationStep``
   - This ensures that position generation works with processed images

3. **Pipeline Integration**:
   - Use specialized steps within a pipeline for automatic directory resolution
   - The steps will automatically access the orchestrator through the context
