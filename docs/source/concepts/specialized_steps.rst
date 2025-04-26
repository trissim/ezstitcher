.. _specialized-steps:

=================
Specialized Steps
=================

EZStitcher provides two types of specialized steps:

1. **Orchestrator-specific steps** that leverage the orchestrator's plate-specific services for operations like position generation and image stitching.

2. **Step factories** that inherit from the regular :class:`Step` class and provide a higher-level interface for common image processing operations like Z-stack flattening, focus selection, and channel compositing.

Both types of specialized steps are designed to simplify your code and reduce boilerplate while maintaining the power and flexibility of the EZStitcher pipeline architecture.

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

.. _step-factories:

Step Factories
------------

In addition to the specialized steps that work with the orchestrator, EZStitcher provides step factory classes that inherit from the regular ``Step`` class and pre-configure parameters for common operations.

Step factories follow the "factory pattern" design principle, creating pre-configured :class:`Step` instances with appropriate parameters for specific tasks. This approach offers several benefits:

- **Simplified Interface**: Fewer parameters to configure manually
- **Pre-configured Parameters**: Appropriate defaults for common operations
- **Semantic Names**: Clear naming that indicates the step's purpose
- **Reduced Boilerplate**: Less code to write for common operations
- **Consistent Patterns**: Standardized approach to common tasks

Here's a comparison of raw Steps vs. specialized steps for common operations:

.. list-table:: Raw Steps vs. Specialized Steps
   :header-rows: 1
   :widths: 20 40 40

   * - Operation
     - Raw Step
     - Specialized Step
   * - Z-stack flattening
     - .. code-block:: python

          Step(
              func=(IP.create_projection,
                    {'method': 'max_projection'}),
              variable_components=['z_index'],
              group_by=None,
              name="Maximum Intensity Projection"
          )
     - .. code-block:: python

          ZFlatStep(
              method="max"
          )
   * - Focus selection
     - .. code-block:: python

          focus_analyzer = FocusAnalyzer(
              metric='laplacian'
          )
          Step(
              func=(IP.create_projection,
                    {'method': 'best_focus',
                     'focus_analyzer': focus_analyzer}),
              variable_components=['z_index'],
              group_by=None,
              name="Best Focus (laplacian)"
          )
     - .. code-block:: python

          FocusStep(
              focus_options={
                  'metric': 'laplacian'
              }
          )
   * - Channel compositing
     - .. code-block:: python

          Step(
              func=(IP.create_composite,
                    {'weights': [0.7, 0.3]}),
              variable_components=['channel'],
              group_by=None,
              name="Channel Composite"
          )
     - .. code-block:: python

          CompositeStep(
              weights=[0.7, 0.3]
          )

EZStitcher provides the following step factories:

ZFlatStep
^^^^^^^

The ``ZFlatStep`` is a specialized step for Z-stack flattening:

.. code-block:: python

    from ezstitcher.core.step_factories import ZFlatStep

    # Create a maximum intensity projection step
    step = ZFlatStep(
        method="max",  # Options: "max", "mean", "median", "min", "std", "sum"
        input_dir=orchestrator.workspace_path
    )

This step pre-configures:
- ``variable_components=['z_index']``
- ``group_by=None``
- ``func=(IP.create_projection, {'method': method})``

FocusStep
^^^^^^^

The ``FocusStep`` is a specialized step for focus-based Z-stack processing:

.. code-block:: python

    from ezstitcher.core.step_factories import FocusStep

    # Create a best focus step
    step = FocusStep(
        focus_options={'metric': 'laplacian'},  # Focus metric options
        input_dir=orchestrator.workspace_path
    )

This step pre-configures:
- ``variable_components=['z_index']``
- ``group_by=None``
- ``func=(IP.create_projection, {'method': 'best_focus', 'focus_analyzer': focus_analyzer})``

CompositeStep
^^^^^^^^^^

The ``CompositeStep`` is a specialized step for creating composite images from multiple channels:

.. code-block:: python

    from ezstitcher.core.step_factories import CompositeStep

    # Create a composite step with custom weights
    step = CompositeStep(
        weights=[0.7, 0.3],  # 70% channel 1, 30% channel 2
        input_dir=orchestrator.workspace_path
    )

This step pre-configures:
- ``variable_components=['channel']``
- ``group_by=None``
- ``func=(IP.create_composite, {'weights': weights})``

.. _when-to-use-specialized-steps:

When to Use Specialized Steps
---------------------------

**Use orchestrator-specific steps when:**

- You need to generate position files for stitching (``PositionGenerationStep``)
- You need to stitch images using position files (``ImageStitchingStep``)
- You're working with plate-specific operations that leverage the orchestrator

**Use step factories when:**

- You need to perform common operations like Z-stack flattening, focus selection, or channel compositing
- You want to reduce boilerplate code and simplify your pipeline
- You prefer a more intuitive interface for common tasks
- You're building pipelines for non-expert users

**Use raw Steps when:**

- You need to perform custom operations not covered by specialized steps
- You need fine-grained control over all parameters
- You're building complex workflows with custom function chains
- You're creating your own specialized steps

As a general rule, start with specialized steps for common operations before falling back to raw Steps. This approach will make your code more concise, readable, and maintainable.

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

3. **Step Factory Usage**:
   - Start with step factories for common operations before falling back to raw Steps
   - Combine step factories with raw Steps when needed for complex workflows
   - Consider creating custom step factories for operations you perform frequently

4. **Custom Step Factories**:
   - Use consistent naming when creating custom step factories
   - Document pre-configured parameters in custom step factories
   - Consider variable components carefully when creating custom step factories
   - Test step factories thoroughly to ensure they behave as expected

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

Here's an example using both specialized steps and step factories:

.. code-block:: python

    from ezstitcher.core.steps import PositionGenerationStep, ImageStitchingStep, Step
    from ezstitcher.core.step_factories import ZFlatStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create a pipeline for stitching
    stitching_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched",
        steps=[
            # Flatten Z-stacks using ZFlatStep (if working with Z-stacks)
            ZFlatStep(method="max"),

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

When working with multiple channels, it's important to create a composite image before position generation. Using step factories makes this more concise:

.. code-block:: python

    from ezstitcher.core.steps import PositionGenerationStep, ImageStitchingStep, Step
    from ezstitcher.core.step_factories import CompositeStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

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

            # Create composite image for position generation using CompositeStep
            CompositeStep(),  # Equal weighting for all channels by default

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

.. _creating-custom-step-factories:

Creating Custom Step Factories
---------------------------

You can create your own step factories for operations you perform frequently. Here's an example of a custom step factory for adaptive histogram equalization:

.. code-block:: python

    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_processor import ImageProcessor as IP
    from typing import Optional, Union, List
    from pathlib import Path

    class AdaptiveHistogramStep(Step):
        """
        Specialized step for adaptive histogram equalization.

        This step performs adaptive histogram equalization on images to enhance contrast.
        It pre-configures variable_components=['site'] and group_by=None.
        """

        def __init__(
            self,
            clip_limit: float = 0.03,
            tile_grid_size: tuple = (8, 8),
            input_dir: Optional[Union[str, Path]] = None,
            output_dir: Optional[Union[str, Path]] = None,
            well_filter: Optional[List[str]] = None,
        ):
            """
            Initialize an adaptive histogram equalization step.

            Args:
                clip_limit: Clipping limit for contrast enhancement (default: 0.03)
                tile_grid_size: Size of grid for local histogram equalization (default: (8, 8))
                input_dir: Input directory
                output_dir: Output directory
                well_filter: Wells to process
            """
            # Initialize the Step with pre-configured parameters
            super().__init__(
                func=(IP.adaptive_histogram_equalization, {
                    'clip_limit': clip_limit,
                    'tile_grid_size': tile_grid_size
                }),
                variable_components=['site'],  # Process each site individually
                group_by=None,
                input_dir=input_dir,
                output_dir=output_dir,
                well_filter=well_filter,
                name="Adaptive Histogram Equalization"
            )

    # Usage example
    step = AdaptiveHistogramStep(
        clip_limit=0.02,
        tile_grid_size=(16, 16),
        input_dir=orchestrator.workspace_path
    )
