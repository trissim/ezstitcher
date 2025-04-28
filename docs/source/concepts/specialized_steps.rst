.. _specialized-steps:

=================
Specialized Steps
=================

Specialized steps extend the base Step class with specific functionality.
For detailed information about step configuration, see :doc:`step`.

EZStitcher provides two types of specialized steps:

1. **Orchestrator-specific steps** that leverage the orchestrator's plate-specific services for operations like position generation and image stitching.

2. **Step factories** that inherit from the regular :class:`Step` class and provide a higher-level interface for common image processing operations like Z-stack flattening, focus selection, and channel compositing.

Both types of specialized steps are designed to simplify your code and reduce boilerplate while maintaining the power and flexibility of the EZStitcher pipeline architecture.

.. important::
   While this document explains how specialized steps work and how to use them directly,
   ``AutoPipelineFactory`` is the recommended way to use specialized steps for most users.
   ``AutoPipelineFactory`` automatically configures and connects specialized steps for common workflows,
   making it much easier to create effective pipelines. See :doc:`pipeline_factory` for details.

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

4. **Image Loading**: The orchestrator uses FileSystemManager to find and load images from the plate path.

This abstraction allows the steps to focus on their specific tasks without needing to know the details of different plate formats.

.. _specialized-step-parameters:

Specialized Step Parameters
----------------------

For detailed API documentation, see :doc:`../api/specialized_steps`.

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

Specialized Steps
---------------

EZStitcher provides specialized step classes that inherit from the regular ``Step`` class and pre-configure parameters for common operations. These specialized steps are implemented as subclasses of Step rather than factory classes. This approach offers several benefits:

- **Simplified Interface**: Fewer parameters to configure manually
- **Pre-configured Parameters**: Appropriate defaults for common operations
- **Semantic Names**: Clear naming that indicates the step's purpose
- **Reduced Boilerplate**: Less code to write for common operations
- **Consistent Patterns**: Standardized approach to common tasks

These specialized steps are used by the :doc:`pipeline_factory` to create pre-configured pipelines for common workflows. The ``AutoPipelineFactory`` uses these specialized steps internally to create position generation and image assembly pipelines with appropriate configurations.

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

          Step(
              func=(IP.create_projection,
                    {'method': 'best_focus',
                     'metric': 'laplacian'}),
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

    from ezstitcher.core.specialized_steps import ZFlatStep

    # Create a maximum intensity projection step
    step = ZFlatStep(
        method="max",  # Options: "max", "mean", "median", "min", "std", "sum"
        input_dir=orchestrator.workspace_path
    )

This step pre-configures:
- ``variable_components=['z_index']``
- ``group_by=None``
- ``func=(IP.create_projection, {'method': method})``

The ``ZFlatStep`` is used by the ``AutoPipelineFactory``:
- Always used in the position generation pipeline to flatten Z-stacks for position generation
- Optionally used in the image assembly pipeline when ``flatten_z=True``

FocusStep
^^^^^^^

The ``FocusStep`` is a specialized step for focus-based Z-stack processing:

.. code-block:: python

    from ezstitcher.core.specialized_steps import FocusStep

    # Create a best focus step
    step = FocusStep(
        focus_options={'metric': 'laplacian'},  # Focus metric options
        input_dir=orchestrator.workspace_path
    )

    # Create a best focus step with custom weights
    step = FocusStep(
        focus_options={'metric': {'nvar': 0.4, 'lap': 0.4, 'ten': 0.1, 'fft': 0.1}},
        input_dir=orchestrator.workspace_path
    )

This step pre-configures:
- ``variable_components=['z_index']``
- ``group_by=None``
- Uses static FocusAnalyzer methods to find the best focus plane

CompositeStep
^^^^^^^^^^

The ``CompositeStep`` is a specialized step for creating composite images from multiple channels:

.. code-block:: python

    from ezstitcher.core.specialized_steps import CompositeStep

    # Create a composite step with custom weights
    step = CompositeStep(
        weights=[0.7, 0.3, 0],  # 70% channel 1, 30% channel 2, 0% channel 3
        input_dir=orchestrator.workspace_path
    )

This step pre-configures:
- ``variable_components=['channel']``
- ``group_by=None``
- ``func=(IP.create_composite, {'weights': weights})``

The ``CompositeStep`` is used by the ``AutoPipelineFactory``:
- Always used in the position generation pipeline to create a reference image for position generation
- If ``channel_weights`` is None, weights are distributed evenly across all channels
- Weights control which channels contribute to the reference image (e.g., [0.7, 0.3, 0] uses only the first two channels)

.. _when-to-use-specialized-steps:

When to Use Specialized Steps
---------------------------

Specialized steps should be used whenever possible for common operations:

1. **ZFlatStep**: Use for Z-stack flattening instead of manually configuring ``variable_components=['z_index']``
2. **FocusStep**: Use for focus detection in Z-stacks
3. **CompositeStep**: Use for channel compositing instead of manually configuring ``variable_components=['channel']``

These specialized steps provide cleaner, more readable code and ensure proper configuration. Use them with minimal parameters unless you need to override defaults.

For channel-specific processing with different functions per channel, using a raw ``Step`` with a dictionary
of functions and ``group_by='channel'`` is the appropriate approach:

.. code-block:: python

    # Channel-specific processing with function dictionary and group_by
    Step(
        name="Channel-Specific Processing",
        func={
            "1": process_dapi,  # Apply process_dapi to channel 1
            "2": process_gfp    # Apply process_gfp to channel 2
        },
        group_by='channel'  # Specifies that keys "1" and "2" refer to channel values
    )

For detailed information about function handling in steps, see :doc:`function_handling`.
For more information about the ``group_by`` parameter, see :ref:`group-by` in :doc:`step`.

**Use orchestrator-specific steps when:**

- You need to generate position files for stitching (``PositionGenerationStep``)
- You need to stitch images using position files (``ImageStitchingStep``)
- You're working with plate-specific operations that leverage the orchestrator

**Use step factories when:**

- You need to perform common operations like Z-stack flattening, focus selection, or channel compositing
- You want to reduce boilerplate code and simplify your pipeline
- You prefer a more intuitive interface for common tasks
- You're building pipelines for non-expert users
- You're creating custom pipelines with standardized components

**Use raw Steps when:**

- You need to perform custom operations not covered by specialized steps
- You need fine-grained control over all parameters
- You're building complex workflows with custom function chains
- You're creating your own specialized steps

As a general rule, start with specialized steps for common operations before falling back to raw Steps. This approach will make your code more concise, readable, and maintainable.

Choosing Between AutoPipelineFactory and Custom Pipelines
--------------------------------------------------------

Both approaches are valid and powerful, with different strengths depending on your needs:

**AutoPipelineFactory Strengths:**

- Convenient for common stitching workflows
- Minimizes code and complexity
- Handles directory resolution automatically
- Configures specialized steps appropriately
- Good for getting started quickly

**Custom Pipeline Strengths:**

- Complete control over pipeline structure
- Flexibility for highly customized workflows
- Direct access to all pipeline features
- Ability to create specialized processing sequences
- Terse and elegant for specific use cases

Many users start with ``AutoPipelineFactory`` for simple tasks and move to custom pipelines as their needs become more specialized, or use a combination of both approaches.

.. _specialized-steps-and-pipeline-factory:

Specialized Steps and AutoPipelineFactory
--------------------------------------

The specialized steps described in this document are used by the :doc:`pipeline_factory` to create pre-configured pipelines for common workflows. The ``AutoPipelineFactory`` creates two pipelines:

1. **Position Generation Pipeline**: Creates position files for stitching
   - Steps: [flatten Z (always), normalize (optional), create_composite (always), generate positions (always)]
   - Uses: ``ZFlatStep``, ``CompositeStep``, and ``PositionGenerationStep``

2. **Image Assembly Pipeline**: Stitches images using the position files
   - Steps: [normalize (optional), flatten Z (optional), stitch_images (always)]
   - Uses: ``ZFlatStep`` or ``FocusStep`` (optional, depending on z_method) and ``ImageStitchingStep``

The factory parameters control which specialized steps are included and how they are configured:

- ``flatten_z``: Controls whether Z-stacks are flattened in the assembly pipeline (Z-stacks are always flattened for position generation)
- ``z_method``: Specifies the Z-stack processing method (default: "max")
  - Projection methods: "max", "mean", "median", etc.
  - Focus detection methods: "combined", "laplacian", "tenengrad", "normalized_variance", "fft"
- ``channel_weights``: Controls which channels contribute to the reference image for position generation

For more information about the ``AutoPipelineFactory``, see :doc:`pipeline_factory`.

.. _specialized-steps-best-practices:

Specialized Step Best Practices
-----------------------------

Here are some key recommendations for using specialized steps:

.. _specialized-steps-directory-resolution:

1. **Directory Resolution**:
   - Let EZStitcher automatically resolve directories when possible
   - Only specify directories when you need a specific directory structure
   - You can explicitly set ``input_dir=orchestrator.workspace_path`` to use original images for stitching

Directory Resolution with AutoPipelineFactory
------------------------------------------

When using ``AutoPipelineFactory``, directory resolution for specialized steps is handled automatically:

- The ``input_dir`` parameter of ``AutoPipelineFactory`` is used as the input directory for the first step in each pipeline
- The ``output_dir`` parameter of ``AutoPipelineFactory`` is used as the output directory for the last step in the assembly pipeline
- Intermediate directories are automatically created and managed
- Position files are automatically passed between the position generation pipeline and the assembly pipeline

This automatic directory resolution makes it much easier to create effective pipelines without having to manually specify input and output directories for each step.

.. code-block:: python

    # AutoPipelineFactory handles directory resolution automatically
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,  # Used as input for first step in each pipeline
        output_dir=Path("path/to/output"),      # Used as output for last step in assembly pipeline
        normalize=True,
        flatten_z=True,
        z_method="max"
    )
    pipelines = factory.create_pipelines()

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

Using Specialized Steps in Custom Pipelines
----------------------------------------

Specialized steps are designed to work seamlessly in custom pipelines. When building custom pipelines, use specialized steps for common operations instead of configuring raw Steps with variable_components:

.. code-block:: python

    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step
    from ezstitcher.core.specialized_steps import ZFlatStep, CompositeStep
    from ezstitcher.core.steps import PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create a custom pipeline with specialized steps
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Normalize images
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 2: Flatten Z-stacks using specialized step
            ZFlatStep(method="max"),

            # Step 3: Create composite for position generation
            CompositeStep(weights=[0.7, 0.3, 0]),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Custom Position Generation Pipeline"
    )

    # Create assembly pipeline
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Normalize images
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 2: Flatten Z-stacks using specialized step
            ZFlatStep(method="max"),

            # Step 3: Stitch images
            ImageStitchingStep()
        ],
        name="Custom Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

This approach provides several benefits:

1. **Readability**: The pipeline structure is explicit and easy to understand
2. **Maintainability**: Changes can be made directly to the pipeline definition
3. **Flexibility**: Complete control over each step and its parameters
4. **Consistency**: Specialized steps ensure consistent behavior for common operations

.. note::
   For common operations, always prefer specialized steps over raw Steps with variable_components:

   - Use ``ZFlatStep`` instead of ``Step`` with ``variable_components=['z_index']``
   - Use ``CompositeStep`` instead of ``Step`` with ``variable_components=['channel']`` for compositing
   - Use ``FocusStep`` instead of manually implementing focus detection

.. seealso::
   - :doc:`pipeline` for more information about creating custom pipelines
   - :doc:`pipeline_factory` for information about when to use factory vs. custom pipelines

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

The simplest approach is to use the ``AutoPipelineFactory``:

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Create orchestrator
    orchestrator = PipelineOrchestrator(plate_path=plate_path)

    # Create a factory with default settings
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True  # Apply normalization (default)
    )

    # Create the pipelines
    pipelines = factory.create_pipelines()

    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

Alternatively, you can build custom pipelines using specialized steps:

.. code-block:: python

    from ezstitcher.core.steps import PositionGenerationStep, ImageStitchingStep, Step
    from ezstitcher.core.specialized_steps import ZFlatStep, CompositeStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create position generation pipeline
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Process images (optional)
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Create composite for position generation
            CompositeStep(),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    positions_dir = position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched",
        steps=[
            # Step 1: Flatten Z-stacks (if working with Z-stacks)
            ZFlatStep(method="max"),

            # Step 2: Process images (optional)
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize
            ),

            # Step 3: Stitch images using position files
            ImageStitchingStep(positions_dir=positions_dir)
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

Multi-Channel Stitching
^^^^^^^^^^^^^^^^^^^^

When working with multiple channels, it's important to create a composite image before position generation. The simplest approach is to use the ``AutoPipelineFactory`` with channel weights:

.. code-block:: python

    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

    # Create orchestrator
    orchestrator = PipelineOrchestrator(plate_path=plate_path)

    # Create a factory for multi-channel stitching
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True,
        channel_weights=[0.7, 0.3, 0]  # Use only first two channels for reference image
    )

    # Create the pipelines
    pipelines = factory.create_pipelines()

    # Run the pipelines
    orchestrator.run(pipelines=pipelines)

Alternatively, you can build custom pipelines using specialized steps:

.. code-block:: python

    from ezstitcher.core.steps import PositionGenerationStep, ImageStitchingStep, Step
    from ezstitcher.core.specialized_steps import ZFlatStep, CompositeStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create position generation pipeline for multi-channel data
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Process channels
            Step(
                name="Normalize Channels",
                func=IP.stack_percentile_normalize,
                variable_components=['channel']
            ),

            # Step 3: Create composite image for position generation
            CompositeStep(weights=[0.7, 0.3, 0]),  # 70% channel 1, 30% channel 2, 0% channel 3

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    positions_dir = position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline for multi-channel data
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched",
        steps=[
            # Step 1: Process channels
            Step(
                name="Normalize Channels",
                func=IP.stack_percentile_normalize,
                variable_components=['channel']
            ),

            # Step 2: Stitch images using position files
            ImageStitchingStep(
                positions_dir=positions_dir,
                variable_components=['channel']  # Stitch each channel separately
            )
        ],
        name="Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

Using Original Images for Stitching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes you want to process images for position generation but use the original images for stitching. The recommended approach is to build custom pipelines that explicitly specify this behavior:

.. code-block:: python

    from ezstitcher.core.steps import PositionGenerationStep, ImageStitchingStep, Step
    from ezstitcher.core.specialized_steps import ZFlatStep, CompositeStep
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create position generation pipeline with processed images
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        steps=[
            # Step 1: Flatten Z-stacks (always included for position generation)
            ZFlatStep(method="max"),

            # Step 2: Process images for position generation
            Step(
                name="Normalize Images",
                func=IP.stack_percentile_normalize,
                output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_processed"
            ),

            # Step 3: Create composite for position generation
            CompositeStep(),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Get the position files directory
    positions_dir = position_pipeline.steps[-1].output_dir

    # Create image assembly pipeline using original images
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,  # Use original images for stitching
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched",
        steps=[
            # Stitch using original images
            ImageStitchingStep(positions_dir=positions_dir)
        ],
        name="Original Image Assembly Pipeline"
    )

    # Run the pipelines
    orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

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
