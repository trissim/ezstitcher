===========================
Transitioning from EZ Module
===========================

.. note::
   **Complexity Level: Beginner to Intermediate**

   This section is designed for users who are familiar with the EZ module and want to gain more control over their image processing workflows.

The EZ module provides a simplified interface for common stitching workflows, but as your needs become more specialized, you may want more control over the processing steps. This section helps you bridge the gap between the EZ module and custom pipelines.

Understanding the EZ Module Under the Hood
-----------------------------------------

The EZ module is built on top of the pipeline architecture. When you call ``stitch_plate()``, it creates pipelines and steps behind the scenes:

.. code-block:: python

   from ezstitcher import stitch_plate

   # This simple call...
   stitch_plate("path/to/plate")

   # ...creates pipelines and steps similar to this:
   # 1. Position Generation Pipeline with:
   #    - ZFlatStep (if Z-stacks are detected)
   #    - NormStep (for normalization)
   #    - CompositeStep (for channel compositing)
   #    - PositionGenerationStep
   #
   # 2. Assembly Pipeline with:
   #    - NormStep (for normalization)
   #    - ImageStitchingStep

By understanding this structure, you can create custom pipelines that provide more control while still leveraging the power of wrapped steps.

From EZ Module to Wrapped Steps
------------------------------

The first step in transitioning from the EZ module is to use wrapped steps (NormStep, ZFlatStep, CompositeStep, etc.) instead of the EZ module's one-liner approach:

.. code-block:: python

   from pathlib import Path
   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
   from ezstitcher.core.pipeline import Pipeline
   from ezstitcher.core.steps import NormStep, ZFlatStep, CompositeStep, PositionGenerationStep, ImageStitchingStep

   plate_path = Path("~/data/PlateA").expanduser()
   orchestrator = PipelineOrchestrator(plate_path)

   # Position generation pipeline
   pos_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(method="max"),  # Z-stack flattening
           NormStep(),  # Normalization
           CompositeStep(),  # Channel compositing
           PositionGenerationStep(),  # Position generation
       ],
       name="Position Generation",
   )
   positions_dir = pos_pipe.steps[-1].output_dir

   # Assembly pipeline
   asm_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       output_dir=plate_path.parent / f"{plate_path.name}_stitched",
       steps=[
           NormStep(),  # Normalization
           ImageStitchingStep(positions_dir=positions_dir),  # Image stitching
       ],
       name="Assembly",
   )

   orchestrator.run(pipelines=[pos_pipe, asm_pipe])

This approach gives you more control over the processing steps while still using the wrapped steps that provide a clean interface for common operations.

Wrapped Steps vs. Low-Level Step Class
------------------------------------

EZStitcher provides wrapped steps (NormStep, ZFlatStep, CompositeStep, etc.) that encapsulate common operations with a clean interface. These wrapped steps are built on top of the base Step class, but provide a more user-friendly interface:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Wrapped Step
     - Equivalent Low-Level Step
   * - ``NormStep()``
     - ``Step(func=IP.stack_percentile_normalize)``
   * - ``ZFlatStep(method="max")``
     - ``Step(func=IP.z_project_max, variable_components=["z"])``
   * - ``CompositeStep()``
     - ``Step(func=IP.composite_channels, variable_components=["channel"])``
   * - ``FocusStep()``
     - ``Step(func=IP.select_focus_z, variable_components=["z"])``

Using wrapped steps makes your code more readable and less error-prone, while still providing the flexibility you need.

Customizing Wrapped Steps
-----------------------

Wrapped steps can be customized with parameters to control their behavior:

.. code-block:: python

   # Customize Z-flattening
   ZFlatStep(method="focus", focus_options={"metric": "variance_of_laplacian"})

   # Customize normalization
   NormStep(percentile=95)

   # Customize channel compositing
   CompositeStep(weights=[0.7, 0.3, 0])

This allows you to fine-tune the processing while still using the clean interface provided by wrapped steps.

When to Move to Intermediate Usage
--------------------------------

Consider moving to the intermediate usage level when:

* You need more control over the processing steps
* You want to apply different processing to different channels
* You need to customize the Z-flattening method
* You want to create multiple output types (e.g., max projection and best focus)
* The EZ module doesn't provide the flexibility you need

The intermediate usage level provides more control while still using wrapped steps (NormStep, ZFlatStep, etc.) that encapsulate common operations with a clean interface.

Next Steps
---------

* For beginners who want to stay with the EZ module, return to :doc:`basic_usage` for more examples
* For intermediate users ready to create custom pipelines, proceed to :doc:`intermediate_usage`
* For detailed information about pipeline configuration, see :doc:`../concepts/pipeline`
* For detailed information about step configuration, see :doc:`../concepts/step`
* For best practices at all levels, see :doc:`best_practices`
