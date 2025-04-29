=================
Intermediate Usage
=================

.. note::
   **Complexity Level: Intermediate**

   This section is designed for users who need more control with wrapped steps (NormStep, ZFlatStep, etc.).

This section introduces custom pipelines with wrapped steps (NormStep, ZFlatStep, CompositeStep, etc.) for users who need more control than the EZ module provides. It covers:

1. Creating custom pipelines with wrapped steps
2. Processing Z-stacks and multi-channel plates
3. Combining Z-flattening, focus selection, and channel-specific processing

**Learning Path:**

1. If you are new to EZStitcher, start with the :doc:`ez_module` guide (beginner level)
2. Then read the :doc:`transitioning_from_ez` guide to understand how to bridge the gap between the EZ module and custom pipelines
3. Now you're ready for this intermediate usage guide with wrapped steps
4. For advanced usage with the base Step class, see :doc:`advanced_usage`

.. note::
   EZStitcher automatically chains *input_dir* / *output_dir* between steps.
   See :doc:`../concepts/directory_structure` for details on how directories are managed.

.. important::
   The interplay between ``variable_components`` and ``group_by`` controls how loops over Z-index or channel are executed.
   See :doc:`../concepts/step` and :doc:`../concepts/function_handling` for detailed explanations.

--------------------------------------------------------------------
Creating Custom Pipelines with Wrapped Steps
--------------------------------------------------------------------

Custom pipelines provide more control and flexibility than the EZ module. Here's a basic example of creating custom pipelines with wrapped steps:

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

This approach gives you more control over the processing steps while still using wrapped steps that provide a clean interface for common operations.

--------------------------------------------------------------------
Z-stack processing with the EZ module
--------------------------------------------------------------------

.. code-block:: python

   from pathlib import Path
   from ezstitcher import stitch_plate

   plate_path = Path("~/data/PlateA")  # <-- edit me

   # Process Z-stacks with maximum intensity projection
   stitch_plate(
       plate_path,
       flatten_z=True,
       z_method="max"          # "mean", "median", "laplacian", "combined", ...
   )

For more control, use custom pipelines:

.. code-block:: python

   from pathlib import Path
   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
   from ezstitcher.core.pipeline import Pipeline
   from ezstitcher.core.steps import NormStep, PositionGenerationStep, ImageStitchingStep, ZFlatStep, CompositeStep

   plate_path = Path("~/data/PlateA")  # <-- edit me
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
           ZFlatStep(method="max"),  # Z-stack flattening
           ImageStitchingStep(positions_dir=positions_dir),  # Image stitching
       ],
       name="Assembly",
   )

   orchestrator.run(pipelines=[pos_pipe, asm_pipe])

--------------------------------------------------------------------
Custom position-generation + assembly pipelines
--------------------------------------------------------------------

Below we flatten Z by **max projection** for position finding, then
assemble the final mosaic with **best-focus** selection.

.. code-block:: python

   from pathlib import Path
   from ezstitcher.core.pipeline import Pipeline
   from ezstitcher.core.steps import NormStep, PositionGenerationStep, ImageStitchingStep, ZFlatStep, FocusStep

   # --- reusable position pipeline ---------------------------------
   position_pipeline = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(method="max"),  # Z-stack flattening
           NormStep(),  # Normalization
           PositionGenerationStep()  # Position generation
       ],
       name="Position Generation"
   )
   positions_dir = position_pipeline.steps[-1].output_dir

   # --- assembly pipeline with focus selection --------------------
   assembly_pipeline = Pipeline(
       input_dir=orchestrator.workspace_path,
       output_dir=Path("out/best_focus"),
       steps=[
           FocusStep(focus_options={"metric": "variance_of_laplacian"}),  # Focus selection
           NormStep(),  # Normalization
           ImageStitchingStep(positions_dir=positions_dir)  # Image stitching
       ],
       name="Assembly (best focus)"
   )

   orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

--------------------------------------------------------------------
When to choose which approach
--------------------------------------------------------------------

* **Use the EZ module** for standard plates or slides when you want minimal code and default settings are sufficient.

* **Write custom pipelines** when you need bespoke steps, per-channel logic, or multiple outputs (e.g. max-projection + best-focus).

* For more information on the three-tier approach and when to use each approach, see the :ref:`three-tier-approach` section in the introduction.

Next up: :doc:`advanced_usage`.

