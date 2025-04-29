=================
Intermediate Usage
=================

This section shows **three intermediate-level skills**:

1. Process Z-stacks and multi-channel plates with the EZ module or custom pipelines.
2. Build **custom pipelines** when you need precise control.
3. Combine Z-flattening, focus selection and channel-specific processing.

If you are new to EZStitcher, start with :doc:`basic_usage` first.

.. note::

   EZStitcher automatically chains *input_dir* / *output_dir* between steps.
   See :doc:`../concepts/directory_structure` for details on how directories are managed.

.. important::

   The interplay between ``variable_components`` and ``group_by`` controls how loops over Z-index or channel are executed.
   See :doc:`../concepts/step` and :doc:`../concepts/function_handling` for detailed explanations.

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
   from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep, ZFlatStep, CompositeStep
   from ezstitcher.core.image_processor import ImageProcessor as IP

   plate_path = Path("~/data/PlateA")  # <-- edit me
   orchestrator = PipelineOrchestrator(plate_path)

   # Position generation pipeline
   pos_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(method="max"),  # Z-stack flattening
           Step(func=IP.stack_percentile_normalize),
           CompositeStep(),
           PositionGenerationStep(),
       ],
       name="Position Generation",
   )
   positions_dir = pos_pipe.steps[-1].output_dir

   # Assembly pipeline
   asm_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       output_dir=plate_path.parent / f"{plate_path.name}_stitched",
       steps=[
           Step(func=IP.stack_percentile_normalize),
           ZFlatStep(method="max"),  # Z-stack flattening
           ImageStitchingStep(positions_dir=positions_dir),
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
   from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep, ZFlatStep, FocusStep
   from ezstitcher.core.image_processor import ImageProcessor as IP

   # --- reusable position pipeline ---------------------------------
   position_pipeline = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(method="max"),
           Step(func=IP.stack_percentile_normalize),
           PositionGenerationStep()
       ],
       name="Position Generation"
   )
   positions_dir = position_pipeline.steps[-1].output_dir

   # --- assembly pipeline with focus selection --------------------
   assembly_pipeline = Pipeline(
       input_dir=orchestrator.workspace_path,
       output_dir=Path("out/best_focus"),
       steps=[
           FocusStep(focus_options={"metric": "variance_of_laplacian"}),
           Step(func=IP.stack_percentile_normalize),
           ImageStitchingStep(positions_dir=positions_dir)
       ],
       name="Assembly (best focus)"
   )

   orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

--------------------------------------------------------------------
Channel-specific processing via ``Step(group_by='channel')``
--------------------------------------------------------------------

.. code-block:: python

   def process_dapi(images):
       return IP.stack_percentile_normalize([IP.tophat(i, size=15) for i in images])

   def process_gfp(images):
       return IP.stack_percentile_normalize([IP.sharpen(i, sigma=1.0, amount=1.5) for i in images])

   channel_proc = Step(
       func={"1": process_dapi, "2": process_gfp},
       group_by="channel"
   )

   position_pipeline = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(),
           channel_proc,
           PositionGenerationStep()
       ],
       name="Position Generation (per-channel)"
   )

   assembly_pipeline = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           channel_proc,
           ImageStitchingStep(positions_dir=position_pipeline.steps[-1].output_dir)
       ],
       name="Assembly (per-channel)"
   )

--------------------------------------------------------------------
When to choose which approach
--------------------------------------------------------------------

* **Use the EZ module** for standard plates or slides when you want minimal code and default settings are sufficient.

* **Write custom pipelines** when you need bespoke steps, per-channel logic, or multiple outputs (e.g. max-projection + best-focus).

* For more information on the three-tier approach and when to use each approach, see the :ref:`three-tier-approach` section in the introduction.

Next up: :doc:`advanced_usage`.

