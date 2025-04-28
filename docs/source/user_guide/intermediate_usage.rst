=================
Intermediate Usage
=================

This section shows **three intermediate-level skills**:

1. Configure ``AutoPipelineFactory`` for Z-stacks and multi-channel plates.
2. Build **custom pipelines** when you need precise control.
3. Combine Z-flattening, focus selection and channel-specific processing.

If you are new to EZStitcher, start with :doc:`basic_usage` first.

.. note::

   EZStitcher automatically chains *input_dir* / *output_dir*
   between steps.  Only the **first** step must receive
   ``input_dir=orchestrator.workspace_path``; the rest inherit paths.
   See :doc:`../concepts/directory_structure` for details.

.. important::

   The interplay between ``variable_components`` and ``group_by``
   controls how loops over Z-index or channel are executed.
   Review :doc:`../concepts/step` before writing advanced pipelines.

--------------------------------------------------------------------
Z-stack processing with ``AutoPipelineFactory``
--------------------------------------------------------------------

.. code-block:: python

   from pathlib import Path
   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
   from ezstitcher.factories import AutoPipelineFactory

   plate_path = Path("~/data/PlateA")  # <-- edit me
   orchestrator = PipelineOrchestrator(plate_path)

   factory = AutoPipelineFactory(
       input_dir=orchestrator.workspace_path,
       normalize=True,
       flatten_z=True,
       z_method="max"          # "mean", "median", "laplacian", "combined", ...
   )
   pipelines = factory.create_pipelines()
   orchestrator.run(pipelines=pipelines)

--------------------------------------------------------------------
Custom position-generation + assembly pipelines
--------------------------------------------------------------------

Below we flatten Z by **max projection** for position finding, then
assemble the final mosaic with **best-focus** selection.

.. code-block:: python

   from pathlib import Path
   from ezstitcher.core.pipeline import Pipeline
   from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
   from ezstitcher.core.specialized_steps import ZFlatStep, FocusStep
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

* **Use ``AutoPipelineFactory``** for standard plates or slides when
  you only need to toggle *normalize*, *flatten_z* or *z_method*.

* **Write custom pipelines** when you need bespoke steps, per-channel
  logic, or multiple outputs (e.g. max-projection + best-focus).

Next up: :doc:`advanced_usage`.

