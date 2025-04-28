===========
Basic Usage
===========

This page shows the **minimal, repeatable recipe** for stitching a
multi‑well plate with EZStitcher.  If you only need a 10‑line working
example, see :doc:`../getting_started/quick_start`.

Two workflows are presented:

1. **AutoPipelineFactory** – quickest path; no code beyond parameters.
2. **Custom pipelines** – explicit steps for full control.

--------------------------------------------------------------------
1. One‑liner stitching with ``AutoPipelineFactory``
--------------------------------------------------------------------

.. code-block:: python

   from pathlib import Path
   from ezstitcher.factories import AutoPipelineFactory
   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

   plate_path = Path("~/data/PlateA").expanduser()
   orchestrator = PipelineOrchestrator(plate_path)

   factory = AutoPipelineFactory(
       input_dir=orchestrator.workspace_path,
       output_dir=plate_path.parent / f"{plate_path.name}_stitched",
       normalize=True,          # percentile normalisation 1–99
       flatten_z=True,          # Z‑stack → 2‑D max‑projection
       z_method="max",         # max, mean, laplacian, combined …
   )

   orchestrator.run(pipelines=factory.create_pipelines())

That single call produces two pipelines:

* **position** – generates stage‑coordinate CSVs per well.
* **assembly** – stitches tiles using Ashlar; writes an OME‑TIFF per
  channel (or per well for plates).

--------------------------------------------------------------------
Frequently‑tweaked factory knobs
--------------------------------------------------------------------

* ``channel_weights=[0.7, 0.3, 0]`` – choose which channels build the
  reference composite.
* ``z_method="combined"`` – use a focus metric instead of projection.
* ``normalization_params={'low_percentile':0.5, 'high_percentile':99.5}``
  to fine‑tune contrast stretch.

--------------------------------------------------------------------
2. Explicit pipelines (full control)
--------------------------------------------------------------------

.. code-block:: python

   from pathlib import Path
   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
   from ezstitcher.core.pipeline           import Pipeline
   from ezstitcher.core.steps              import Step, PositionGenerationStep, ImageStitchingStep
   from ezstitcher.core.specialized_steps  import ZFlatStep, CompositeStep
   from ezstitcher.core.image_processor    import ImageProcessor as IP

   plate_path   = Path("~/data/PlateA").expanduser()
   orchestrator = PipelineOrchestrator(plate_path)

   # ----- position pipeline --------------------------------------
   pos_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(method="max"),
           Step(func=IP.stack_percentile_normalize),
           CompositeStep(),
           PositionGenerationStep(),
       ],
       name="Position Generation",
   )
   positions_dir = pos_pipe.steps[-1].output_dir

   # ----- assembly pipeline --------------------------------------
   asm_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       output_dir=plate_path.parent / f"{plate_path.name}_stitched",
       steps=[
           Step(func=IP.stack_percentile_normalize),
           ImageStitchingStep(positions_dir=positions_dir),
       ],
       name="Assembly",
   )

   orchestrator.run(pipelines=[pos_pipe, asm_pipe])

--------------------------------------------------------------------
Which approach should I pick?
--------------------------------------------------------------------

| Use **AutoPipelineFactory** when… | Use **custom pipelines** when… |
|----------------------------------|--------------------------------|
| • default steps are enough        | • need bespoke processing      |
| • quick turnaround / notebook     | • want per‑channel logic       |
| • prototyping / demo              | • desire full transparency     |

--------------------------------------------------------------------
Next steps
--------------------------------------------------------------------

* Proceed to :doc:`intermediate_usage` for channel‑specific and Z‑stack tricks.
* Deep‑dive into :doc:`../concepts/pipeline` to learn every
  parameter and how directories resolve automatically.

