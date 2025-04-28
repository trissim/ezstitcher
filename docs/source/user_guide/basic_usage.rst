===========
Basic Usage
===========

This page shows the **minimal, repeatable recipe** for stitching a
multi‑well plate with EZStitcher.  If you only need a 10‑line working
example, see :doc:`../getting_started/quick_start`.

.. note::
   For a simplified interface with minimal code, see the :doc:`ez_module` guide.
   The EZ module is recommended for most users, especially beginners.

This guide focuses on creating custom pipelines for maximum flexibility and control.

.. note::
   While AutoPipelineFactory is used internally by the EZ module, it is generally
   not recommended for direct use by end users. For most use cases, either the
   EZ module (for simplicity) or custom pipelines (for flexibility) are preferred.



--------------------------------------------------------------------
Creating Custom Pipelines
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

| Use **EZ Module** when… | Use **custom pipelines** when… |
|------------------------|--------------------------------|
| • You want minimal code | • You need bespoke processing  |
| • You're new to EZStitcher | • You want per‑channel logic |
| • Default settings are sufficient | • You need maximum flexibility |
| • You want auto-detection | • You want full transparency |

--------------------------------------------------------------------
Next steps
--------------------------------------------------------------------

* Proceed to :doc:`intermediate_usage` for channel‑specific and Z‑stack tricks.
* Deep‑dive into :doc:`../concepts/pipeline` to learn every
  parameter and how directories resolve automatically.

