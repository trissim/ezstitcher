===========
Basic Usage
===========

This page shows the **minimal, repeatable recipe** for stitching a
multi‑well plate with EZStitcher.  If you only need a 10‑line working
example, see :doc:`../getting_started/quick_start`.

.. note::
   For a simplified interface with minimal code, see the :doc:`ez_module` guide.
   The EZ module is recommended for most users, especially beginners.

This guide focuses on creating custom pipelines for maximum flexibility and control, as described in the :ref:`three-tier-approach` section of the introduction.

.. note::
   For information about the underlying architecture, see :doc:`../concepts/architecture_overview`.



--------------------------------------------------------------------
Creating Custom Pipelines
--------------------------------------------------------------------

.. code-block:: python

   from pathlib import Path
   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
   from ezstitcher.core.pipeline           import Pipeline
   from ezstitcher.core.steps              import Step, PositionGenerationStep, ImageStitchingStep, ZFlatStep, CompositeStep
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
Next steps
--------------------------------------------------------------------

* For more information on the three-tier approach and when to use each approach, see the :ref:`three-tier-approach` section in the introduction.
* For detailed information about directory structure, see :doc:`../concepts/directory_structure`.
* For detailed information about step configuration, see :doc:`../concepts/step`.
* Proceed to :doc:`intermediate_usage` for channel‑specific and Z‑stack tricks.
* Deep‑dive into :doc:`../concepts/pipeline` to learn every
  parameter and how directories resolve automatically.

