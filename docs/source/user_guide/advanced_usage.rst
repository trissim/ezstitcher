==============
Advanced Usage
==============

This page shows **three advanced skills**:

1. Write *custom processing functions* and wire them into pipelines.
2. Enable **multithreaded** execution for large plates.
3. Implement advanced functional patterns for complex workflows.

If you are new to the library, first read :doc:`basic_usage` and :doc:`intermediate_usage`.

.. note::
   Use specialised steps (``ZFlatStep``, ``CompositeStep``, ``FocusStep``) whenever you need to loop over *z-index* or *channel*.  A raw :class:`~ezstitcher.core.steps.Step` is only required when no specialised variant exists.

.. important::
   The interplay between ``group_by`` and ``variable_components`` controls **how your function loops**.  Review :doc:`../concepts/step` before writing advanced pipelines.

---------------------------------------------------------------------
1. Creating custom processing functions
---------------------------------------------------------------------

Custom functions receive **a list of NumPy arrays** (images) and must return the *same‑length* list.

.. code-block:: python

   import numpy as np
   from skimage import filters

   def custom_enhance(images, sigma=1.0, contrast=1.5):
       """Gaussian blur + contrast stretch."""
       out = []
       for im in images:
           blurred = filters.gaussian(im, sigma=sigma)
           mean    = blurred.mean()
           out.append(np.clip(mean + contrast * (blurred - mean), 0, 1))
       return out

---------------------------------------------------------------------
2. Building an advanced custom pipeline
---------------------------------------------------------------------

Below we denoise, normalise, enhance and then stitch — all with **two concise pipelines**.

.. code-block:: python

   from pathlib import Path

   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
   from ezstitcher.core.pipeline           import Pipeline
   from ezstitcher.core.steps              import Step, PositionGenerationStep, ImageStitchingStep
   from ezstitcher.core.specialized_steps  import ZFlatStep, CompositeStep
   from ezstitcher.core.image_processor    import ImageProcessor as IP

   # ---------- orchestrator ----------------------------------------
   plate_path   = Path("~/data/PlateA").expanduser()
   orchestrator = PipelineOrchestrator(plate_path)

   # ---------- helper functions -----------------------------------
   def denoise(images, strength=0.5):
       from skimage.restoration import denoise_nl_means
       return [denoise_nl_means(im, h=strength) for im in images]

   # ---------- position pipeline ----------------------------------
   pos_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(method="max"),
           Step(func=(denoise, {"strength": 0.4})),
           Step(func=IP.stack_percentile_normalize),
           CompositeStep(),
           PositionGenerationStep(),
       ],
       name="Position Generation",
   )
   positions_dir = pos_pipe.steps[-1].output_dir

   # ---------- assembly pipeline ----------------------------------
   asm_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       output_dir=Path("out/stitched"),
       steps=[
           Step(func=(denoise, {"strength": 0.4})),
           Step(func=IP.stack_percentile_normalize),
           ImageStitchingStep(positions_dir=positions_dir),
       ],
       name="Assembly",
   )

   orchestrator.run(pipelines=[pos_pipe, asm_pipe])

---------------------------------------------------------------------
3. Channel‑aware processing with ``group_by='channel'``
---------------------------------------------------------------------

.. code-block:: python

   def process_dapi(images):
       return IP.stack_percentile_normalize([IP.tophat(im, size=15) for im in images])

   def process_gfp(images):
       return IP.stack_percentile_normalize([IP.sharpen(im, sigma=1.0, amount=1.5) for im in images])

   channel_step = Step(func={"1": process_dapi, "2": process_gfp}, group_by="channel")

---------------------------------------------------------------------
4. Conditional processing based on context
---------------------------------------------------------------------

The *context* dict is passed to every Step when ``pass_context=True``.

.. code-block:: python

   def conditional(images, context):
       if context["well"] == "A01":
           return process_control(images)
       return process_treatment(images)

   cond_step = Step(func=conditional, pass_context=True)

---------------------------------------------------------------------
5. Multithreading for large plates
---------------------------------------------------------------------

.. code-block:: python

   from ezstitcher.core.config import PipelineConfig

   cfg = PipelineConfig(num_workers=4)  # use 4 threads
   orchestrator = PipelineOrchestrator(plate_path, config=cfg)
   orchestrator.run(pipelines=[pos_pipe, asm_pipe])

Threads are allocated **per well**; inside a well, steps run sequentially.
Adjust `num_workers` to avoid memory exhaustion.

---------------------------------------------------------------------
6. Advanced Functional Patterns
---------------------------------------------------------------------

Create powerful processing pipelines without extending core classes:

.. code-block:: python

   from pathlib import Path
   from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
   from ezstitcher.core.pipeline import Pipeline
   from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
   from ezstitcher.core.specialized_steps import ZFlatStep, CompositeStep
   from ezstitcher.core.image_processor import ImageProcessor as IP

   # ---------- orchestrator ----------------------------------------
   plate_path   = Path("~/data/PlateA").expanduser()
   orchestrator = PipelineOrchestrator(plate_path)

   # ---------- position pipeline ----------------------------------
   pos_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(method="max"),
           Step(func=IP.stack_percentile_normalize),
           CompositeStep(),
           Step(func=custom_enhance),  # Custom processing
           PositionGenerationStep(),
       ],
       name="Position Generation",
   )
   positions_dir = pos_pipe.steps[-1].output_dir

   # ---------- assembly pipeline ----------------------------------
   asm_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           Step(func=IP.stack_percentile_normalize),
           ImageStitchingStep(positions_dir=positions_dir),
       ],
       name="Assembly",
   )

   # ---------- analysis pipeline ---------------------------------
   # Add a third pipeline for post-processing analysis
   analysis_pipe = Pipeline(
       input_dir=asm_pipe.output_dir,  # Use output from assembly
       steps=[
           Step(func=analyze_histograms),  # Custom analysis
       ],
       name="Analysis",
   )

   # ---------- run all pipelines ---------------------------------
   orchestrator.run(pipelines=[pos_pipe, asm_pipe, analysis_pipe])

   # ---------- analysis function ---------------------------------
   def analyze_histograms(images):
       from skimage.exposure import histogram
       return [histogram(im)[0] for im in images]

---------------------------------------------------------------------
7. Adding a new microscope handler
---------------------------------------------------------------------

Implement :class:`~ezstitcher.core.microscope_handler.BaseMicroscopeHandler` and register it via ``register_handler``.
See :doc:`../development/extending` for the full walkthrough.

---------------------------------------------------------------------
Choosing the right tool
---------------------------------------------------------------------

* **EZ module** → quick wins with minimal code for standard plates.
* **Custom pipelines** → full control for research prototypes and advanced workflows.
* **Custom handlers** → organisation‑wide automation (for contributors).


Next steps
~~~~~~~~~~

* Read the :doc:`integration` guide for napari and CellProfiler hooks.
* Follow the "learning path" outline in :ref:`learning-path` to master EZStitcher.


