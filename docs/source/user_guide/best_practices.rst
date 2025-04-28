.. _best-practices:

===============================================
Best Practices
===============================================

This page distills day-to-day advice into **fixe short check-lists**. Each list links to the concept doc where details live—so this page stays light.

Sections:

1. :ref:`bp-factory`  - AutoPipelineFactory cheatsheet
2. :ref:`bp-manual`   - manual-pipeline dos & don'ts
3. :ref:`bp-dir`      - directory hygiene
4. :ref:`bp-steps`    - specialised step order
5. :ref:`bp-func`     - function-handling patterns

.. _bp-factory:

----------------------------------------
AutoPipelineFactory in one minute
----------------------------------------

*Use when built-in steps are enough.*  Only four knobs matter 90% of the time:

+ ``normalize``              → percentile stretch (1/99 by default)
+ ``flatten_z`` + ``z_method`` → convert stacks ("max", "mean", "combined" …)
+ ``channel_weights``        → which channels build reference composite

.. code-block:: python

   factory = AutoPipelineFactory(
       input_dir=orchestrator.workspace_path,
       normalize=True,
       flatten_z=True,
       z_method="max",          # or "combined" for focus metric
       channel_weights=[0.7, 0.3, 0],
   )
   orchestrator.run(pipelines=factory.create_pipelines())

*Rule of thumb*  → don't edit the generated pipelines; write your own if you need a custom step.

.. _bp-manual:

----------------------------------------
Manual pipeline checklist
----------------------------------------

✔ Start with **ZFlatStep → Normalize → Composite → Position → Stitch** and add/remove steps from there.

✔ Wrap repeated code in a *factory function* so notebooks stay clean.

✔ Name your pipeline (`name="Plate A - Max proj"`)—the logger shows it.

✘ Avoid inserting steps **after** `ImageStitchingStep`; do post-analysis in a separate pipeline or script.

.. _bp-dir:

----------------------------------------
Directory hygiene
----------------------------------------

* First step → `input_dir=orchestrator.workspace_path`.
* Omit `output_dir` unless you truly need it; EZStitcher auto‑chains.
* Use `pipeline.output_dir` when another script needs the results.

.. _bp-steps:

----------------------------------------
Specialised step order (golden path)
----------------------------------------

1. **ZFlatStep / FocusStep**  - reduce stacks.
2. **Channel processing + CompositeStep** - build reference image.
3. **PositionGenerationStep** - writes CSV.
4. **ImageStitchingStep**     - uses CSV.

Anything else is an optimisation *before* or *between* 1-2.

.. _bp-func:

----------------------------------------
Function-handling patterns
----------------------------------------

> Always "stack-in / stack-out"—each function receives a list of images and returns a list of the **same length**.

| Pattern     | Example                                                       |
|-------------|---------------------------------------------------------------|
| Single fn   | `Step(func=IP.stack_percentile_normalize)`                    |
| Fn + kwargs | `Step(func=(IP.tophat, {'size':15}))`                         |
| Chain       | `Step(func=[(IP.tophat,{'size':15}), IP.stack_percentile_normalize])` |
| Per-channel | `Step(func={'1': proc_dapi, '2': proc_gfp}, group_by='channel')` |


--------------------------------------------------------------------
Need more depth?
--------------------------------------------------------------------

* :doc:`../concepts/pipeline_factory`
* :doc:`../concepts/pipeline`
* :doc:`../concepts/directory_structure`
* :doc:`../concepts/specialized_steps`
* :doc:`../concepts/function_handling`