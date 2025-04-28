.. _best-practices:

===============================================
Best Practices
===============================================

This page distills day-to-day advice into **five short check-lists**. Each list links to the concept doc where details live—so this page stays light.

Sections:

1. :ref:`bp-ez-module`  - EZ module cheatsheet
2. :ref:`bp-manual`   - manual-pipeline dos & don'ts
3. :ref:`bp-dir`      - directory hygiene
4. :ref:`bp-steps`    - specialised step order
5. :ref:`bp-func`     - function-handling patterns

.. _bp-ez-module:

----------------------------------------
EZ Module in one minute
----------------------------------------

*Use for quick results with minimal code.* Only four parameters matter 90% of the time:

+ ``normalize``              → percentile stretch (1/99 by default)
+ ``flatten_z`` + ``z_method`` → convert stacks ("max", "mean", "combined" …)
+ ``channel_weights``        → which channels build reference composite

.. code-block:: python

   from ezstitcher import stitch_plate

   # Basic usage
   stitch_plate("path/to/plate")

   # With options
   stitch_plate(
       "path/to/plate",
       normalize=True,
       flatten_z=True,
       z_method="max",          # or "combined" for focus metric
       channel_weights=[0.7, 0.3, 0]
   )

*Rule of thumb* → use the EZ module for standard workflows; create custom pipelines when you need more control.

.. _bp-manual:

----------------------------------------
Manual pipeline checklist
----------------------------------------

✔ Start with **ZFlatStep → Normalize → Composite → Position → Stitch** and add/remove steps from there. See :doc:`../concepts/specialized_steps` for details on specialized steps.

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
* For detailed information on directory handling, see :doc:`../concepts/directory_structure`.

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

* For detailed information on function handling patterns, see :doc:`../concepts/function_handling`.


.. _bp-custom-pipelines:

----------------------------------------
Custom Pipeline Best Practices
----------------------------------------

When creating custom pipelines:

1. **Use specialized steps for common operations**:
   - ``ZFlatStep`` for Z-stack flattening
   - ``CompositeStep`` for channel compositing
   - ``PositionGenerationStep`` and ``ImageStitchingStep`` for stitching

2. **Leverage functional programming patterns**:
   - Use the ``func`` parameter to pass processing functions
   - Compose complex operations with multiple steps
   - Use ``variable_components`` and ``group_by`` for fine-grained control

3. **Follow a consistent pipeline structure**:
   - Position generation pipeline: process → composite → generate positions
   - Assembly pipeline: process → stitch
   - Analysis pipeline (optional): analyze stitched images

Example of a well-structured custom pipeline:

.. code-block:: python

   # Position generation pipeline
   pos_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(method="max"),
           Step(func=IP.stack_percentile_normalize),
           CompositeStep(weights=[0.7, 0.3, 0]),
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
           ZFlatStep(method="max"),
           ImageStitchingStep(positions_dir=positions_dir),
       ],
       name="Assembly",
   )

--------------------------------------------------------------------
Need more depth?
--------------------------------------------------------------------

* :doc:`../concepts/pipeline`
* :doc:`../concepts/directory_structure`
* :doc:`../concepts/specialized_steps`
* :doc:`../concepts/function_handling`
