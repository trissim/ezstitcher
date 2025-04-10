# Plan: Investigate Nested Config Overrides Applied to PlateProcessor Object

Status: Complete  
Progress: 100%  
Last Updated: 2025-04-10  
Dependencies: [[plans/features/nested-config-override-support-plan.md]]

---

## 1. Problem Analysis

Currently, `process_plate_auto()` creates a `PlateProcessorConfig` object, applies overrides, then instantiates a `PlateProcessor` with this config.

However, `PlateProcessor` **creates new component objects** using **nested configs** from the passed config.  
This means nested overrides **after** instantiation **do not affect** these components.

---

## 2. Investigation Result

- The nested configs **must be overridden twice**:
  1. **Before** creating the `PlateProcessor`, apply nested overrides to the root config.
  2. **After** creating the `PlateProcessor`, apply the **same nested overrides** directly to the internal configs of its components:
     - `processor.stitcher.config`
     - `processor.focus_analyzer.config`
     - `processor.zstack_processor.config`
     - `processor.image_preprocessor.config`

This guarantees **all nested configs** are updated, regardless of how `PlateProcessor` initializes its components.

---

## 3. Implementation Plan

- Refactor `process_plate_auto()` to:
  1. Apply nested overrides to the root config.
  2. Instantiate the `PlateProcessor`.
  3. Apply the **same nested overrides** to the nested component configs inside the `PlateProcessor` instance.
- This can be done by calling `apply_nested_overrides()` twice:
  - Once on the root config.
  - Once on each nested component's `.config` attribute.

---

## 4. Validation

- Pass nested overrides and verify they update both the root config and nested component configs.
- Confirm that processing behavior changes accordingly.
- Add tests to ensure nested overrides propagate correctly.

---

## 5. Summary

Nested overrides should be applied **both before and after** `PlateProcessor` instantiation to ensure all nested configs are updated.

This plan is now **complete** and ready for implementation.