# Plan: Investigate Issues with `process_plate_auto` and Nested Overrides

Status: In Progress  
Progress: 0%  
Last Updated: 2025-04-10  
Dependencies: [[plans/features/nested-config-override-support-plan.md]], [[plans/features/nested-config-override-plate-processor-plan.md]]

---

## 1. Problem Analysis

- The function `process_plate_auto()` is designed to:
  - Create a `PlateProcessorConfig` (or accept one)
  - Apply nested overrides to it
  - Instantiate a `PlateProcessor`
  - Run the processing pipeline
  - Return success/failure

- However, the user reports:
  - The nested override logic (`apply_nested_overrides`) **creates the `PlateProcessor` inside itself** (which is incorrect).
  - The **actual processing also happens inside `apply_nested_overrides`**.
  - As a result, **`process_plate_auto()` does almost nothing** meaningful.
  - This breaks the intended override-then-run flow and causes confusion.

---

## 2. High-Level Solution

- **Decouple** the override logic from instantiation and execution.
- `apply_nested_overrides()` should **only**:
  - Recursively update config objects.
  - **Not** instantiate or run anything.
- `process_plate_auto()` should:
  - Prepare or accept a config.
  - Apply nested overrides.
  - Instantiate the `PlateProcessor`.
  - Run the processing pipeline.
  - Return success/failure.

---

## 3. Implementation Details

- **Audit** the current `apply_nested_overrides()` implementation:
  - Confirm if it **creates `PlateProcessor` instances** or **runs processing** inside itself.
  - If so, **refactor** it to **only** update configs.
- **Refactor** `process_plate_auto()` to:
  - Call `apply_nested_overrides()` **before** instantiating `PlateProcessor`.
  - Instantiate `PlateProcessor` **after** overrides are applied.
  - Run the processing pipeline.
  - Return the result.

---

## 4. Validation

- Add debug prints or logs to confirm:
  - When overrides are applied.
  - When the processor is instantiated.
  - When processing starts and ends.
- Run tests to ensure:
  - Overrides are respected.
  - Processing completes successfully.
  - The function returns correct success/failure status.

---

## 5. Next Steps

1. Read the current implementation of `apply_nested_overrides()`.
2. Confirm if it improperly instantiates or runs processing.
3. Refactor as needed.
4. Update `process_plate_auto()` accordingly.
5. Validate with tests.

---

## 6. References

- `ezstitcher/core/main.py` - `process_plate_auto` and `apply_nested_overrides`
- [[plans/features/nested-config-override-support-plan.md]]
- [[plans/features/nested-config-override-plate-processor-plan.md]]