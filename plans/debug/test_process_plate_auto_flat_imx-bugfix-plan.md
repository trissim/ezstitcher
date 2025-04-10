# Debugging Plan: `test_process_plate_auto_flat_imx` Failure

```
Status: In Progress
Progress: 75%
Last Updated: 2025-04-10
Dependencies: []
```

---

## Updated Problem Analysis

- When no images are found, the fallback parser (`ImageXpressFilenameParser`) is set.
- However, **`config.microscope_type` remains `'auto'`**.
- Later, `create_parser(config.microscope_type)` is called with `'auto'`, which is unsupported, causing the error.
- The user wants **no fallback** to a default microscope type.
- Instead, the system should **attempt autodetection**.
- If autodetection **fails** (i.e., no parser can be determined), the system should **raise a clear error**.

---

## Updated Proposed Fix

- When `microscope_type` is `'auto'`:
  - Attempt to **auto-detect** the microscope type using sample filenames.
  - If **autodetection fails** (returns `None` or unsupported), **raise a `ValueError`** with a clear message.
- **Do NOT** default to `'imagexpress'` or any other type silently.

---

## Validation Steps

1. Implement the fix:
   - Use autodetection logic.
   - Raise an error if autodetection fails.
2. Rerun `test_process_plate_auto_flat_imx`.
3. If the test passes, mark this plan as **Complete**.
4. If it fails, update this plan with new analysis and repeat.

---

## Completion Criteria

- The test `test_process_plate_auto_flat_imx` passes with correct autodetection behavior.
- The code **raises a clear error** if microscope type cannot be determined.
- No silent fallback to a default microscope type.