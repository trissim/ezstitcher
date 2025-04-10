# Nested Config Override Support Plan

Status: In Progress  
Progress: 0%  
Last Updated: 2025-04-10  
Dependencies: None

---

## 1. Problem Analysis

Currently, `process_plate_auto()` allows overriding only **top-level** attributes of the `PlateProcessorConfig`.  
However, many important parameters (e.g., `tile_overlap`) are nested inside sub-configs like `stitcher`, `focus_analyzer`, etc.

Passing overrides like `tile_overlap=10` does **not** affect the nested configs, leading to confusion and limited flexibility.

**Goal:**  
Enable users to override **nested** configuration parameters dynamically, without explicitly creating nested config objects.

**Constraints:**

- Must maintain backward compatibility.
- Should be intuitive, e.g., using dot notation keys: `"stitcher.tile_overlap": 10`.
- Should not require explicit nested config instantiation.

**Edge Cases:**

- Invalid nested keys.
- Deeply nested attributes.
- Non-existent attributes.

---

## 2. High-Level Solution

- Implement a **recursive override helper** that:
  - Accepts a config object and a dict of overrides.
  - Supports dot notation keys for nested attributes.
  - Traverses nested configs and sets the final attribute if it exists.
- Replace the current shallow override loop in `process_plate_auto()` with this helper.

---

## 3. Implementation Details

### Helper Function: `apply_nested_overrides(config_obj, overrides_dict)`

- For each key in `overrides_dict`:
  - Split by `"."` into parts.
  - Traverse the nested attributes except the last.
  - If the path exists, set the last attribute to the override value.
  - If any part of the path is invalid, skip that override with a warning (optional).

### Integration

- Call `apply_nested_overrides(config, kwargs)` inside `process_plate_auto()` instead of the current loop.
- Document the new nested override capability in the function docstring.

### Sample override usage:

```python
process_plate_auto(
    "/path/to/plate",
    **{
        "stitcher.tile_overlap": 15,
        "focus_analyzer.method": "max_intensity",
        "z_stack_processor.stitch_all_z_planes": True
    }
)
```

---

## 4. Validation

- **Unit tests**:
  - Pass nested overrides and verify nested config attributes are updated.
  - Pass invalid nested keys and ensure no crash.
  - Pass top-level overrides and ensure they still work.

- **Integration tests**:
  - Run `process_plate_auto()` with nested overrides and verify processing behavior changes accordingly.

- **Manual test**:
  - Print or inspect the config after overrides to confirm nested values are set.

---

## 5. References

- `ezstitcher/core/main.py` - `process_plate_auto` implementation
- `ezstitcher/core/config.py` - `PlateProcessorConfig` and nested configs
- User request to support nested overrides dynamically