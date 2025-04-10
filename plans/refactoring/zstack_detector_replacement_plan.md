# Plan to Replace `ZStackDetector` with `ZStackProcessor` in `ezstitcher`

## Background

An `ImportError` occurs because `ezstitcher/core/main.py` imports a non-existent module:

```python
from ezstitcher.core.zstack_detector import ZStackDetector
```

The class `ZStackDetector` does **not** exist in the codebase. However, the detection functionality is implemented in the existing `ZStackProcessor` class.

---

## Goals

- Eliminate the `ModuleNotFoundError`
- Maintain Z-stack detection functionality
- Clean up obsolete code references

---

## Step-by-Step Plan

### 1. Remove obsolete import

**File:** `ezstitcher/core/main.py`

**Remove:**

```python
from ezstitcher.core.zstack_detector import ZStackDetector
```

---

### 2. Replace instantiation of `ZStackDetector`

**Locate:**

```python
detector = ZStackDetector(config.z_stack_processor)
```

**Replace with:**

```python
detector = ZStackProcessor(config.z_stack_processor)
```

---

### 3. Verify method compatibility

- The method call:

```python
has_zstack, _ = detector.detect_zstack_images(plate_folder / "TimePoint_1")
```

- is compatible with `ZStackProcessor.detect_zstack_images()`, which accepts a folder path and returns a tuple.

---

### 4. Confirm existing import of `ZStackProcessor`

Already present:

```python
from ezstitcher.core.zstack_processor import ZStackProcessor
```

No changes needed here.

---

### 5. Optional: Update comments or docstrings

If any comments or docstrings refer to `ZStackDetector`, update them to reflect the use of `ZStackProcessor`.

---

## Visual Overview

```mermaid
flowchart TD
    A[process_plate_auto()] --> B[Instantiate ZStackProcessor with config]
    B --> C[Call detect_zstack_images()]
    C --> D{Has Z-stack?}
    D -- Yes --> E[Run Z-stack workflow]
    D -- No --> F[Run 2D workflow]
```

---

## Summary

- The error is caused by an obsolete import and usage of a non-existent `ZStackDetector`.
- The detection logic is implemented in `ZStackProcessor`.
- The fix involves **removing the invalid import** and **replacing the instantiation**.
- This will resolve the error and maintain intended functionality.