# Analysis of Current Pipeline and Recommendations

---

## User-Described Pipeline

1. **Microscope detection and renaming**
2. **Flatten directory structure**
3. **Group tiles by well, wavelength, z-plane**
4. **Preprocess groups (single image or stack aware)**
5. **Optionally save preprocessed groups**
6. **Apply Z-reference function on preprocessed groups**
7. **Save flattened, preprocessed tiles**
8. **Generate stitching positions**
9. **Apply positions to original or differently processed tiles**

---

## Current Implementation (Based on `PlateProcessor.run()` and `ZStackProcessor`)

- **Microscope detection and renaming**: Supported (lines 54-119)
- **Flattening and directory setup**: Supported (lines 120-209)
- **Z-stack detection and processing**: Supported (lines 211-285)
- **Grouping by well and wavelength**: Supported via `PatternMatcher` and `auto_detect_patterns` (lines 310-366)
- **Preprocessing functions**: Passed into `prepare_reference_channel` and `process_well_wavelengths`
- **Filename parsing abstraction**: Supported via `FilenameParser`
- **File system abstraction**: Supported via `FileSystemManager`
- **Adaptation of single-image functions to stacks**: Supported via `ZStackProcessor._adapt_function`

---

## Gaps Identified

### 1. Explicit Grouped Preprocessing Before Z-Reference

- Currently, preprocessing is **passed into downstream methods** but **not explicitly applied to groups before Z-reference**.
- **Recommendation:** Implement a **grouped preprocessing step** that:
  - Loads all tiles for a well/wavelength/z-plane
  - Applies preprocessing functions (adapted as needed)
  - Optionally saves the preprocessed group

### 2. Intermediate Saving of Preprocessed Tiles

- No clear support for **saving preprocessed groups** before Z-reference.
- **Recommendation:** Add optional saving of preprocessed groups to a dedicated folder.

### 3. Separate Preprocessing for Stitching

- No clear support for **different preprocessing functions** for:
  - Z-reference generation
  - Final stitching
- **Recommendation:** Allow specifying **two sets of preprocessing functions** in config.

### 4. Explicit Grouping Logic

- Grouping is **implicit** via pattern matching.
- **Recommendation:** Use or extend `PatternMatcher` to **explicitly group tiles** by well, wavelength, z-plane, and pass these groups through the pipeline.

### 5. Pipeline Modularity

- `run()` is **monolithic**.
- **Recommendation:** Break into smaller methods:
  - `_detect_and_rename()`
  - `_group_tiles()`
  - `_preprocess_groups()`
  - `_apply_z_reference()`
  - `_generate_stitching_positions()`
  - `_final_stitching()`

---

## Summary

The current OOP design **largely supports** the desired pipeline, but to fully realize it:

- Add **explicit grouped preprocessing** before Z-reference.
- Support **saving intermediate preprocessed tiles**.
- Allow **different preprocessing for Z-reference and final stitching**.
- Modularize the pipeline into clear, testable steps.

This will enable a **flexible, extensible, and maintainable** processing pipeline aligned with the user's description.