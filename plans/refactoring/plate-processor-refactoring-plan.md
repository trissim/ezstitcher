# PlateProcessor Refactoring Plan

```
Status: In Progress
Progress: 20%
Last Updated: 2025-04-10
Dependencies: []
```

## Problem Analysis

### Description
The `run` method in the `PlateProcessor` class is overly large and handles multiple responsibilities, making it difficult to maintain, test, and extend.

### Current Implementation
- Initializes filename parser based on microscope type
- Handles Opera Phenix file conversion
- Creates output directories
- Detects and processes Z-stacks
- Finds HTD files and parses grid dimensions
- Auto-detects patterns
- Processes each well
- Cleans up temporary folders

### Constraints and Requirements
- Must support both Opera Phenix and ImageXpress data
- Maintain backward compatibility with existing workflows
- Avoid breaking existing API
- Ensure all existing tests pass post-refactor

### Potential Edge Cases
- Missing or corrupt HTD files
- Mixed microscope data in one plate
- Unexpected file naming conventions
- Empty or partially transferred folders
- Z-stack detection failures

---

## High-Level Solution

### Architectural Overview
Refactor `PlateProcessor.run()` into a **modular pipeline** with clear, single-responsibility private methods. This will improve readability, maintainability, and testability.

### Component Interactions
- `PlateProcessor` orchestrates the workflow
- Delegates filename parsing to `FilenameParser`
- Uses `ZStackProcessor` for Z-stack detection
- Calls conversion utilities for Opera Phenix data
- Manages output directory creation and cleanup

### Pseudo-code for Key Algorithm

```python
def run(self, plate_folder):
    try:
        plate_path = self._initialize_and_validate(plate_folder)
        self._initialize_filename_parser(plate_path)
        self._handle_opera_phenix_conversion(plate_path)
        dirs = self._create_output_directories(plate_path)
        has_zstack = self.zstack_processor.detect_z_stacks(plate_folder)
        if has_zstack:
            return self._process_zstack_plate(plate_path, dirs)
        else:
            return self._process_regular_plate(plate_path, dirs)
    except Exception as e:
        logger.error(f"Error in PlateProcessor.run: {e}", exc_info=True)
        return False
```

### Data Flow Diagram (to be created)
- Input: Plate folder path
- Output: Processed stitched images and metadata
- Intermediate: Converted files, temporary folders, logs

---

## Implementation Details

### Files to Modify
- `ezstitcher/core/plate_processor.py`

### New Methods to Create
- `_initialize_and_validate`
- `_initialize_filename_parser`
- `_handle_opera_phenix_conversion`
- `_create_output_directories`
- `_process_zstack_plate`
- `_process_regular_plate`
- `_process_well_wavelengths`

### API Specifications
Each new private method will:
- Accept clearly defined inputs (e.g., `Path`, config objects)
- Return well-defined outputs (e.g., bool, dict, Path)
- Raise exceptions on failure, handled in `run()`

### Data Structures
- `config`: PlateProcessorConfig object
- `plate_path`: Path object
- `dirs`: dict of output directories
- `filename_parser`: instance of parser class
- `zstack_processor`: instance of ZStackProcessor

### Error Handling Strategies
- Use try/except in `run()` to catch and log errors
- Raise specific exceptions in private methods for invalid inputs
- Log warnings for recoverable issues (e.g., missing files)

---

## Validation

### Similarity Check
- Compare refactored output with current implementation on test datasets

### Potential Conflicts
- Changes in internal method signatures should not affect public API
- Ensure compatibility with both Opera Phenix and ImageXpress workflows

### Performance Considerations
- Refactoring should not introduce significant overhead
- Profile before and after to confirm

### Testing Approach
- Run existing test suite after each incremental change
- Add new unit tests for extracted private methods
- Use sample datasets for both microscope types

---

## References

- [[plans/refactoring/zstack-processor-refactoring-plan.md]]
- [[plans/refactoring/code-smells-and-refactoring-plan.md]]
- [[plans/features/opera-phenix-support-plan.md]]

---

## Next Steps

1. Create a backup branch
2. Incrementally extract private methods
3. Add unit tests for new methods
4. Run full test suite after each step
5. Update documentation if needed
6. Mark plan as complete when done
