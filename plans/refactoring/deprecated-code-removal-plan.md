# Deprecated Code Removal Plan

## Overview

This plan outlines the approach for removing deprecated code from the ezstitcher codebase while maintaining backward compatibility. The refactoring will focus on leveraging the new classes (`DirectoryStructureManager`, `ImageLocator`, etc.) that have been implemented to replace older, less flexible approaches.

## Identified Deprecated Code

### 1. Directory Structure Handling

#### Classes to Remove/Replace:
- `DirectoryManager` class (replace with `DirectoryStructureManager` and `ImageLocator`)

#### Methods to Remove/Replace:
- `DirectoryManager.find_wells()` - Replace with `DirectoryStructureManager.get_wells()`
- `DirectoryManager.clean_temp_folders()` - Move to `FileSystemManager` with improved pattern matching

### 2. Z-Stack Processing

#### Deprecated Parameters in `ZStackProcessorConfig`:
- `reference_method` - Use `z_reference_function` instead
- `focus_detect` - Use `z_reference_function="best_focus"` instead
- `stitch_z_reference` - Use `z_reference_function` instead
- `create_projections` - Use `save_reference` instead
- `save_projections` - Use `save_reference` instead
- `projection_types` - Use `additional_projections` instead

#### Deprecated Methods in `ZStackProcessor`:
- Methods that directly use `Path` operations instead of `FileSystemManager`
- Methods that hardcode "TimePoint_1" instead of using configuration

### 3. Legacy Configuration Classes

#### Classes to Remove:
- `StitchingConfig` - Replace with `PlateProcessorConfig`
- `ZStackConfig` - Replace with `ZStackProcessorConfig`

### 4. Utility Functions

#### Functions to Remove/Replace:
- `list_image_files()` in utils.py - Replace with `ImageLocator.find_images_in_directory()`

## Refactoring Approach

### Phase 1: Update References to Deprecated Code

1. Identify all places in the codebase that call deprecated methods
2. Update these calls to use the new methods instead
3. Ensure all tests pass with the updated references

### Phase 2: Create Adapter Methods

1. For each deprecated method that is still being used externally:
   - Create an adapter method that calls the new implementation
   - Add a deprecation warning to the adapter method
   - Document the recommended replacement

Example:
```python
def find_wells(self, timepoint_dir):
    """
    Find all wells in the timepoint directory.
    
    Deprecated: Use DirectoryStructureManager.get_wells() instead.
    
    Args:
        timepoint_dir (str or Path): Path to the TimePoint_1 directory
        
    Returns:
        list: List of well names
    """
    warnings.warn(
        "DirectoryManager.find_wells() is deprecated. Use DirectoryStructureManager.get_wells() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Initialize directory structure manager
    dir_structure = DirectoryStructureManager(timepoint_dir.parent)
    
    # Return wells
    return dir_structure.get_wells()
```

### Phase 3: Remove Deprecated Code

1. Remove deprecated methods that are no longer used internally
2. Keep adapter methods for backward compatibility
3. Update documentation to reflect the new API

## Implementation Plan

### 1. Update `FileSystemManager`

1. Ensure all methods use `DirectoryStructureManager` internally
2. Add adapter methods for backward compatibility
3. Update documentation to recommend new methods

### 2. Update `ZStackProcessor`

1. Remove all direct `Path` operations
2. Use `DirectoryStructureManager` for all directory structure handling
3. Add adapter methods for backward compatibility

### 3. Update `PlateProcessor`

1. Remove all direct `Path` operations
2. Use `DirectoryStructureManager` for all directory structure handling
3. Add adapter methods for backward compatibility

### 4. Update Tests

1. Update all tests to use the new methods
2. Add tests for adapter methods
3. Ensure all tests pass

## Backward Compatibility Strategy

1. Keep adapter methods for all deprecated methods
2. Add deprecation warnings to adapter methods
3. Document the recommended replacements
4. Ensure all tests pass with both old and new methods

## Testing Strategy

1. Run the existing tests to ensure they still pass
2. Run the new tests for `DirectoryStructureManager` and `ImageLocator`
3. Run the refactored synthetic tests for both ImageXpress and Opera Phenix formats

## Documentation Updates

1. Update the README to reflect the new API
2. Add migration guide for users of the old API
3. Update docstrings to include deprecation warnings and recommended replacements

## Timeline

1. Phase 1: Update References to Deprecated Code - 1 day
2. Phase 2: Create Adapter Methods - 1 day
3. Phase 3: Remove Deprecated Code - 1 day
4. Testing and Documentation - 1 day

Total: 4 days
