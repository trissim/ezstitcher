# Deprecated Code Removal Plan

This plan outlines the approach for removing deprecated code that is not used in the new synthetic tests (`test_synthetic_imagexpress_refactored.py` and `test_synthetic_opera_phenix_refactored.py`).

## 1. Files to Remove

### `ezstitcher/core/utils.py`

The `utils.py` file contains utility functions that have been replaced by methods in the new classes. The new synthetic tests don't import or use any functions from this file directly. All functionality has been moved to:

- `FileSystemManager` for file operations
- `ImageLocator` for image finding operations
- `DirectoryStructureManager` for directory structure handling

**Action**: Remove the entire file.

### `ezstitcher/core/directory_manager.py`

The `DirectoryManager` class has been replaced by the `DirectoryStructureManager` class. The new synthetic tests don't import or use the `DirectoryManager` class directly.

**Action**: Remove the entire file.

## 2. Methods to Remove

### In `FileSystemManager`

1. **`find_wells`**: This method has been replaced by `DirectoryStructureManager.get_wells()`. We've already added a deprecation warning to this method.

**Action**: Keep the method with the deprecation warning for backward compatibility.

### In `ZStackProcessor`

1. **Methods that directly use `Path` operations**: These have been updated to use `FileSystemManager` methods.

**Action**: Keep the updated methods.

2. **Methods that hardcode "TimePoint_1"**: These have been updated to use the `DirectoryStructureManager`.

**Action**: Keep the updated methods.

## 3. Legacy Configuration Classes

### In `ezstitcher/core/config.py`

1. **`StitchingConfig`**: This class has been replaced by `PlateProcessorConfig`.
2. **`ZStackConfig`**: This class has been replaced by `ZStackProcessorConfig`.

**Action**: Keep these classes with deprecation warnings for backward compatibility.

## 4. Implementation Steps

### Step 1: Remove `utils.py`

1. Remove the file `ezstitcher/core/utils.py`
2. Update any imports in other files that still reference `utils.py`

### Step 2: Remove `directory_manager.py`

1. Remove the file `ezstitcher/core/directory_manager.py`
2. Update the `FileSystemManager` class to remove the `directory_manager` attribute and any references to it

### Step 3: Update `__init__.py`

1. Remove imports of removed classes from `ezstitcher/core/__init__.py`

### Step 4: Update Tests

1. Ensure all tests pass after the changes

## 5. Backward Compatibility Considerations

Since we're removing entire files, there will be some backward compatibility issues. However, since the new synthetic tests don't use these files, and we've already added deprecation warnings to the methods that are being replaced, the impact should be minimal.

For any code that still depends on the removed files, users will need to update their code to use the new classes and methods.

## 6. Testing Strategy

1. Run the new synthetic tests to ensure they still pass:
   ```bash
   python -m pytest tests/test_synthetic_imagexpress_refactored.py
   python -m pytest tests/test_synthetic_opera_phenix_refactored.py
   ```

2. Run the full test suite to ensure no regressions:
   ```bash
   python -m pytest
   ```

## 7. Documentation Updates

1. Update the README to reflect the removed files and classes
2. Add migration guide for users of the old API
