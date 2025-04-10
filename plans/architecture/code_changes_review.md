# Code Changes Review: Removing Deprecated Code

## Overview

This commit focused on removing deprecated code and refactoring the codebase for a cleaner architecture. The main goals were:

1. Remove deprecated utility files and move their functionality to appropriate classes
2. Replace the deprecated `DirectoryManager` with the new `DirectoryStructureManager`
3. Remove the deprecated `ZStackConfig` class and update related code
4. Fix circular imports between modules
5. Ensure all tests pass with the refactored code

## Detailed Changes

### 1. Removed Files

#### `ezstitcher/core/utils.py`

This file contained utility functions that were moved to more appropriate classes:

- `create_linear_weight_mask`: Moved to `ImagePreprocessor` class
- `load_image`: Moved to `FileSystemManager` class
- `save_image`: Moved to `FileSystemManager` class
- `parse_positions_csv`: Already existed in `CSVHandler` class

#### `ezstitcher/core/directory_manager.py`

This file contained the deprecated `DirectoryManager` class that was replaced by the new `DirectoryStructureManager` class. The following methods were updated to use `DirectoryStructureManager` or direct `Path` operations:

- `ensure_directory`: Updated to use `Path` directly
- `clean_temp_folders`: Updated to use `Path` directly
- `create_output_directories`: Updated to use `Path` directly

### 2. Updated Files

#### `ezstitcher/core/__init__.py`

- Removed imports of removed classes (`DirectoryManager`)
- Added imports for new classes (`DirectoryStructureManager`, `ImageLocator`)

#### `ezstitcher/core/config.py`

- Removed the deprecated `ZStackConfig` class
- Updated `PlateConfig` to use `ZStackProcessorConfig` instead of `ZStackConfig`

#### `ezstitcher/core/file_system_manager.py`

- Removed the `directory_manager` attribute and all references to it
- Updated the `ensure_directory` method to use `Path` directly
- Updated the `clean_temp_folders` method to use `Path` directly
- Updated the `create_output_directories` method to use `Path` directly
- Updated the `load_image` method to implement the functionality directly instead of importing from `image_preprocessor`

#### `ezstitcher/core/image_preprocessor.py`

- Removed duplicate utility functions (`load_image`, `save_image`, `parse_positions_csv`)
- Kept only the `create_linear_weight_mask` function as it's genuinely related to image processing
- Added a comment indicating where the removed functions were moved to

#### `ezstitcher/core/stitcher.py`

- Updated imports to use the correct locations for functions
- Removed import of `load_image` and `save_image` from `image_preprocessor`

### 3. Architectural Improvements

#### Reduced Circular Dependencies

- Removed circular dependency between `file_system_manager.py` and `image_preprocessor.py`
- Ensured that utility functions are in their most appropriate classes

#### Clearer Responsibility Boundaries

- `FileSystemManager`: Handles all file operations (loading, saving, listing, etc.)
- `ImagePreprocessor`: Handles image processing operations (not file operations)
- `CSVHandler`: Handles CSV parsing operations
- `DirectoryStructureManager`: Handles directory structure detection and management

#### Simplified Code

- Removed unnecessary indirection through the `DirectoryManager` class
- Simplified imports by having functions in their most logical locations

### 4. Testing

All tests are passing, confirming that the refactoring was successful:

- `test_synthetic_imagexpress_refactored.py`
- `test_synthetic_opera_phenix_refactored.py`

This includes tests for:
- Directory structure detection
- Non-Z-stack workflow
- Z-stack projection stitching
- Z-stack per-plane stitching
- Multi-channel reference

## Conclusion

This refactoring has significantly improved the codebase by:

1. Removing deprecated code that was no longer needed
2. Simplifying the architecture by reducing indirection
3. Clarifying responsibility boundaries between classes
4. Fixing circular dependencies
5. Ensuring all functionality works correctly through comprehensive testing

The codebase is now cleaner, more maintainable, and has a more consistent architecture.
