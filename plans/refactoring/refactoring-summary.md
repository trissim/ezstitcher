# Refactoring Summary: Removing Deprecated Code

## Overview

This document summarizes the plan to remove deprecated code from the ezstitcher codebase while maintaining backward compatibility. The refactoring will leverage the new `DirectoryStructureManager` and `ImageLocator` classes that have been implemented to replace older, less flexible approaches.

## Key Changes

### 1. Directory Structure Handling

- Replace `DirectoryManager.find_wells()` with `DirectoryStructureManager.get_wells()`
- Replace direct `Path` operations with `FileSystemManager` methods
- Remove hard-coded "TimePoint_1" references

### 2. Z-Stack Processing

- Update `ZStackProcessor` to use `DirectoryStructureManager` for all directory structure handling
- Simplify configuration by using only the new parameters in `ZStackProcessorConfig`
- Add adapter methods for backward compatibility

### 3. Configuration

- Simplify configuration by using only the new configuration classes
- Add adapter methods to convert legacy configs to new configs
- Update documentation to recommend new configuration approach

### 4. Utility Functions

- Replace utility functions with methods from the new classes
- Add adapter methods for backward compatibility

## Implementation Phases

### Phase 1: Update References to Deprecated Code

- Identify all places in the codebase that call deprecated methods
- Update these calls to use the new methods instead
- Ensure all tests pass with the updated references

### Phase 2: Create Adapter Methods

- For each deprecated method that is still being used externally, create an adapter method
- Add deprecation warnings to adapter methods
- Document the recommended replacements

### Phase 3: Remove Deprecated Code

- Remove deprecated methods that are no longer used internally
- Keep adapter methods for backward compatibility
- Update documentation to reflect the new API

## Testing Strategy

- Run the existing tests to ensure they still pass
- Run the new tests for `DirectoryStructureManager` and `ImageLocator`
- Run the refactored synthetic tests for both ImageXpress and Opera Phenix formats

## Benefits

1. **Cleaner Code**: Removing deprecated code will make the codebase cleaner and easier to maintain
2. **Better Abstraction**: Using the new classes will provide better abstraction and flexibility
3. **Improved Testability**: The new classes are designed to be more testable
4. **Enhanced Extensibility**: The new classes are designed to be more extensible
5. **Consistent API**: The new API will be more consistent and easier to use

## Backward Compatibility

Backward compatibility will be maintained through:

1. **Adapter Methods**: Deprecated methods will be replaced with adapter methods that call the new implementations
2. **Deprecation Warnings**: Adapter methods will include deprecation warnings to encourage migration
3. **Documentation**: Documentation will be updated to guide users to the new API

## Conclusion

This refactoring will improve the quality of the codebase while maintaining backward compatibility. The new classes provide better abstraction, flexibility, and testability, making the codebase more maintainable and extensible.
