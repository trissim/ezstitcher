# Pull Request: Class-Based Refactoring and Enhanced Configuration System

## Overview

This pull request represents a comprehensive refactoring of the ezstitcher codebase from a static method-based approach to a more modular, object-oriented architecture using instance methods. The refactoring improves code organization, maintainability, and extensibility while adding several new features and enhancements.

## Major Changes

### 1. Architecture Refactoring

- **Class-Based Design**: Replaced static methods with instance methods throughout the codebase, allowing for better state management and more intuitive object interactions.
- **Modular Components**: Refactored code into clear OOP classes with well-defined responsibilities:
  - `PlateProcessor`: High-level orchestrator for processing microscopy plates
  - `ZStackProcessor`: Handles Z-stack detection, organization, and processing
  - `FocusAnalyzer`: Implements focus quality detection algorithms (replaces FocusDetector)
  - `Stitcher`: Implements image stitching algorithms
  - `ImagePreprocessor`: Handles image preprocessing operations
  - `FileSystemManager`: Centralizes file system operations

### 2. Configuration System

- **Pydantic Models**: Implemented a robust configuration system using Pydantic for validation, serialization, and hierarchical configuration management.
- **Configuration Presets**: Added predefined configuration presets for common use cases.
- **Backward Compatibility**: Maintained compatibility with the legacy dataclass-based configuration system.

### 3. New Features

- **Custom Projection Functions**: Added support for custom functions in `stitch_z_reference`, allowing users to define their own projection methods for Z-stack stitching.
- **Percentile Normalization**: Implemented percentile-based normalization for improved contrast in microscopy images.
- **Stack Histogram Normalization**: Added global histogram normalization across Z-stacks for consistent visualization.
- **Custom Focus ROI**: Added support for specifying regions of interest for focus detection.

### 4. Documentation and Examples

- **Comprehensive Documentation**: Added extensive documentation including:
  - API reference for all classes and methods
  - Usage examples for common workflows
  - Installation and quickstart guides
  - Detailed user guide
- **Example Scripts**: Added example scripts demonstrating key features and workflows.

### 5. Testing

- **Comprehensive Test Suite**: Expanded test coverage with unit tests, integration tests, and synthetic workflow tests.
- **Documentation Examples Tests**: Added tests to verify that the examples in the documentation work correctly.
- **Test Organization**: Improved test organization and documentation.

### 6. Code Cleanup

- **Standardized Naming**: Standardized on snake_case for folder names and consistent naming conventions.
- **Pattern Matching**: Used pattern matching to clean up temporary folders.
- **Removed Redundancy**: Eliminated duplicate code and consolidated similar functionality.

## Detailed Changes

### Added Files

- **Core Components**:
  - `ezstitcher/core/pydantic_config.py`: Pydantic configuration models
  - `ezstitcher/core/file_system_manager.py`: Centralized file system operations
  - `ezstitcher/core/plate_processor.py`: High-level orchestrator
  - `ezstitcher/core/image_preprocessor.py`: Image preprocessing operations
  - `ezstitcher/core/zstack_processor.py`: Z-stack processing
  - Several utility classes for specific operations

- **Documentation**:
  - Added comprehensive documentation in `docs/source/`
  - Added examples demonstrating key features and workflows

- **Tests**:
  - Added unit tests for all new components
  - Added integration tests for the full workflow
  - Added tests for documentation examples

### Modified Files

- **Core API**:
  - `ezstitcher/core/main.py`: Updated to use the new class-based architecture
  - `ezstitcher/core/stitcher.py`: Refactored to use instance methods
  - `ezstitcher/core/focus_analyzer.py`: Renamed from focus_detector.py and refactored

- **Configuration**:
  - `ezstitcher/core/config.py`: Added dataclass-based configuration classes

- **Documentation**:
  - `README.md`: Updated to reflect current API usage and tested features

### Removed Files

- **Legacy Components**:
  - `ezstitcher/core/image_process.py`: Functionality moved to ImagePreprocessor
  - `ezstitcher/core/image_processor.py`: Functionality moved to ImagePreprocessor
  - `ezstitcher/core/focus_detect.py`: Functionality moved to FocusAnalyzer
  - `ezstitcher/core/stitcher_manager.py`: Functionality moved to PlateProcessor
  - `ezstitcher/core/z_stack_handler.py`: Functionality moved to ZStackProcessor
  - `ezstitcher/core/z_stack_manager.py`: Functionality moved to ZStackProcessor

## Breaking Changes

1. The API has changed significantly, with static methods replaced by instance methods.
2. Configuration is now primarily handled through Pydantic models.
3. Directory structure for processed files has been standardized.

## Migration Guide

For users of the previous version:

1. Replace calls to static methods with instance method calls:
   ```python
   # Old
   from ezstitcher.core.image_process import preprocess_image
   processed = preprocess_image(img)
   
   # New
   from ezstitcher import ImagePreprocessor
   preprocessor = ImagePreprocessor()
   processed = preprocessor.preprocess_image(img)
   ```

2. Use the new configuration system:
   ```python
   # Old
   from ezstitcher.core.main import process_plate_folder
   process_plate_folder(plate_folder, tile_overlap=10.0)
   
   # New
   from ezstitcher import PlateProcessorConfig, PlateProcessor
   config = PlateProcessorConfig(stitcher=StitcherConfig(tile_overlap=10.0))
   processor = PlateProcessor(config)
   processor.run(plate_folder)
   ```

3. For backward compatibility, the old function signatures are still supported:
   ```python
   # Still works
   from ezstitcher import process_plate_folder
   process_plate_folder(plate_folder, tile_overlap=10.0)
   ```

## Testing

All tests pass, including:
- Unit tests for individual components
- Integration tests for the full workflow
- Tests for documentation examples
- Synthetic workflow tests

## Future Work

1. Further improve documentation with more examples
2. Add support for additional microscopy formats
3. Optimize performance for large datasets
4. Add more configuration presets for common use cases

## Conclusion

This refactoring represents a significant improvement in the architecture and capabilities of ezstitcher. The new class-based design provides a more intuitive API, better state management, and improved extensibility, while the enhanced configuration system offers robust validation and serialization capabilities. The addition of new features like custom projection functions and percentile normalization enhances the library's utility for microscopy image processing.
