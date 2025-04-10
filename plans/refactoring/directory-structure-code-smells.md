# Code Smells and Refactoring Suggestions

## 1. Hard-Coded Directory Structure References

### Smell:
- "TimePoint_1" is hard-coded throughout the codebase
- Assumptions about directory structure (e.g., expecting TimePoint_1 subfolder)
- Multiple checks for different directory structures scattered across classes

### Examples:
- `DirectoryManager.find_wells()` assumes images are in a TimePoint_1 directory
- `PatternMatcher.path_list_from_pattern()` has a special case for TimePoint_1
- `PlateProcessor.run()` has complex logic to check multiple directory structures
- `ZStackProcessor` and related classes assume specific directory structures

### Solution:
- Create a `DirectoryStructureManager` class to handle different directory structures
- Centralize all directory structure detection and navigation
- Make the timepoint directory name configurable (already in config, but not used consistently)
- Provide a consistent interface for accessing images regardless of structure

## 2. Hard-Coded Pattern Detection

### Smell:
- Regex patterns for filename parsing are duplicated across classes
- Direct use of regex in methods that should delegate to specialized parsers
- Inconsistent handling of different file formats

### Examples:
- `DirectoryManager.find_wells()` has hard-coded regex for ImageXpress and Opera Phenix
- `PatternMatcher.auto_detect_patterns()` duplicates filename parsing logic
- `FileSystemManager.find_files_by_metadata()` has its own parsing logic

### Solution:
- Ensure all filename parsing goes through the `FilenameParser` classes
- Remove direct regex usage from methods that should delegate to parsers
- Create a unified pattern detection system that uses the existing parsers

## 3. Inconsistent File System Operations

### Smell:
- Mix of direct Path operations and FileSystemManager methods
- Inconsistent error handling for file operations
- Duplicated file listing and filtering logic

### Examples:
- Some classes use `Path.glob()` directly, others use `FileSystemManager.list_image_files()`
- Error handling varies from ignoring errors to logging to raising exceptions
- File extension handling is inconsistent (sometimes uppercase, sometimes lowercase)

### Solution:
- Ensure all file system operations go through FileSystemManager
- Standardize error handling for file operations
- Create helper methods for common file operations

## 4. Lack of Abstraction for Directory Structure Variations

### Smell:
- Code assumes specific directory structures
- Complex conditional logic to handle different structures
- Duplicated directory structure detection logic

### Examples:
- `PlateProcessor.run()` has complex logic to check for TimePoint_1, Images, etc.
- `ZStackProcessor.detect_zstack_folders()` assumes specific folder structure
- `PatternMatcher.path_list_from_pattern()` has special case for TimePoint_1

### Solution:
- Create a `DirectoryStructureManager` class with clear abstractions
- Define standard directory structure types (e.g., FLAT, TIMEPOINT, IMAGES, ZSTACK)
- Provide methods to navigate the structure regardless of type

## 5. Duplicated Image Location Logic

### Smell:
- Multiple methods to find images in different locations
- Duplicated logic for handling different directory structures
- Inconsistent handling of file extensions

### Examples:
- `PlateProcessor.run()` has complex logic to find images
- `ZStackProcessor` has its own logic for finding Z-stack images
- `PatternMatcher.path_list_from_pattern()` has its own logic

### Solution:
- Create an `ImageLocator` class to centralize image finding logic
- Provide methods to find images by metadata, pattern, etc.
- Ensure consistent handling of file extensions

## 6. Inconsistent Use of FilenameParser

### Smell:
- Some code uses FilenameParser, some uses direct regex
- Inconsistent handling of different file formats
- Duplicated parsing logic

### Examples:
- `DirectoryManager.find_wells()` uses direct regex instead of FilenameParser
- `PatternMatcher.auto_detect_patterns()` has its own parsing logic
- `FileSystemManager.find_files_by_metadata()` uses FilenameParser inconsistently

### Solution:
- Ensure all filename parsing goes through FilenameParser
- Remove direct regex usage from methods that should delegate to parsers
- Create helper methods for common parsing operations

## Implementation Plan

1. Create a `DirectoryStructureManager` class to handle different directory structures
2. Create an `ImageLocator` class to centralize image finding logic
3. Refactor `FileSystemManager` to use these new classes
4. Update all classes that currently have hard-coded directory structure references
5. Ensure all filename parsing goes through FilenameParser
6. Create tests for the new classes and refactored methods
