# Directory Structure Refactoring Plan

## Problem Statement
The current codebase has several issues related to directory structure handling:
1. Hard-coded "TimePoint_1" references throughout the code
2. Inconsistent handling of different directory structures
3. Duplicated logic for finding images in different locations
4. Lack of abstraction for directory structure variations

## Refactoring Plan

### 1. Create a new `DirectoryStructureManager` class

This class will be responsible for:
- Detecting and managing different directory structures
- Finding image files regardless of their location in the directory structure
- Providing a consistent interface for accessing images regardless of the underlying structure

```python
class DirectoryStructureManager:
    """
    Manages different directory structures for microscopy data.
    
    Supports various directory structures:
    - Images directly in the plate folder
    - Images in a TimePoint_1 subfolder
    - Images in Z-stack folders in the plate folder
    - Images in Z-stack folders in the TimePoint_1 subfolder
    - Images in an Images subfolder
    - Images in an Images/TimePoint_1 subfolder
    """
    
    def __init__(self, plate_folder, filename_parser=None, timepoint_dir_name="TimePoint_1"):
        self.plate_folder = Path(plate_folder)
        self.filename_parser = filename_parser
        self.timepoint_dir_name = timepoint_dir_name
        self.structure_type = None
        self.image_locations = {}
        self._detect_structure()
        
    def _detect_structure(self):
        """Detect the directory structure and catalog image locations."""
        # Implementation details
        
    def get_image_path(self, well, site, channel, z_index=None):
        """Get the path to an image based on its metadata."""
        # Implementation details
        
    def list_images(self, well=None, site=None, channel=None, z_index=None):
        """List images matching the specified criteria."""
        # Implementation details
        
    def get_timepoint_dir(self):
        """Get the path to the TimePoint directory if it exists."""
        # Implementation details
        
    def get_z_stack_dirs(self):
        """Get the paths to Z-stack directories if they exist."""
        # Implementation details
```

### 2. Create a new `ImageLocator` class for finding images

This class will be used by `DirectoryStructureManager` to locate images:

```python
class ImageLocator:
    """Locates images in various directory structures."""
    
    @staticmethod
    def find_images_in_directory(directory, extensions=None):
        """Find all images in a directory."""
        # Implementation details
        
    @staticmethod
    def find_images_by_pattern(directory, pattern, extensions=None):
        """Find images matching a pattern in a directory."""
        # Implementation details
        
    @staticmethod
    def find_timepoint_dir(plate_folder, timepoint_dir_name="TimePoint_1"):
        """Find the TimePoint directory in various locations."""
        # Implementation details
        
    @staticmethod
    def find_z_stack_dirs(plate_folder, timepoint_dir_name="TimePoint_1"):
        """Find Z-stack directories in various locations."""
        # Implementation details
```

### 3. Refactor `FileSystemManager` to use `DirectoryStructureManager`

```python
class FileSystemManager:
    def __init__(self, config=None, filename_parser=None):
        # Existing initialization
        self.dir_structure_manager = None  # Will be initialized when needed
        
    def initialize_dir_structure(self, plate_folder):
        """Initialize the directory structure manager for a plate folder."""
        timepoint_dir_name = getattr(self.config, 'timepoint_dir_name', "TimePoint_1")
        self.dir_structure_manager = DirectoryStructureManager(
            plate_folder, 
            self.filename_parser,
            timepoint_dir_name
        )
        return self.dir_structure_manager
        
    def get_image_path(self, plate_folder, well, site, channel, z_index=None):
        """Get the path to an image based on its metadata."""
        if self.dir_structure_manager is None or Path(plate_folder) != Path(self.dir_structure_manager.plate_folder):
            self.initialize_dir_structure(plate_folder)
        return self.dir_structure_manager.get_image_path(well, site, channel, z_index)
        
    def list_images_by_metadata(self, plate_folder, well=None, site=None, channel=None, z_index=None):
        """List images matching the specified metadata criteria."""
        if self.dir_structure_manager is None or Path(plate_folder) != Path(self.dir_structure_manager.plate_folder):
            self.initialize_dir_structure(plate_folder)
        return self.dir_structure_manager.list_images(well, site, channel, z_index)
```

### 4. Refactor `PlateProcessor` to use the enhanced `FileSystemManager`

```python
def run(self, plate_folder):
    # Existing initialization
    
    # Initialize directory structure manager
    dir_structure = self.fs_manager.initialize_dir_structure(plate_folder)
    
    # Get input directory (no need to check multiple locations)
    input_dir = dir_structure.get_timepoint_dir() or plate_path
    
    # Rest of the method
```

### 5. Refactor `ZStackProcessor` and related classes

Update all classes that currently have hard-coded "TimePoint_1" references to use the directory structure manager:

```python
def detect_z_stacks(self, plate_folder):
    # Initialize directory structure if needed
    dir_structure = self.fs_manager.initialize_dir_structure(plate_folder)
    
    # Get Z-stack directories
    z_stack_dirs = dir_structure.get_z_stack_dirs()
    
    # Process Z-stack directories
    # ...
```

### 6. Refactor `PatternMatcher` to use `DirectoryStructureManager`

```python
def path_list_from_pattern(self, directory, pattern):
    # Initialize directory structure
    dir_structure = DirectoryStructureManager(directory, self.filename_parser)
    
    # Use the directory structure to find matching files
    # ...
```

## Classes to be Modified

1. **New Classes**:
   - `DirectoryStructureManager`: To handle different directory structures
   - `ImageLocator`: To locate images in various directory structures

2. **Existing Classes to Modify**:
   - `FileSystemManager`: To use `DirectoryStructureManager`
   - `PlateProcessor`: To use the enhanced `FileSystemManager`
   - `ZStackProcessor`: To remove hard-coded "TimePoint_1" references
   - `ZStackOrganizer`: To use `DirectoryStructureManager`
   - `ZStackProjector`: To use `DirectoryStructureManager`
   - `ZStackFocusManager`: To use `DirectoryStructureManager`
   - `ZStackStitcher`: To use `DirectoryStructureManager`
   - `PatternMatcher`: To use `DirectoryStructureManager`

## Purpose of Modifications

1. **Eliminate Hard-Coded Paths**: Remove all hard-coded "TimePoint_1" references
2. **Support Flexible Directory Structures**: Handle various directory structures without code duplication
3. **Centralize Directory Structure Logic**: Move all directory structure handling to a single class
4. **Improve Code Maintainability**: Make the code more maintainable by reducing duplication
5. **Enhance Extensibility**: Make it easier to add support for new directory structures
