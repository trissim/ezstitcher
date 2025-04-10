# PlateProcessor Refactoring Plan

```
Status: In Progress
Progress: 0%
Last Updated: 2023-04-09
Dependencies: []
```

## Problem Analysis

Based on the codebase review, one of the issues identified is that the `run` method in the `PlateProcessor` class is quite large (over 100 lines) and handles multiple responsibilities. This makes the code harder to understand, test, and maintain.

### Current Implementation

The `run` method in `PlateProcessor` currently:
1. Initializes the filename parser based on microscope type
2. Handles Opera Phenix file conversion if needed
3. Creates output directories
4. Detects and processes Z-stacks
5. Finds HTD files and parses grid dimensions
6. Auto-detects patterns
7. Processes each well
8. Cleans up temporary folders

This is too many responsibilities for a single method and violates the Single Responsibility Principle.

### Requirements

1. Break down the `run` method into smaller, more focused methods
2. Maintain the same functionality
3. Improve readability and maintainability
4. Ensure all tests pass after the changes

## High-Level Solution

The solution involves:

1. Identifying logical sections in the `run` method
2. Extracting these sections into separate private methods
3. Updating the `run` method to call these new methods
4. Ensuring proper error handling throughout

### Proposed Method Structure

```python
def run(self, plate_folder):
    """Main entry point for processing a plate folder."""
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

## Implementation Details

### Files to Modify

Based on the codebase review, the main file to modify is:
- `ezstitcher/core/plate_processor.py`

### New Methods to Create

1. `_initialize_and_validate(self, plate_folder)`: Validates the plate folder and returns the Path object
2. `_initialize_filename_parser(self, plate_path)`: Initializes the filename parser based on microscope type
3. `_handle_opera_phenix_conversion(self, plate_path)`: Handles Opera Phenix file conversion if needed
4. `_create_output_directories(self, plate_path)`: Creates output directories and returns them
5. `_process_zstack_plate(self, plate_path, dirs)`: Processes a plate with Z-stacks
6. `_process_regular_plate(self, plate_path, dirs)`: Processes a regular plate without Z-stacks
7. `_process_well_wavelengths(self, well, wavelength_patterns, dirs, grid_dims)`: Processes wavelengths for a well

### Sample Implementation

Here's a sample of how the refactored code might look:

```python
def run(self, plate_folder):
    """
    Process a plate folder with microscopy images.

    Args:
        plate_folder (str or Path): Path to the plate folder

    Returns:
        bool: True if successful, False otherwise
    """
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

def _initialize_and_validate(self, plate_folder):
    """
    Initialize and validate the plate folder.

    Args:
        plate_folder (str or Path): Path to the plate folder

    Returns:
        Path: Path object for the plate folder
    """
    plate_path = Path(plate_folder)
    if not plate_path.exists():
        raise ValueError(f"Plate folder does not exist: {plate_path}")
    return plate_path

def _initialize_filename_parser(self, plate_path):
    """
    Initialize the filename parser based on microscope type.

    Args:
        plate_path (Path): Path to the plate folder
    """
    config = self.config
    if config.microscope_type.lower() == 'auto':
        # Auto-detect the microscope type from the filenames
        sample_files = self.fs_manager.list_image_files(plate_path, extensions=['.tif', '.tiff', '.TIF', '.TIFF'])[:10]
        sample_files = [Path(f).name for f in sample_files]
        
        if not sample_files:
            logger.warning(f"No image files found in {plate_path}, cannot auto-detect microscope type")
            self.filename_parser = ImageXpressFilenameParser()
        else:
            self.filename_parser = detect_parser(sample_files)
            logger.info(f"Auto-detected microscope type: {self.filename_parser.__class__.__name__}")
    elif config.microscope_type.lower() == 'imagexpress':
        self.filename_parser = ImageXpressFilenameParser()
    elif config.microscope_type.lower() == 'operaphenix':
        self.filename_parser = OperaPhenixFilenameParser()
    else:
        raise ValueError(f"Unsupported microscope type: {config.microscope_type}")
```

### Testing Strategy

1. Run the existing tests to ensure they pass with the current implementation
2. Refactor the code one method at a time
3. Run the tests after each change to identify any issues
4. Fix any issues that arise
5. Continue until all methods are refactored

## Validation

The changes should not affect the behavior of the `PlateProcessor` class, as we're only reorganizing the code, not changing its logic. The tests should continue to pass after the changes.

## Next Steps

1. Examine the current implementation in detail
2. Create a backup of the current code
3. Implement the changes one method at a time
4. Run tests after each change
5. Document any issues encountered and their solutions
