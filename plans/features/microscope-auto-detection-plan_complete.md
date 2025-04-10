# Microscope Auto-Detection Plan

Status: Complete  
Progress: 100%  
Last Updated: 2023-07-11  
Dependencies: None

## 1. Problem Analysis

Currently, when `microscope_type='auto'` is specified, the system doesn't properly detect and use the appropriate parser based on the folder structure and file names. Instead, it defaults to ImageXpress without performing actual detection. This leads to:

1. **Inconsistent behavior**: The system claims to auto-detect but actually defaults to ImageXpress
2. **Potential errors**: When processing Opera Phenix data with the wrong parser
3. **Redundant code**: Auto-detection logic is duplicated in multiple places
4. **Confusing API**: The `auto` option doesn't behave as users would expect

The auto-detection should:
- Examine folder structure and file names to determine the microscope type
- Use the appropriate parser based on the detected microscope type
- Provide clear logging about what was detected and why
- Be consistent across all components that need to detect microscope types

## 2. High-Level Solution

1. **Refactor `create_parser`** to properly handle 'auto' by performing actual detection
2. **Centralize detection logic** in the `FilenameParser` class
3. **Update `PlateProcessor`** to use the centralized detection logic
4. **Ensure consistency** across all components that need to detect microscope types

## 3. Implementation Details

### 3.1 Refactor `create_parser` in `filename_parser.py`

```python
def create_parser(microscope_type: str, sample_files: Optional[List[str]] = None, plate_folder: Optional[Union[str, Path]] = None) -> FilenameParser:
    """
    Factory function to create the appropriate parser for a microscope type.

    Args:
        microscope_type (str): Type of microscope ('ImageXpress', 'OperaPhenix', 'auto', etc.)
        sample_files (list, optional): List of sample filenames for auto-detection
        plate_folder (str or Path, optional): Path to plate folder for auto-detection

    Returns:
        FilenameParser: Instance of the appropriate parser

    Raises:
        ValueError: If microscope_type is not supported or auto-detection fails
    """
    microscope_type = microscope_type.lower()
    
    if microscope_type == 'imagexpress':
        return ImageXpressFilenameParser()
    elif microscope_type == 'operaphenix':
        return OperaPhenixFilenameParser()
    elif microscope_type == 'auto':
        # Perform actual auto-detection
        if sample_files:
            # Detect based on sample filenames
            detected_type = FilenameParser.detect_format(sample_files)
            if detected_type:
                logger.info(f"Auto-detected microscope type from filenames: {detected_type}")
                return create_parser(detected_type)  # Recursive call with detected type
        
        if plate_folder:
            # Detect based on folder structure
            from ezstitcher.core.image_locator import ImageLocator
            
            # Find all image locations
            image_locations = ImageLocator.find_image_locations(plate_folder)
            
            # Collect sample files from all locations
            all_samples = []
            for location_type, images in image_locations.items():
                if location_type == 'z_stack':
                    # Handle z_stack specially since it's a nested dictionary
                    for z_index, z_images in images.items():
                        all_samples.extend([Path(f).name for f in z_images[:5]])
                else:
                    all_samples.extend([Path(f).name for f in images[:5]])
            
            if all_samples:
                detected_type = FilenameParser.detect_format(all_samples)
                if detected_type:
                    logger.info(f"Auto-detected microscope type from folder structure: {detected_type}")
                    return create_parser(detected_type)  # Recursive call with detected type
        
        # If we couldn't detect, default to ImageXpress but log a warning
        logger.warning("Could not auto-detect microscope type, defaulting to ImageXpress")
        return ImageXpressFilenameParser()
    else:
        raise ValueError(f"Unsupported microscope type: {microscope_type}")
```

### 3.2 Update `PlateProcessor._initialize_filename_parser_and_convert`

```python
def _initialize_filename_parser_and_convert(self, plate_path):
    config = self.config
    from ezstitcher.core.filename_parser import create_parser
    
    # Use the enhanced create_parser with plate_folder for auto-detection
    self.filename_parser = create_parser(config.microscope_type, plate_folder=plate_path)
    logger.info(f"Using microscope type: {self.filename_parser.__class__.__name__}")
    
    # Handle Opera Phenix conversion if needed
    if self.filename_parser.__class__.__name__ == 'OperaPhenixFilenameParser':
        logger.info(f"Converting Opera Phenix files to ImageXpress format...")
        from ezstitcher.core.image_locator import ImageLocator
        
        # Use ImageLocator to find the appropriate directory
        image_locations = ImageLocator.find_image_locations(plate_path)
        if 'images' in image_locations:
            logger.info(f"Found Images directory, using it for Opera Phenix files")
            self.filename_parser.rename_all_files_in_directory(plate_path / "Images")
        else:
            self.filename_parser.rename_all_files_in_directory(plate_path)
```

### 3.3 Update `ZStackProcessor` to use the enhanced `create_parser`

```python
# In ZStackProcessor.__init__
if self.filename_parser is None:
    from ezstitcher.core.filename_parser import create_parser
    self.filename_parser = create_parser('auto')
```

### 3.4 Update `main.py` to handle auto-detection properly

```python
# In process_plate_auto function
# Create default config if none provided
if config is None:
    config = PlateProcessorConfig()
    # Ensure microscope_type is 'auto' for auto-detection
    config.microscope_type = 'auto'
```

## 4. Validation

### 4.1 Unit Tests

1. Test `create_parser` with 'auto' and sample files from ImageXpress
2. Test `create_parser` with 'auto' and sample files from Opera Phenix
3. Test `create_parser` with 'auto' and a plate folder containing ImageXpress files
4. Test `create_parser` with 'auto' and a plate folder containing Opera Phenix files
5. Test `PlateProcessor._initialize_filename_parser_and_convert` with 'auto'
6. Test `process_plate_auto` with no config (should default to 'auto')

### 4.2 Integration Tests

1. Test with real Opera Phenix data and 'auto' microscope type
2. Test with real ImageXpress data and 'auto' microscope type
3. Test with mixed data and 'auto' microscope type

## 5. Implementation Order

1. ✅ Refactor `create_parser` in `filename_parser.py`
2. ✅ Update `PlateProcessor._initialize_filename_parser_and_convert`
3. ✅ Update `ZStackProcessor` to use the enhanced `create_parser`
4. ✅ Update `main.py` to handle auto-detection properly
5. ✅ Add unit tests for the enhanced auto-detection
6. ✅ Add integration tests for the enhanced auto-detection

## 6. Benefits

1. **Improved user experience**: Auto-detection works as expected
2. **Reduced errors**: Correct parser is used for each microscope type
3. **Centralized logic**: Auto-detection logic is in one place
4. **Better logging**: Clear information about what was detected and why
5. **Consistent behavior**: All components use the same auto-detection logic

## 7. Risks and Mitigations

1. **Risk**: Changes might break existing functionality
   **Mitigation**: Comprehensive unit and integration tests

2. **Risk**: Auto-detection might be incorrect in some cases
   **Mitigation**: Fallback to ImageXpress with clear warning logs

3. **Risk**: Circular dependencies between modules
   **Mitigation**: Careful import management, possibly using lazy imports

## 8. References

- `ezstitcher/core/filename_parser.py` - `create_parser` and `detect_parser` functions
- `ezstitcher/core/plate_processor.py` - `PlateProcessor._initialize_filename_parser_and_convert` method
- `ezstitcher/core/zstack_processor.py` - `ZStackProcessor` class
- `ezstitcher/core/main.py` - `process_plate_auto` function
- `ezstitcher/core/image_locator.py` - `ImageLocator` class

## 9. Completion Summary

All planned changes have been implemented and tested successfully. The auto-detection functionality now works as expected, with the following improvements:

1. The `create_parser` function now properly handles 'auto' by performing actual detection based on sample files or folder structure
2. The `PlateProcessor` class now uses the enhanced `create_parser` function for auto-detection
3. The `ZStackProcessor` class now uses the enhanced `create_parser` function for auto-detection
4. The `main.py` file now ensures that when no config is provided, it defaults to 'auto' for microscope type
5. Comprehensive unit tests have been added to verify the enhanced auto-detection functionality
6. All tests are passing, confirming that the changes work as expected

Date: 2023-07-11
