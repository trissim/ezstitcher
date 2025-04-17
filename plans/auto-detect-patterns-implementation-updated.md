# Auto-Detect Patterns Implementation Plan

## Problem Analysis

The current `auto_detect_patterns` method in `FilenameParser` is causing an error because it doesn't accept a parameter named `variable_components`, but the `PipelineOrchestrator` class is trying to pass this parameter. We need to implement a new version that:

1. Accepts the `variable_components` parameter
2. Leverages the existing `parse_filename` and `construct_filename` methods
3. Is microscope-agnostic (doesn't contain format-specific logic)
4. Properly handles pattern generation for different variable components

## Current Error

```
ERROR    ezstitcher.core.processing_pipeline:processing_pipeline.py:132 Pipeline failed: FilenameParser.auto_detect_patterns() got an unexpected keyword argument 'variable_components'
```

## High-Level Solution

Create a new implementation of `auto_detect_patterns` that:

1. Uses the parser's `parse_filename` method to extract components from filenames
2. Generates patterns by replacing specific components with placeholders
3. Groups patterns by well and then by channel or z-index
4. Is implemented in the base `FilenameParser` class to avoid duplication

## Implementation Details

### 1. Define Constants in FilenameParser ABC

Add constants to the `FilenameParser` class for:
- The list of possible filename components
- The placeholder pattern to use for variable components

```python
class FilenameParser(ABC):
    # Constants
    FILENAME_COMPONENTS = ['well', 'site', 'channel', 'z_index', 'extension']
    PLACEHOLDER_PATTERN = '{iii}'
    
    # ... existing methods ...
```

### 2. Update the `auto_detect_patterns` Method Signature

```python
def auto_detect_patterns(self, folder_path, well_filter=None, extensions=None,
                       group_by='channel', variable_components=None):
    """
    Automatically detect image patterns in a folder.

    Args:
        folder_path (str or Path): Path to the folder
        well_filter (list): Optional list of wells to include
        extensions (list): Optional list of file extensions to include
        group_by (str): How to group patterns ('channel' or 'z_index')
        variable_components (list): List of components to make variable (e.g., ['site', 'z_index'])

    Returns:
        dict: Dictionary mapping wells to patterns grouped by channel or z-index
    """
```

### 3. Implementation Algorithm

1. Find all image files in the directory
2. Parse each filename to extract components using `parse_filename`
3. Group files by well
4. For each well, identify unique combinations of non-variable components
5. For each combination, generate a pattern by:
   - Using the first file's metadata as a template
   - Replacing variable components with the placeholder pattern
   - Using `construct_filename` to create the pattern
6. Group patterns by channel or z-index using existing helper methods
7. Return the result

### 4. Detailed Implementation

```python
def auto_detect_patterns(self, folder_path, well_filter=None, extensions=None,
                       group_by='channel', variable_components=None):
    """
    Automatically detect image patterns in a folder.

    Args:
        folder_path (str or Path): Path to the folder
        well_filter (list): Optional list of wells to include
        extensions (list): Optional list of file extensions to include
        group_by (str): How to group patterns ('channel' or 'z_index')
        variable_components (list): List of components to make variable (e.g., ['site', 'z_index'])

    Returns:
        dict: Dictionary mapping wells to patterns grouped by channel or z-index
    """
    from collections import defaultdict
    from ezstitcher.core.image_locator import ImageLocator
    
    # Set default variable components if not provided
    if variable_components is None:
        variable_components = ['site']
        
    # Find all image files
    folder_path = Path(folder_path)
    extensions = extensions or ['.tif', '.TIF', '.tiff', '.TIFF']
    image_dir = ImageLocator.find_image_directory(folder_path)
    logger.info("Using image directory: %s", image_dir)
    image_paths = ImageLocator.find_images_in_directory(image_dir, extensions, recursive=True)
    
    if not image_paths:
        logger.warning("No image files found in %s", folder_path)
        return {}
        
    # Group files by well
    files_by_well = defaultdict(list)
    for img_path in image_paths:
        metadata = self.parse_filename(img_path.name)
        if not metadata:
            continue
            
        well = metadata['well']
        if not well_filter or well in well_filter:
            files_by_well[well].append(img_path)
            
    # Generate patterns for each well
    result = {}
    for well, files in files_by_well.items():
        # Get unique combinations of non-variable components
        component_combinations = defaultdict(list)
        
        for file_path in files:
            metadata = self.parse_filename(file_path.name)
            if not metadata:
                continue
                
            # Create a key based on non-variable components
            key_parts = []
            for comp in self.FILENAME_COMPONENTS:
                if comp not in variable_components and comp in metadata and metadata[comp] is not None:
                    key_parts.append(f"{comp}={metadata[comp]}")
                    
            key = ",".join(key_parts)
            component_combinations[key].append((file_path, metadata))
        
        # Generate patterns for each combination
        patterns = []
        for _, files_metadata in component_combinations.items():
            if not files_metadata:
                continue
                
            # Use the first file's metadata as a template
            _, template_metadata = files_metadata[0]
            
            # Create pattern by replacing variable components with placeholders
            pattern_args = {}
            for comp in self.FILENAME_COMPONENTS:
                if comp in metadata:  # Only include components that exist in the metadata
                    if comp in variable_components:
                        pattern_args[comp] = self.PLACEHOLDER_PATTERN
                    else:
                        pattern_args[comp] = template_metadata[comp]
            
            # Construct the pattern
            pattern = self.construct_filename(
                well=pattern_args['well'],
                site=pattern_args.get('site'),  # Use .get() for optional components
                channel=pattern_args.get('channel'),
                z_index=pattern_args.get('z_index'),
                extension=pattern_args.get('extension', '.tif')
            )
            
            patterns.append(pattern)
        
        # Group patterns by channel or z-index
        if group_by == 'z_index':
            result[well] = self.group_patterns_by_z_index(patterns)
        else:  # Default to channel grouping
            result[well] = self.group_patterns_by_channel(patterns)
            
    return result
```

### 5. Update the `MicroscopeHandler` Class

The `MicroscopeHandler.auto_detect_patterns` method should properly delegate to the parser's implementation:

```python
def auto_detect_patterns(self, folder_path, well_filter=None, extensions=None,
                       group_by='channel', variable_components=None):
    """Delegate to parser."""
    return self.parser.auto_detect_patterns(
        folder_path, 
        well_filter=well_filter, 
        extensions=extensions, 
        group_by=group_by, 
        variable_components=variable_components
    )
```

## Implementation Steps

1. Add constants to the `FilenameParser` class for filename components and placeholder pattern
2. Update the `auto_detect_patterns` method in `FilenameParser` class to accept the `variable_components` parameter
3. Implement the new algorithm that leverages `parse_filename` and `construct_filename`
4. Update the `MicroscopeHandler.auto_detect_patterns` method to properly delegate to the parser

## Validation

1. Run the `processing_pipeline.py` script to verify that the error is fixed
2. Test with different `variable_components` values to ensure patterns are generated correctly
3. Test with both ImageXpress and Opera Phenix formats to ensure the implementation is microscope-agnostic

## Benefits of This Approach

1. **Microscope-Agnostic**: The implementation doesn't contain any format-specific logic
2. **Leverages Existing Methods**: Uses the parser's `parse_filename` and `construct_filename` methods
3. **Flexible**: Can handle any combination of variable components
4. **Maintainable**: Implemented in the base class to avoid duplication
5. **Robust**: Handles edge cases like missing components and filtering
6. **Modular**: Uses constants for filename components and placeholder pattern
