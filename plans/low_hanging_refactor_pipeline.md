# Low-Hanging Fruit Refactoring Plan for Processing Pipeline

## Overview

This plan outlines specific, actionable improvements to make the `ezstitcher/core/processing_pipeline.py` file leaner and more maintainable without major architectural changes. These are prioritized by impact and ease of implementation.

## Improvements

### 1. Remove Unused Method: `check_for_zstack`

**Issue:**
- The `check_for_zstack` method (lines 543-556) is never called within the codebase
- It's redundant since `fs_manager.detect_zstack_folders` is called directly elsewhere

**Action:**
- Remove the entire method
- This will eliminate 14 lines of code with zero risk

**Benefits:**
- Reduces code size
- Eliminates maintenance burden of unused code
- Improves code clarity

### 2. Eliminate Duplicate Code for Creating Flatten Patterns

**Issue:**
- Code for creating flatten patterns is duplicated in `process_reference_images` (lines 209-220) and `process_final_images` (lines 260-272)

**Action:**
- Extract this into a helper method:
```python
def _create_flatten_patterns(self, patterns, include_channel=False):
    """
    Create patterns for flattening Z-stacks.
    
    Args:
        patterns: List of patterns to process
        include_channel: Whether to include channel in the output pattern
        
    Returns:
        list: List of patterns for flattening
    """
    flatten_patterns = []
    for pattern in patterns:
        sample = pattern.replace('{iii}', '001')
        meta = self.microscope_handler.parser.parse_filename(sample)
        
        kwargs = {
            'well': meta['well'],
            'site': meta['site'],
            'z_index': '{iii}',
            'extension': '.tif',
            'site_padding': DEFAULT_PADDING,
            'z_padding': DEFAULT_PADDING
        }
        
        if include_channel and 'channel' in meta:
            kwargs['channel'] = meta['channel']
            
        flatten_patterns.append(
            self.microscope_handler.parser.construct_filename(**kwargs)
        )
    
    return flatten_patterns
```

**Benefits:**
- Reduces code duplication
- Makes the code more maintainable
- Centralizes pattern creation logic

### 4. Consolidate Directory Creation Logic

**Issue:**
- Directory creation code (lines 100-110) could be extracted to improve readability

**Action:**
- Create a helper method:
```python
def _setup_directories(self, plate_path):
    """
    Set up directory structure for processing.
    
    Args:
        plate_path: Path to the plate folder
        
    Returns:
        dict: Dictionary of directories
    """
    dirs = {
        'input': self._prepare_images(plate_path),
        'processed': plate_path.parent / f"{plate_path.name}{self.config.processed_dir_suffix}",
        'post_processed': plate_path.parent / f"{plate_path.name}{self.config.post_processed_dir_suffix}",
        'positions': plate_path.parent / f"{plate_path.name}{self.config.positions_dir_suffix}",
        'stitched': plate_path.parent / f"{plate_path.name}{self.config.stitched_dir_suffix}"
    }
    
    for dir_path in dirs.values():
        self.fs_manager.ensure_directory(dir_path)
        
    return dirs
```

**Benefits:**
- Makes the `run` method cleaner
- Centralizes directory structure logic
- Improves maintainability

### 5. Simplify Pattern Detection Logic

**Issue:**
- Code for detecting patterns (lines 113-122) is verbose and duplicated

**Action:**
- Create a helper method:
```python
def _detect_patterns(self, input_dir, well_filter=None, variable_component='site'):
    """
    Detect patterns in the input directory.
    
    Args:
        input_dir: Input directory
        well_filter: Optional list of wells to include
        variable_component: Component to vary in the pattern ('site' or 'z_index')
        
    Returns:
        dict: Dictionary mapping wells to patterns
    """
    return self.microscope_handler.auto_detect_patterns(
        input_dir,
        well_filter=well_filter,
        variable_components=[variable_component]
    )
```

**Benefits:**
- Reduces code duplication
- Makes the code more readable
- Centralizes pattern detection logic

### 6. Remove Redundant Variable Assignment

**Issue:**
- In `process_well` (line 137), `wavelength_patterns` and `wavelength_patterns_z` are passed as parameters but then immediately reassigned

**Action:**
- Remove these redundant assignments to simplify the code
- Update the method signature and references accordingly

**Benefits:**
- Reduces confusion
- Simplifies the code
- Eliminates redundant operations
