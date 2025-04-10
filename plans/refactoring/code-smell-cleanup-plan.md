# Code Smell Cleanup Plan

This document outlines code smells identified in the ezstitcher codebase and provides a plan for cleaning them up.

## Identified Code Smells

### 1. Duplicate Method Selection Logic in FocusAnalyzer

**Issue**: Both `find_best_focus` and `compute_focus_metrics` methods in `FocusAnalyzer` have identical code for selecting the focus measure function based on the method parameter.

**Solution**: Extract this logic into a private helper method:

```python
def _get_focus_function(self, method):
    """Get the appropriate focus measure function based on method name."""
    if method == 'combined':
        return self.combined_focus_measure
    elif method == 'nvar' or method == 'normalized_variance':
        return self.normalized_variance
    elif method == 'lap' or method == 'laplacian':
        return self.laplacian_energy
    elif method == 'ten' or method == 'tenengrad':
        return self.tenengrad_variance
    elif method == 'fft':
        return self.adaptive_fft_focus
    else:
        raise ValueError(f"Unknown focus method: {method}")
```

Then update both methods to use this helper.

### 2. Redundant FocusAnalyzer Initialization in ZStackProcessor

**Issue**: In `ZStackProcessor.__init__`, a `FocusAnalyzer` is initialized twice:

```python
# First initialization
self.focus_analyzer = FocusAnalyzer(config.focus_config)

# ...

# Second initialization with the same method
focus_config = FocusAnalyzerConfig(method=config.focus_method)
self.focus_analyzer = FocusAnalyzer(focus_config)
```

**Solution**: Remove the redundant initialization and keep only one.

### 3. Standalone `find_best_focus` Function in main.py

**Issue**: The `find_best_focus` function in `main.py` creates a new `FocusAnalyzer` instance instead of reusing an existing one:

```python
def find_best_focus(image_stack, method='combined', roi=None):
    # Create a FocusAnalyzer with the specified method
    config = FocusAnalyzerConfig(method=method, roi=roi)
    analyzer = FocusAnalyzer(config)
    return analyzer.find_best_focus(image_stack)
```

**Solution**: If this function is used within a context where a `FocusAnalyzer` instance already exists, modify it to accept an optional analyzer parameter:

```python
def find_best_focus(image_stack, method='combined', roi=None, analyzer=None):
    if analyzer is None:
        # Create a new analyzer only if one isn't provided
        config = FocusAnalyzerConfig(method=method, roi=roi)
        analyzer = FocusAnalyzer(config)
    return analyzer.find_best_focus(image_stack)
```

### 4. Direct Use of cv2.imwrite in ZStackProcessor

**Issue**: In `ZStackProcessor.find_best_focus`, there's direct use of `cv2.imwrite` instead of using the `FileSystemManager`:

```python
output_path = output_dir / f"{stack_id}_w{wavelength}.tif"
cv2.imwrite(str(output_path), best_img)
```

**Solution**: Use the instance's `fs_manager` instead:

```python
output_path = output_dir / f"{stack_id}_w{wavelength}.tif"
self.fs_manager.save_image(output_path, best_img)
```

### 5. Redundant Preprocessing Functions Parameter in PlateProcessorConfig

**Issue**: In `process_plate_folder`, the `preprocessing_funcs` parameter is passed twice to `PlateProcessorConfig`:

```python
plate_config = PlateProcessorConfig(
    reference_channels=reference_channels,
    well_filter=well_filter,
    use_reference_positions=use_reference_positions,
    microscope_type=microscope_type,
    preprocessing_funcs=preprocessing_funcs,  # First time
    composite_weights=composite_weights,
    stitcher=stitcher_config,
    focus_analyzer=focus_config,
    image_preprocessor=image_preprocessor_config,  # Contains preprocessing_funcs again
    z_stack_processor=zstack_config
)
```

**Solution**: Remove the redundant parameter and rely on the `image_preprocessor_config`:

```python
plate_config = PlateProcessorConfig(
    reference_channels=reference_channels,
    well_filter=well_filter,
    use_reference_positions=use_reference_positions,
    microscope_type=microscope_type,
    composite_weights=composite_weights,
    stitcher=stitcher_config,
    focus_analyzer=focus_config,
    image_preprocessor=image_preprocessor_config,
    z_stack_processor=zstack_config
)
```

### 6. Redundant ZStackProcessor Creation in modified_process_plate_folder

**Issue**: In `modified_process_plate_folder`, a new `ZStackProcessor` is created with default config just to detect Z-stacks, then discarded:

```python
# Create a ZStackProcessor with default config
z_config = ZStackProcessorConfig()
z_processor = ZStackProcessor(z_config)

# Detect Z-stacks
has_zstack = z_processor.detect_z_stacks(plate_folder)
```

**Solution**: Create a single `ZStackProcessor` with the proper config and reuse it:

```python
# Create a ZStackProcessor with the proper config
zstack_config = ZStackProcessorConfig(
    # ... config parameters ...
)
z_processor = ZStackProcessor(zstack_config)

# Detect Z-stacks
has_zstack = z_processor.detect_z_stacks(plate_folder)

# Later, use the same processor for other operations
```

### 7. Duplicate Channel Extraction Logic in ZStackProcessor

**Issue**: There's duplicated code for extracting the channel from filenames in multiple places:

```python
# In one place
channel_match = re.search(r'-ch(\d+)', base_name)
channel = channel_match.group(1) if channel_match else '1'

# In another place
channel_match = re.search(r'_w(\d+)', base_name)
channel = channel_match.group(1) if channel_match else '1'
```

**Solution**: Use the `filename_parser` that's already available to extract the channel:

```python
metadata = self.filename_parser.parse_filename(filename)
channel = str(metadata['channel']) if metadata and 'channel' in metadata else '1'
```

### 8. Direct File Operations in ZStackProcessor

**Issue**: There are direct file operations in `ZStackProcessor` methods instead of using the `FileSystemManager`:

```python
# Direct file operations
shutil.copy2(img_file, new_path)
```

**Solution**: Use the `fs_manager` for all file operations:

```python
self.fs_manager.copy_file(img_file, new_path)
```

### 9. Redundant Grayscale Conversion in FocusAnalyzer Methods

**Issue**: Each focus measure method in `FocusAnalyzer` has the same code for ensuring the image is grayscale:

```python
# Ensure image is grayscale
if len(img.shape) > 2:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**Solution**: Extract this into a helper method:

```python
def _ensure_grayscale(self, img):
    """Ensure the image is grayscale."""
    if len(img.shape) > 2:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
```

Then use this helper in all focus measure methods.

## Implementation Plan

1. **Step 1**: Fix redundant initializations and direct file operations
   - Remove redundant FocusAnalyzer initialization in ZStackProcessor
   - Replace direct file operations with FileSystemManager methods

2. **Step 2**: Extract duplicate logic into helper methods
   - Add _get_focus_function helper to FocusAnalyzer
   - Add _ensure_grayscale helper to FocusAnalyzer
   - Update methods to use these helpers

3. **Step 3**: Improve method signatures for better reuse
   - Update find_best_focus in main.py to accept an optional analyzer
   - Update other methods as needed for better reuse

4. **Step 4**: Remove redundant parameters
   - Remove redundant preprocessing_funcs parameter in PlateProcessorConfig
   - Improve ZStackProcessor creation in modified_process_plate_folder

5. **Step 5**: Use existing OOP methods for channel extraction
   - Replace regex-based channel extraction with filename_parser

6. **Step 6**: Update documentation
   - Update docstrings to reflect the changes
   - Add comments explaining the changes where necessary
