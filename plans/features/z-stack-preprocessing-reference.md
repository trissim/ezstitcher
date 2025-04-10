# Z-Stack Preprocessing Pipeline Reference

## Current Pipeline Flow

1. **Directory Loading & Detection**:
   - The plate directory is loaded
   - Microscope type is detected
   - Images are loaded, renamed, and directories are flattened
   - `PlateProcessor.run()` handles this process
   - `ZStackProcessor` manages Z-stack detection and organization

2. **Preprocessing (Current Implementation)**:
   - Preprocessing functions are applied to 2D images
   - Groups of tiles (all sites of same well, wavelength, and Z-plane) are processed together
   - `Stitcher.prepare_reference_channel()` and `Stitcher.process_imgs_from_pattern()` handle this
   - Preprocessing happens after Z-stack processing, not before

3. **Z-Reference Function (Current Implementation)**:
   - Z-reference functions convert 3D stacks to 2D images
   - `ZStackProcessor.stitch_across_z()` applies these functions
   - This happens before preprocessing is applied to the resulting 2D images

4. **Position Generation & Stitching**:
   - Positions are generated from the reference images
   - These positions are applied to original tiles or optionally to separately processed tiles

## Desired Pipeline Flow

1. **Directory Loading & Detection** (unchanged)

2. **Preprocessing (Desired Implementation)**:
   - Preprocessing functions should be applied to each image in a Z-stack before Z-reference functions
   - Functions may accept a single image or a stack
   - If they accept a single image, use an adapter to make them operate on the whole stack
   - If they accept a stack, use them directly
   - Optionally save preprocessed images to a folder

3. **Z-Reference Function (Desired Implementation)**:
   - Apply Z-reference function to preprocessed Z-stacks
   - Save the flattened, preprocessed tiles to a folder
   - Use this folder for generating stitching positions

4. **Position Generation & Stitching** (unchanged)

## Required Code Changes

1. **Update `ZStackProcessor.__init__`**:
```python
def __init__(self, config: ZStackProcessorConfig, filename_parser=None, preprocessing_funcs=None):
    self.config = config
    self.fs_manager = FileSystemManager()
    self._z_info = None
    self._z_indices = []
    self.preprocessing_funcs = preprocessing_funcs or {}
    # Rest of initialization...
```

2. **Add Preprocessing Method**:
```python
def _preprocess_stack(self, stack, channel):
    """Apply preprocessing to each image in a Z-stack."""
    if channel in self.preprocessing_funcs:
        func = self.preprocessing_funcs[channel]
        return [func(img) for img in stack]
    return stack
```

3. **Update `stitch_across_z`**:
```python
def stitch_across_z(self, plate_folder, reference_z=None, stitch_all_z_planes=True, processor=None, preprocessing_funcs=None):
    """Stitch all Z-planes in a plate using a reference Z-plane for positions."""
    # Use provided preprocessing functions or the ones from initialization
    preprocessing_funcs = preprocessing_funcs or self.preprocessing_funcs
    
    # When loading images for a Z-stack:
    for z_idx in sorted(z_indices):
        # Load image...
        
        # Extract channel from filename
        channel_match = re.search(r'_w(\d+)', base_name)
        channel = channel_match.group(1) if channel_match else '1'
        
        # Apply preprocessing if available for this channel
        if channel in preprocessing_funcs:
            img = preprocessing_funcs[channel](img)
        
        image_stack.append(img)
    
    # Apply reference function to the preprocessed stack
    reference_image = reference_function(image_stack)
```

4. **Update `PlateProcessor._process_zstack_plate`**:
```python
def _process_zstack_plate(self, plate_folder: Path) -> bool:
    # ...
    
    # Pass preprocessing functions to ZStackProcessor
    success = self.zstack_processor.stitch_across_z(
        plate_folder,
        reference_z=None,  # Use reference_method from config
        stitch_all_z_planes=True,
        processor=z_processor,
        preprocessing_funcs=self.config.preprocessing_funcs
    )
    
    # ...
```

5. **Update `create_zstack_projections`**:
```python
def create_zstack_projections(self, input_dir, output_dir, projection_types=None, preprocessing_funcs=None):
    """Create projections from Z-stack images."""
    # Use provided preprocessing functions or the ones from initialization
    preprocessing_funcs = preprocessing_funcs or self.preprocessing_funcs
    
    # When loading images for a Z-stack:
    for z_idx in sorted(z_indices):
        # Load image...
        
        # Extract channel from filename
        channel_match = re.search(r'_w(\d+)', base_name)
        channel = channel_match.group(1) if channel_match else '1'
        
        # Apply preprocessing if available for this channel
        if channel in preprocessing_funcs:
            img = preprocessing_funcs[channel](img)
        
        image_stack.append(img)
    
    # Create projections from preprocessed stack
    # ...
```

## Function Adapter Pattern

The function adapter pattern should be extended to handle both preprocessing functions and Z-reference functions:

```python
def _adapt_function(self, func: Callable, function_type=None) -> Callable:
    """
    Adapt a function to the appropriate interface.
    
    This allows both:
    - Functions that take a single image and return a processed image
    - Functions that take a stack and return a stack (for preprocessing)
    - Functions that take a stack and return a single image (for Z-reference)
    
    Args:
        func: The function to adapt
        function_type: 'preprocess' or 'reference' to specify the expected behavior
        
    Returns:
        A function with the appropriate interface
    """
    # Try to determine if the function works on stacks or single images
    try:
        # Create a small test stack
        test_stack = [np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)]
        
        # Try calling the function with the stack
        result = func(test_stack)
        
        if function_type == 'preprocess':
            # For preprocessing, should return a stack of the same length
            if isinstance(result, list) and len(result) == len(test_stack):
                return func  # Already a stack-to-stack function
            else:
                raise ValueError("Preprocessing function should return a stack of the same length")
        else:  # reference function
            # For reference, should return a single image
            if isinstance(result, np.ndarray) and result.ndim == 2:
                return func  # Already a stack-to-image function
            else:
                raise ValueError("Reference function should return a single 2D image")
    except Exception:
        # If it fails, try with a single image
        try:
            # Try with a single image
            result = func(test_stack[0])
            
            if function_type == 'preprocess':
                # For preprocessing, should return a single image
                if isinstance(result, np.ndarray) and result.ndim == 2:
                    # Create adapter for preprocessing
                    def preprocess_adapter(stack):
                        return [func(img) for img in stack]
                    return preprocess_adapter
                else:
                    raise ValueError("Preprocessing function should return a single 2D image")
            else:  # reference function
                # For reference, should return a single image but we need to adapt it
                if isinstance(result, np.ndarray) and result.ndim == 2:
                    # Create adapter for reference
                    def reference_adapter(stack):
                        processed_stack = [func(img) for img in stack]
                        return np.max(np.array(processed_stack), axis=0)
                    return reference_adapter
                else:
                    raise ValueError("Reference function should return a single 2D image")
        except Exception as e:
            # If both attempts fail, raise an error
            raise ValueError(f"Cannot adapt function {func.__name__}: {str(e)}")
```

## Implementation Notes

1. **Backward Compatibility**:
   - Maintain backward compatibility with existing code
   - Add new parameters with default values that match current behavior

2. **Error Handling**:
   - Add robust error handling for preprocessing functions
   - Log warnings if preprocessing fails but continue processing

3. **Performance Considerations**:
   - Consider memory usage when processing large Z-stacks
   - Implement optional caching of preprocessed images

4. **Testing**:
   - Test with various preprocessing functions
   - Verify that preprocessing happens before Z-reference function application
   - Test with both single-image and stack-based preprocessing functions

## Example Usage

```python
# Define preprocessing functions
def enhance_contrast(image):
    """Enhance contrast of a single image."""
    p2, p98 = np.percentile(image, (2, 98))
    return np.clip((image - p2) / (p98 - p2), 0, 1)

def denoise_stack(stack):
    """Denoise a stack of images."""
    return [ndimage.gaussian_filter(img, sigma=1) for img in stack]

# Process a plate with preprocessing and Z-reference functions
result = process_plate_folder(
    'path/to/plate_folder',
    reference_channels=["1"],
    preprocessing_funcs={
        "1": enhance_contrast,  # Single-image function (will be adapted)
        "2": denoise_stack      # Stack function (will be used directly)
    },
    z_reference_function="max_projection",
    save_reference=True,
    stitch_all_z_planes=True
)
```
