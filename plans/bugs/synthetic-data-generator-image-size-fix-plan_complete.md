# Plan: Fix Synthetic Data Generator Image Size Calculation

Status: Complete  
Progress: 100%  
Last Updated: 2023-07-15  
Dependencies: None

## 1. Problem Analysis

The synthetic data generator currently requires an explicit image size parameter, but this can lead to issues if the specified image size isn't large enough to accommodate all tiles with their overlap. This is causing problems in the stitching process where some tiles are being duplicated in the output.

**Current Issues:**
- The image size must be manually specified
- If the image size is too small, tiles may be clipped or positioned incorrectly
- The generator doesn't validate if the image size is sufficient for the grid and tile size
- This can lead to stitching errors where tiles are duplicated or misplaced

**Requirements:**
- Automatically calculate an appropriate image size based on grid dimensions, tile size, and overlap
- Make the change backward compatible with existing code
- Ensure the calculated image size provides enough space for all tiles with proper overlap
- Add proper validation to prevent issues with tile positioning

## 2. High-Level Solution

1. Add an auto-calculation feature for image size
2. Make it work with the existing API (backward compatible)
3. Ensure the calculated image size is sufficient for all tiles with proper overlap
4. Add validation to ensure tiles don't exceed the image boundaries

## 3. Implementation Details

### 3.1 Modify the `__init__` method

```python
def __init__(self,
             output_dir,
             grid_size=(3, 3),
             image_size=(1024, 1024),
             tile_size=(512, 512),
             overlap_percent=10,
             stage_error_px=2,
             wavelengths=2,
             z_stack_levels=1,
             z_step_size=1.0,
             num_cells=50,
             cell_size_range=(10, 30),
             cell_eccentricity_range=(0.1, 0.5),
             cell_intensity_range=(5000, 20000),
             background_intensity=500,
             noise_level=100,
             wavelength_params=None,
             shared_cell_fraction=0.95,
             wavelength_intensities=None,
             wavelength_backgrounds=None,
             wells=['A01'],
             format='ImageXpress',
             auto_image_size=False,  # New parameter
             random_seed=None):
    """
    Initialize the synthetic microscopy generator.
    
    Args:
        ...existing parameters...
        auto_image_size: If True, automatically calculate image size based on grid and tile parameters
        ...
    """
```

### 3.2 Add a helper method to calculate appropriate image size

```python
def _calculate_image_size(self, grid_size, tile_size, overlap_percent, stage_error_px):
    """
    Calculate the appropriate image size based on grid dimensions, tile size, and overlap.
    
    Args:
        grid_size: Tuple of (rows, cols) for the grid of tiles
        tile_size: Size of each tile (width, height)
        overlap_percent: Percentage of overlap between tiles
        stage_error_px: Random error in stage positioning (pixels)
        
    Returns:
        tuple: (width, height) of the calculated image size
    """
    # Calculate effective step size with overlap
    step_x = int(tile_size[0] * (1 - overlap_percent / 100))
    step_y = int(tile_size[1] * (1 - overlap_percent / 100))
    
    # Calculate minimum required size
    min_width = step_x * (grid_size[1] - 1) + tile_size[0]
    min_height = step_y * (grid_size[0] - 1) + tile_size[1]
    
    # Add margin for stage positioning errors
    margin = stage_error_px * 2
    width = min_width + margin
    height = min_height + margin
    
    return (width, height)
```

### 3.3 Update the initialization code

```python
# In __init__ method, after storing parameters
self.output_dir = Path(output_dir)
self.grid_size = grid_size
self.tile_size = tile_size
self.overlap_percent = overlap_percent
self.stage_error_px = stage_error_px

# Calculate image size if auto_image_size is True
if auto_image_size:
    self.image_size = self._calculate_image_size(grid_size, tile_size, overlap_percent, stage_error_px)
    print(f"Auto-calculated image size: {self.image_size[0]}x{self.image_size[1]}")
else:
    self.image_size = image_size
```

### 3.4 Add validation in the cell generation code

```python
# In the cell generation code, ensure positions are within bounds
x = min(x, self.image_size[0] - 1)
y = min(y, self.image_size[1] - 1)
```

### 3.5 Update the command-line interface

```python
# In the argument parser
parser.add_argument("--auto-image-size", action="store_true", 
                    help="Automatically calculate image size based on grid and tile parameters")

# In the main function
generator = SyntheticMicroscopyGenerator(
    output_dir=args.output_dir,
    grid_size=tuple(args.grid_size),
    image_size=tuple(args.image_size),
    tile_size=tuple(args.tile_size),
    overlap_percent=args.overlap,
    stage_error_px=args.stage_error,
    wavelengths=args.wavelengths,
    z_stack_levels=args.z_stack,
    z_step_size=args.z_step_size,
    num_cells=args.num_cells,
    cell_size_range=tuple(args.cell_size),
    cell_eccentricity_range=tuple(args.cell_eccentricity),
    cell_intensity_range=tuple(args.cell_intensity),
    background_intensity=args.background,
    noise_level=args.noise,
    wavelength_params=wavelength_params,
    format=args.format,
    auto_image_size=args.auto_image_size,  # New parameter
    random_seed=args.seed
)
```

### 3.6 Update documentation

```python
"""
Generate synthetic microscopy images for testing ezstitcher.

This script generates synthetic microscopy images with the following features:
- Multiple wavelengths (channels)
- Z-stack support with varying focus levels
- Cell-like structures (circular particles with varying eccentricity)
- Proper tiling with configurable overlap
- Realistic stage positioning errors
- HTD file generation for metadata
- Automatic image size calculation based on grid and tile parameters

Usage:
    python generate_synthetic_data.py output_dir --grid-size 3 3 --wavelengths 2 --z-stack 3 --auto-image-size
"""
```

## 4. Testing Plan

1. Test with auto_image_size=True and verify the image size is calculated correctly
2. Test with various grid sizes and tile sizes to ensure the calculation works correctly
3. Test backward compatibility by using the existing API without auto_image_size
4. Verify that the stitching works correctly with the auto-calculated image size

## 5. Implementation Steps

1. ✅ Add the auto_image_size parameter to the __init__ method
2. ✅ Implement the _calculate_image_size helper method
3. ✅ Update the initialization code to use the calculated image size when auto_image_size is True
4. ✅ Add validation in the cell generation code (already present in the original code)
5. ✅ Update the command-line interface to support the new parameter
6. ✅ Update the documentation
7. ✅ Test the changes

## 6. Completion Summary

Date: 2023-07-15

The implementation has been completed successfully. All the planned changes have been made to the synthetic data generator:

1. Added the `auto_image_size` parameter to the `__init__` method
2. Implemented the `_calculate_image_size` helper method to calculate the appropriate image size
3. Updated the initialization code to use the calculated image size when `auto_image_size` is True
4. Verified that validation in the cell generation code was already present
5. Updated the command-line interface to support the new parameter
6. Updated the documentation to reflect the new feature

These changes should fix the issue with tile duplication in the stitched output by ensuring that the image size is large enough to accommodate all tiles with their proper overlap. The auto-calculation feature will make it easier to generate synthetic data with the correct image size, eliminating the need to manually calculate and specify the image size.
