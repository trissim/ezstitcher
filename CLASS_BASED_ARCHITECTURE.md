# EZStitcher Class-Based Implementation

This branch contains a class-based implementation of the EZStitcher package. The goal is to improve code organization, reduce circular dependencies, break down monolithic functions, and improve modularity.

## Key Improvements

### 1. Class-Based Architecture

The code has been reorganized into classes with clear responsibilities:

- **ImageProcessor**: Handles all image processing operations
- **FocusDetector**: Handles focus detection algorithms
- **ZStackManager**: Manages Z-stack organization and processing
- **StitcherManager**: Handles image stitching operations

### 2. Reduced Circular Imports

The circular dependencies between modules have been eliminated by:

- Moving common utility functions to a dedicated `utils.py` module
- Using late imports where necessary
- Organizing code into classes with clear dependencies

### 3. Broken Down Monolithic Functions

Large functions have been split into smaller, more focused methods:

- `process_plate_folder` has been broken down into multiple methods in the `StitcherManager` class
- Z-stack handling has been moved to dedicated methods in the `ZStackManager` class
- Image processing operations are now separate methods in the `ImageProcessor` class

### 4. Improved Modularity

Components are now more independent and can be used separately:

- Image processing can be used without stitching
- Focus detection can be used without Z-stack handling
- Z-stack handling can be used without stitching

## Backward Compatibility

The original API is still available for backward compatibility:

```python
from ezstitcher.core import process_plate_folder

process_plate_folder(
    'path/to/plate_folder',
    reference_channels=["1", "2"],
    composite_weights={"1": 0.1, "2": 0.9},
    preprocessing_funcs={"1": process_bf},
    tile_overlap=10,
    max_shift=50,
    focus_detect=True,
    focus_method="combined",
    create_projections=True,
    projection_types=["max", "mean"],
    stitch_z_reference="best_focus"
)
```

## New Class-Based API

The new class-based API provides more flexibility and control:

```python
from ezstitcher.core import (
    ImageProcessor,
    FocusDetector,
    ZStackManager,
    StitcherManager
)

# Preprocess the plate folder
has_zstack, z_info = ZStackManager.preprocess_plate_folder(plate_folder)

# Create projections
projections = ZStackManager.create_zstack_projections(
    input_dir,
    output_dir,
    projection_types=["max", "mean"]
)

# Find best focused images
best_focus_results = ZStackManager.select_best_focus_zstack(
    input_dir,
    output_dir,
    focus_method="combined",
    focus_wavelength="1"
)

# Process the plate folder
StitcherManager.process_plate_folder(
    plate_folder,
    reference_channels=["1", "2"],
    composite_weights={"1": 0.3, "2": 0.7},
    preprocessing_funcs={"1": ImageProcessor.process_bf},
    tile_overlap=10,
    max_shift=50,
    focus_detect=True,
    focus_method="combined",
    create_projections=True,
    projection_types=["max", "mean"],
    stitch_z_reference="best_focus"
)
```

## Examples

See the `examples/class_based_example.py` file for a complete example of using the new class-based API.

## Future Improvements

- Add more unit tests for the new classes
- Implement additional image processing algorithms
- Add support for more focus detection methods
- Improve error handling and logging
- Add more documentation and examples
