# EZStitcher

An easy-to-use microscopy image stitching and processing tool for high-content imaging applications.

## Features

- Microscopy image processing with various filters (blur, edge detection, tophat)
- Histogram matching and normalization for consistent imaging
- Image stitching with subpixel precision 
- Enhanced Z-stack handling with advanced focus detection
- 3D projections for Z-stack visualization (maximum, mean, etc.)
- Automatic best-focus plane detection across Z-stacks
- Support for multi-channel fluorescence microscopy
- Well and pattern detection for plate-based experiments
- Automatic metadata extraction from TIFF files

## Installation

```bash
# Clone the repository
git clone https://github.com/trissim/ezstitcher.git
cd ezstitcher

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

## Command Line Usage

```bash
# Process a plate folder
ezstitcher /path/to/plate_folder --reference-channels 1 2 --tile-overlap 10

# Process a plate folder with Z-stacks
ezstitcher /path/to/plate_folder --focus-detect --focus-method combined

# Create Z-stack projections
ezstitcher /path/to/plate_folder --create-projections --projection-types max,mean,std

# Full Z-stack workflow with best focus detection and stitching
ezstitcher /path/to/plate_folder --focus-detect --create-projections --stitch-method best_focus

# Process specific wells
ezstitcher /path/to/plate_folder --wells A01 B02 C03
```

## Python API Usage

### Basic Stitching

```python
from ezstitcher.core.stitcher import process_plate_folder
from ezstitcher.core.image_process import process_bf

# Process a single plate folder
process_plate_folder(
    'path/to/plate_folder',
    reference_channels=["1", "2"],
    composite_weights={"1": 0.1, "2": 0.9},
    preprocessing_funcs={"1": process_bf},
    tile_overlap=10
)
```

### Enhanced Z-Stack Processing

```python
from ezstitcher.core.z_stack_handler import modified_process_plate_folder

# Process a plate folder with Z-stacks
modified_process_plate_folder(
    'path/to/plate_folder',
    focus_detect=True,
    focus_method="combined",
    create_projections=True,
    projection_types=['max', 'mean', 'std'],
    stitch_z_reference='best_focus',
    reference_channels=["1", "2"],
    tile_overlap=10
)
```

## Z-Stack Processing Features

EZStitcher includes comprehensive support for processing Z-stack microscopy images:

### 1. Z-Stack Organization

Automatically detects and organizes Z-stack data from both:
- Folder-based organization (`ZStep_1`, `ZStep_2`, etc.)
- Filename-based organization with `_z001`, `_z002` suffixes

### 2. Best Focus Detection

Multiple algorithms for identifying the best focused plane in a Z-stack:
- Combined focus measure (weighted combination of multiple metrics)
- Normalized variance
- Laplacian energy (edge detection based)
- Tenengrad variance (gradient-based)
- FFT-based focus measures

### 3. 3D Projections

Create various projection types from Z-stacks:
- Maximum intensity projection
- Mean intensity projection
- Minimum intensity projection
- Standard deviation projection
- Sum projection

### 4. Z-Aware Stitching

Stitch microscopy tiles with Z-awareness:
- Use best focused planes for alignment references
- Create consistent composite images from different wavelengths
- Generate positions from reference Z-planes

## Testing Z-Stack Processing

### 1. Complete Z-Stack Workflow Test

For testing the full Z-stack workflow:

```bash
python test_z_stack_workflow.py /path/to/plate_folder --focus-method combined --stitch-method best_focus
```

This will:
- Organize Z-stack images if needed
- Find best focused images across all tiles
- Create Z-stack projections
- Stitch images using best focused planes for positioning

### 2. Focus Detection Analysis

For detailed focus quality analysis:

```bash
python utils/analyze_focus.py "/path/to/zstack/*.tif" --output focus_report.png
```

This will:
- Load all images matching the pattern
- Analyze focus quality using different methods
- Plot focus scores, best and worst images
- Save or display the results

### 3. Step-by-Step Z-Stack Processing

For demonstrating each processing step individually:

```bash
python test_z_stack_workflow.py /path/to/plate_folder --step-by-step \
    --focus-method combined --projection-types max,mean,std
```

This mode:
- Shows the results of each processing step individually
- Gives more detailed logging and feedback
- Helps understand the Z-stack workflow

## Package Structure

- `ezstitcher/core/image_process.py`: Core image processing functions
- `ezstitcher/core/stitcher.py`: Main stitching pipeline
- `ezstitcher/core/z_stack_handler.py`: Z-stack organization and processing
- `ezstitcher/core/focus_detect.py`: Focus quality detection algorithms

## Requirements

- Python 3.6+
- numpy
- scikit-image
- scipy
- pandas
- imageio
- tifffile
- ashlar
- opencv-python
- matplotlib (for visualization)

## License

MIT