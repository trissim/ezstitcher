# EZStitcher

An easy-to-use microscopy image stitching and processing tool for high-content imaging applications, currently optimized for ImageXpress microscopes with OperaPhenix support coming in future releases.

## Features

- Microscopy image processing with various filters
- Image stitching with subpixel precision
- Enhanced Z-stack handling with advanced focus detection
- 3D projections for Z-stack visualization (maximum, mean, etc.)
- Automatic best-focus plane detection across Z-stacks
- Support for multi-channel fluorescence microscopy
- Well and pattern detection for plate-based experiments
- Automatic metadata extraction from TIFF files
- No dependency on imagecodecs (uses uncompressed TIFF)
- Class-based architecture for improved code organization and modularity

## Supported Microscopes

- **ImageXpress**: Full support for all features
- **OperaPhenix**: Coming in future releases

## Installation

### Requirements

- **Python 3.11.9** (recommended for best compatibility with all dependencies)
- Git

### Linux/macOS

```bash
# Clone the repository
git clone https://github.com/trissim/ezstitcher.git
cd ezstitcher

# Create and activate a virtual environment with Python 3.11.9
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e .
```

### Windows

```powershell
# Clone the repository
git clone https://github.com/trissim/ezstitcher.git
cd ezstitcher

# Create and activate a virtual environment with Python 3.11.9
py -3.11 -m venv .venv
.venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

### Python Version Note

Python 3.11.9 is recommended because it provides the best compatibility with all required dependencies.

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

### Comprehensive Plate Processing

```python
from ezstitcher.core import process_plate_folder

# Process a single plate folder with all features
process_plate_folder(
    'path/to/plate_folder',
    reference_channels=["1", "2"],
    tile_overlap=10,
    max_shift=50,
    focus_detect=True,                # Enable best focus detection for Z-stacks
    focus_method="combined",          # Use combined focus metrics
    create_projections=True,          # Create Z-stack projections
    projection_types=["max", "mean"], # Types of projections to create
    stitch_method="best_focus"        # Use best focus images for stitching
)
```

### Basic Stitching (No Z-stacks)

```python
from ezstitcher.core import process_plate_folder

# Process a plate folder without Z-stack handling
process_plate_folder(
    'path/to/plate_folder',
    reference_channels=["1"],
    tile_overlap=10,
    max_shift=50
)
```

### Multi-Channel Reference Stitching

```python
from ezstitcher.core import process_plate_folder

# Process using multiple reference channels
process_plate_folder(
    'path/to/plate_folder',
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
- Standard deviation projection

### 4. Z-Aware Stitching

Stitch microscopy tiles with Z-awareness:
- Use best focused planes for alignment references
- Create consistent composite images from different wavelengths
- Generate positions from reference Z-planes

## Class-Based Architecture

EZStitcher uses a class-based architecture for better organization and modularity:

### Key Classes

- **ImageProcessor**: Handles all image processing operations
- **FocusDetector**: Handles focus detection algorithms
- **ZStackManager**: Manages Z-stack organization and processing
- **StitcherManager**: Handles image stitching operations

## Running Tests

EZStitcher includes a comprehensive test suite that verifies all core functionality:

```bash
# Make sure you're in the ezstitcher directory with your virtual environment activated
python -m pytest tests/test_synthetic_workflow_class_based.py -v
```

## Requirements

- Python 3.11.9 (recommended)
- numpy
- scikit-image
- scipy
- pandas
- tifffile
- ashlar
- opencv-python
- pytest (for running tests)

## License

MIT
