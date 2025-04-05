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
- Synthetic microscopy data generation for testing
- Comprehensive test suite with code coverage analysis
- No dependency on imagecodecs (uses uncompressed TIFF)

## Installation

```bash
# Clone the repository
git clone https://github.com/trissim/ezstitcher.git
cd ezstitcher

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

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
ezstitcher /path/to/plate_folder --focus-detect --create-projections --stitch-z-reference best_focus

# Process specific wells
ezstitcher /path/to/plate_folder --wells A01 B02 C03

# Generate synthetic test data
generate-synthetic-data --output-dir test_data --grid-size 3 3 --wavelengths 2 --z-stack-levels 3

# Run tests with coverage analysis
python -m coverage run --source=ezstitcher -m unittest tests/test_synthetic_workflow.py
python -m coverage html
```

## Python API Usage

### Comprehensive Plate Processing

```python
from ezstitcher.core.stitcher import process_plate_folder
from ezstitcher.core.image_process import process_bf

# Process a single plate folder with all features
process_plate_folder(
    'path/to/plate_folder',
    reference_channels=["1", "2"],
    composite_weights={"1": 0.1, "2": 0.9},
    preprocessing_funcs={"1": process_bf},
    tile_overlap=10,
    max_shift=50,
    focus_detect=True,                # Enable best focus detection for Z-stacks
    focus_method="combined",          # Use combined focus metrics
    create_projections=True,          # Create Z-stack projections
    projection_types=["max", "mean"], # Types of projections to create
    stitch_z_reference="best_focus"   # Use best focus images for stitching
)
```

### Basic Stitching (No Z-stacks)

```python
from ezstitcher.core.stitcher import process_plate_folder

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
from ezstitcher.core.stitcher import process_plate_folder

# Process using multiple reference channels with custom weights
process_plate_folder(
    'path/to/plate_folder',
    reference_channels=["1", "2"],
    composite_weights={"1": 0.3, "2": 0.7},
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

## Testing

### Synthetic Data Generation

EZStitcher includes a synthetic microscopy data generator for testing purposes:

```python
from utils.generate_synthetic_data import SyntheticMicroscopyGenerator

# Create a generator for synthetic microscopy data
generator = SyntheticMicroscopyGenerator(
    output_dir="synthetic_data",
    grid_size=(3, 3),           # 3x3 grid (9 tiles)
    image_size=(1024, 1024),    # Image dimensions
    tile_size=(512, 512),       # Tile dimensions
    overlap_percent=10,         # Tile overlap
    stage_error_px=5,           # Simulated stage positioning error
    wavelengths=3,              # Number of wavelengths/channels
    z_stack_levels=5,           # Number of Z-stack levels
    num_cells=200,              # Number of synthetic cells
    random_seed=42              # For reproducibility
)

# Generate the dataset
generator.generate_dataset()
```

This will create a synthetic microscopy dataset with:
- Multiple wavelengths/channels
- Z-stacks with varying focus levels
- Overlapping tiles with realistic stage positioning errors
- Proper folder structure and file naming conventions
- All images saved without compression

### Comprehensive Testing

Run the comprehensive test suite that tests all core functionality:

```bash
python -m unittest tests/test_synthetic_workflow.py
```

This test:
- Generates synthetic microscopy data with Z-stacks
- Tests Z-stack detection and organization
- Tests best focus selection
- Tests projection creation
- Tests stitching with various reference methods

### Code Coverage Analysis

Run tests with code coverage analysis:

```bash
python -m coverage run --source=ezstitcher -m unittest tests/test_synthetic_workflow.py
python -m coverage html
```

This will:
- Run the tests with coverage analysis
- Generate an HTML report in the `htmlcov` directory
- Show which parts of the code are well-tested
- Identify areas that need more test coverage

## Package Structure

- `ezstitcher/core/image_process.py`: Core image processing functions
- `ezstitcher/core/stitcher.py`: Main stitching pipeline with comprehensive `process_plate_folder`
- `ezstitcher/core/z_stack_handler.py`: Z-stack organization and processing
- `ezstitcher/core/focus_detect.py`: Focus quality detection algorithms
- `utils/generate_synthetic_data.py`: Synthetic microscopy data generator
- `tests/test_synthetic_workflow.py`: Comprehensive test suite

## Requirements

- Python 3.8+
- numpy
- scikit-image
- scipy
- pandas
- tifffile
- ashlar
- opencv-python
- matplotlib (for visualization)
- coverage (for test coverage analysis)

## License

MIT