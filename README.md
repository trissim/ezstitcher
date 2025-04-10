# EZStitcher

An easy-to-use microscopy image stitching and processing tool for high-content imaging applications, optimized for ImageXpress and Opera Phenix microscopes.

[![Documentation Status](https://readthedocs.org/projects/ezstitcher/badge/?version=latest)](https://ezstitcher.readthedocs.io/en/latest/?badge=latest)

## Documentation

Full documentation is available at [https://ezstitcher.readthedocs.io/](https://ezstitcher.readthedocs.io/)

## Features

- Microscopy image processing with various filters
- Image stitching with subpixel precision
- Enhanced Z-stack handling with advanced focus detection
- 3D projections for Z-stack visualization (maximum, mean, etc.)
- Automatic best-focus plane detection across Z-stacks
- Per-plane Z-stack stitching using projection-derived positions
- Custom projection functions for Z-stack reference generation
- Support for multi-channel fluorescence microscopy
- Well and pattern detection for plate-based experiments
- Automatic metadata extraction from TIFF files
- No dependency on imagecodecs (uses uncompressed TIFF)
- Class-based architecture with instance methods for improved code organization and modularity

## Supported Microscopes

- **ImageXpress**: Full support for all features
- **Opera Phenix**: Full support for all features
- **Auto-detection**: Automatic detection of microscope type based on file patterns

See [Opera Phenix Support](docs/opera_phenix_support.md) for details on using EZStitcher with Opera Phenix data.
See [Auto-Detection](docs/auto_detection.md) for details on the microscope auto-detection feature.

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
# Basic usage (auto-detects microscope type)
ezstitcher /path/to/plate_folder

# Specify microscope type
ezstitcher /path/to/plate_folder --microscope-type ImageXpress

# Process Opera Phenix data
ezstitcher /path/to/plate_folder/Images --microscope-type OperaPhenix

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
from ezstitcher.core.main import process_plate_auto

# Process a single plate folder with all features (auto-detects microscope type)
process_plate_auto(
    'path/to/plate_folder',
    microscope_type="auto",  # This is the default, so you can omit it
    **{
        "reference_channels": ["1", "2"],
        "stitcher.tile_overlap": 10,
        "stitcher.max_shift": 50,
        "z_stack_processor.focus_detect": True,                # Enable best focus detection for Z-stacks
        "z_stack_processor.focus_method": "combined",          # Use combined focus metrics
        "z_stack_processor.create_projections": True,          # Create Z-stack projections
        "z_stack_processor.projection_types": ["max", "mean"], # Types of projections to create
        "z_stack_processor.stitch_z_reference": "max",         # Use max projection images for stitching
        "z_stack_processor.stitch_all_z_planes": True          # Stitch all Z-planes using projection-derived positions
    }
)
```

### Basic Stitching (No Z-stacks)

```python
from ezstitcher.core.main import process_plate_auto

# Process a plate folder without Z-stack handling
process_plate_auto(
    'path/to/plate_folder',
    **{
        "reference_channels": ["1"],
        "stitcher.tile_overlap": 10,
        "stitcher.max_shift": 50
    }
)
```

### Opera Phenix Support

```python
from ezstitcher.core.main import process_plate_auto

# Process Opera Phenix data (explicitly specify microscope type)
process_plate_auto(
    'path/to/opera_phenix_data',
    microscope_type='OperaPhenix',
    **{
        "reference_channels": ["1"],
        "stitcher.tile_overlap": 10.0
    }
)

# Auto-detect Opera Phenix data
process_plate_auto(
    'path/to/opera_phenix_data',
    # microscope_type defaults to 'auto'
    **{
        "reference_channels": ["1"],
        "stitcher.tile_overlap": 10.0
    }
)
```

### Multi-Channel Reference Stitching

```python
from ezstitcher.core.main import process_plate_auto

# Process using multiple reference channels
process_plate_auto(
    'path/to/plate_folder',
    **{
        "reference_channels": ["1", "2"],
        "stitcher.tile_overlap": 10
    }
)
```

### Z-Stack Per-Plane Stitching

```python
from ezstitcher.core.main import process_plate_auto

# Process Z-stack data with per-plane stitching
process_plate_auto(
    'path/to/plate_folder',
    **{
        "reference_channels": ["1"],
        "stitcher.tile_overlap": 10,
        "z_stack_processor.create_projections": True,          # Create projections for position detection
        "z_stack_processor.projection_types": ["max"],         # Use max projection
        "z_stack_processor.stitch_z_reference": "max",         # Use max projection for reference positions
        "z_stack_processor.stitch_all_z_planes": True          # Stitch each Z-plane using the same positions
    }
)
```

### Custom Z-Stack Projection Function

```python
from ezstitcher.core.main import process_plate_auto

# Define a custom function that takes a Z-stack and returns a 2D image
def middle_plane_projection(z_stack):
    """Custom projection function that takes the middle plane of the Z-stack."""
    import numpy as np
    # Convert to numpy array if it's a list
    if isinstance(z_stack, list):
        z_stack = np.array(z_stack)
    # Get the middle plane
    middle_idx = len(z_stack) // 2
    return z_stack[middle_idx]

# Process Z-stack data with custom projection function
process_plate_auto(
    'path/to/plate_folder',
    **{
        "reference_channels": ["1"],
        "stitcher.tile_overlap": 10,
        "z_stack_processor.create_projections": True,
        "z_stack_processor.stitch_z_reference": middle_plane_projection,  # Use custom function
        "z_stack_processor.stitch_all_z_planes": True
    }
)
```

### Advanced Usage with Configuration Objects

```python
from ezstitcher.core.config import (
    StitcherConfig,
    FocusAnalyzerConfig,
    ImagePreprocessorConfig,
    ZStackProcessorConfig,
    PlateProcessorConfig
)
from ezstitcher.core.plate_processor import PlateProcessor

# Create configuration objects
stitcher_config = StitcherConfig(
    tile_overlap=10.0,  # Percentage overlap between tiles
    max_shift=50,       # Maximum shift allowed between tiles in microns
    margin_ratio=0.1    # Blending margin ratio for stitching
)

focus_config = FocusAnalyzerConfig(
    method="combined"   # Focus detection method
)

# Define preprocessing functions
def enhance_contrast(img):
    """Simple contrast enhancement."""
    import numpy as np
    return np.clip(img * 1.2, 0, 255).astype(np.uint8)

preprocessing_funcs = {
    "1": enhance_contrast,  # Apply to channel 1
    "2": enhance_contrast   # Apply to channel 2
}

# Define composite weights
composite_weights = {
    "1": 0.7,  # 70% weight for channel 1
    "2": 0.3   # 30% weight for channel 2
}

image_config = ImagePreprocessorConfig(
    preprocessing_funcs=preprocessing_funcs,
    composite_weights=composite_weights
)

# Define a custom function for Z-stack projection
def weighted_projection(z_stack):
    """Custom weighted projection that emphasizes the middle planes."""
    import numpy as np
    if isinstance(z_stack, list):
        z_stack = np.array(z_stack)

    # Create weights that emphasize the middle planes
    num_planes = len(z_stack)
    middle = num_planes // 2
    weights = 1 - np.abs(np.arange(num_planes) - middle) / (num_planes / 2)

    # Apply weights and sum
    weighted_stack = z_stack * weights[:, np.newaxis, np.newaxis]
    return np.sum(weighted_stack, axis=0) / np.sum(weights)

zstack_config = ZStackProcessorConfig(
    focus_detect=True,           # Enable focus detection
    focus_method="combined",     # Focus detection method
    create_projections=True,     # Create projections from Z-stacks
    stitch_z_reference=weighted_projection,  # Use custom function for stitching reference
    save_projections=True,       # Save projection images
    stitch_all_z_planes=True,    # Stitch all Z-planes using projection-derived positions
    projection_types=["max", "mean"]  # Types of projections to create
)

plate_config = PlateProcessorConfig(
    reference_channels=["1", "2"],  # Use both channels as reference
    well_filter=["A01", "A02"],     # Only process these wells
    use_reference_positions=False,  # Generate new positions
    preprocessing_funcs=preprocessing_funcs,
    composite_weights=composite_weights,
    stitcher=stitcher_config,
    focus_analyzer=focus_config,
    image_preprocessor=image_config,
    z_stack_processor=zstack_config
)

# Create and run the plate processor
processor = PlateProcessor(plate_config)
processor.run("path/to/plate_folder")
```

### Opera Phenix Support

```python
from ezstitcher.core import process_plate_folder

# Process Opera Phenix data with auto-detection
process_plate_folder(
    'path/to/opera_phenix_data',
    reference_channels=["1"],
    tile_overlap=10.0
    # No need to specify microscope_type - it will be auto-detected
)

# Process Opera Phenix data with explicit format specification
process_plate_folder(
    'path/to/opera_phenix_data',
    reference_channels=["1"],
    tile_overlap=10.0,
    microscope_type='OperaPhenix'  # Explicitly specify Opera Phenix format
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
- Support for custom projection functions for reference generation

### 5. Per-Plane Z-Stack Stitching

Stitch each Z-plane independently while maintaining alignment:
- Use projection images (e.g., max projection) to generate tiling positions
- Apply the same positions to all Z-planes for consistent alignment
- Preserve 3D structure while creating stitched Z-stack volumes

## Class-Based Architecture

EZStitcher uses a class-based architecture with instance methods for better organization and modularity:

### Core Components

1. **PlateProcessor**: Main entry point for processing a plate folder. Coordinates the overall workflow.
2. **Stitcher**: Handles image stitching operations, including position detection and image assembly.
3. **ZStackDetector**: Detects Z-stack folders and images.
4. **ZStackOrganizer**: Organizes Z-stack folder structures and filenames.
5. **ZStackMetadata**: Aggregates metadata about wells, sites, channels, and Z-indices.
6. **ZStackFocusSelector**: Performs best focus detection on Z-stacks.
7. **ZStackProjector**: Generates projections (max, mean, custom) from Z-stacks.
8. **ReferenceProjectionGenerator**: Creates reference projections for stitching.
9. **PositionFileManager**: Manages CSV position files for stitching.
10. **ZPlaneStitcher**: Stitches images across Z-planes using position files.
11. **FileResolver**: Handles filename parsing, fallback logic, and image loading.
12. **ZStackProcessingPipeline**: Orchestrates the full Z-stack processing workflow.
13. **FocusAnalyzer**: Analyzes focus quality in Z-stack images.
14. **ImagePreprocessor**: Handles image preprocessing operations like contrast enhancement and composite creation.
15. **FileSystemManager**: Manages file system operations like finding files, creating directories, and cleaning up.

### Configuration Objects

Each component has a corresponding configuration object that encapsulates its settings:

1. **PlateProcessorConfig**: Configuration for the PlateProcessor, including directory naming conventions.
2. **StitcherConfig**: Configuration for the Stitcher, including tile overlap and margin settings.
3. **ZStackProcessorConfig**: Configuration for the ZStackProcessor, including focus detection, projection settings, and custom projection functions.
4. **FocusAnalyzerConfig**: Configuration for the FocusAnalyzer, including focus detection methods.
5. **ImagePreprocessorConfig**: Configuration for the ImagePreprocessor, including preprocessing functions and composite weights.

## Running Tests

EZStitcher includes a comprehensive test suite that verifies all core functionality:

```bash
# Make sure you're in the ezstitcher directory with your virtual environment activated
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

# Run all tests
python -m unittest discover -s tests

# Run specific test files
python -m unittest tests/test_file_system_manager.py
python -m unittest tests/test_stitcher.py
python -m unittest tests/test_zstack_processor.py
python -m unittest tests/test_integration.py

# Run synthetic workflow tests (these test the full pipeline)
python -m unittest tests/test_synthetic_workflow_class_based.py

# Run specific test methods
python -m unittest tests.test_synthetic_workflow_class_based.TestSyntheticWorkflowClassBased.test_zstack_projection_stitching
python -m unittest tests.test_synthetic_workflow_class_based.TestSyntheticWorkflowClassBased.test_zstack_per_plane_stitching
```

### Tested Features

The following features have been thoroughly tested and verified:

- Basic image stitching for non-Z-stack data
- Z-stack detection and organization
- Z-stack projection creation (max, mean)
- Z-stack projection-based stitching
- Z-stack per-plane stitching using projection-derived positions
- Multi-channel reference stitching
- File system operations and directory management

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

## Documentation

- [Opera Phenix Support](docs/opera_phenix_support.md)
- [Auto-Detection](docs/auto_detection.md)
- [Workflow Diagrams](docs/workflow_diagram.md)

## License

MIT
