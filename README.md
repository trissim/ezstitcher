# EZStitcher

An easy-to-use microscopy image stitching and processing tool for high-content imaging applications.

## Features

- Microscopy image processing with various filters (blur, edge detection, tophat)
- Histogram matching and normalization for consistent imaging
- Image stitching with subpixel precision 
- Z-stack handling with focus detection
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
ezstitcher /path/to/plate_folder --z-stack --focus-detect

# Process specific wells
ezstitcher /path/to/plate_folder --wells A01 B02 C03
```

## Python API Usage

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

## Package Structure

- `ezstitcher/core/image_process.py`: Core image processing functions
- `ezstitcher/core/stitcher.py`: Main stitching pipeline
- `ezstitcher/core/z_stack_handler.py`: Z-stack organization and preprocessing
- `ezstitcher/core/focus_detect.py`: Focus quality detection for Z-stacks

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

## License

MIT