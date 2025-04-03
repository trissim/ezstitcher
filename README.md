# Axon Quant

A microscopy image stitching and processing tool for neuronal axon quantification.

## Features

- Microscopy image processing with various filters (blur, edge detection, tophat)
- Histogram matching and normalization for consistent imaging
- Image stitching with subpixel precision
- Support for multi-channel fluorescence microscopy
- Z-stack handling and organization
- Well and pattern detection for plate-based experiments

## Installation

```bash
# Clone the repository
git clone https://github.com/trissim/axon_quant.git
cd axon_quant

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install numpy scikit-image scipy pandas imageio tifffile ashlar
```

## Usage

```python
from stitcher_claude_v3 import process_plate_folder

# Process a single plate folder
process_plate_folder(
    'path/to/plate_folder',
    reference_channels=["1", "2"],
    composite_weights={"1": 0.1, "2": 0.9},
    preprocessing_funcs={"1": process_bf},
    tile_overlap=10
)
```

## File Structure

- `image_process.py`: Core image processing functions
- `stitcher_claude_v3.py`: Main stitching pipeline
- `z_stack_handler.py`: Z-stack organization and preprocessing

## Requirements

- Python 3.6+
- numpy
- scikit-image
- scipy
- pandas
- imageio
- tifffile
- ashlar

## License

MIT