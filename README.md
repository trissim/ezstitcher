# EZStitcher

A modern, object-oriented microscopy image stitching and processing toolkit for high-content imaging applications, with comprehensive support for ImageXpress and Opera Phenix microscopes.

[![Documentation Status](https://readthedocs.org/projects/ezstitcher/badge/?version=latest)](https://ezstitcher.readthedocs.io/en/latest/?badge=latest)

## Overview

EZStitcher provides a robust framework for processing and stitching microscopy images, with special emphasis on handling multi-channel fluorescence data and Z-stacks. The library features a clean, modular architecture that makes it easy to extend and customize for specific research needs.

## Key Features

### Core Capabilities
- **High-precision image stitching** with subpixel alignment
- **Multi-channel support** for fluorescence microscopy
- **Automatic microscope type detection** for seamless workflow
- **Comprehensive Z-stack handling** with multiple processing options
- **Modular pipeline architecture** for customizable workflows

### Z-Stack Processing
- Advanced focus detection algorithms
- Multiple projection methods (max, mean, standard deviation)
- Per-plane Z-stack stitching with consistent alignment
- Custom projection function support

### Microscope Support
- **ImageXpress**: Full support with HTD metadata parsing
- **Opera Phenix**: Full support with XML metadata parsing
- Extensible architecture for adding new microscope types

### Architecture
- Clean object-oriented design with clear separation of concerns
- Composable pipeline components for flexible workflows
- Comprehensive configuration system with sensible defaults
- Robust error handling and logging

## Installation

### Requirements

- **Python 3.8-3.11** (3.11 recommended for best compatibility)
- Core dependencies are installed automatically with the package

### Recommended Installation with pyenv

```bash
# macOS
brew install pyenv

# Linux/WSL
curl https://pyenv.run | bash

# Setup Python environment
pyenv install 3.11
pyenv global 3.11

# Clone and install
git clone https://github.com/trissim/ezstitcher.git
cd ezstitcher
python -m venv .venv
source .venv/bin/activate  # Linux/macOS/WSL
# or
.venv\Scripts\activate     # Windows
python -m pip install -e .
```

For detailed installation instructions, including troubleshooting common issues, see the [Installation Guide](https://ezstitcher.readthedocs.io/en/latest/getting_started/installation.html).

## Usage Examples

### Quick Start

```python
# Simple function-based API for common tasks
from ezstitcher.core.main import process_plate

# Process a plate folder with automatic microscope detection
process_plate('path/to/plate_folder', reference_channels=["1"])
```

### Object-Oriented Pipeline

```python
from ezstitcher.core.config import PipelineConfig, StitcherConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Create configuration
config = PipelineConfig(
    reference_channels=["1", "2"],  # Use channels 1 and 2 as reference
    well_filter=["A01", "B02"],    # Only process these wells
    stitcher=StitcherConfig(
        tile_overlap=10.0,         # 10% overlap between tiles
        max_shift=50               # Maximum shift in pixels
    )
)

# Create and run the pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/plate_folder")
```

### Z-Stack Processing

```python
from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Configure Z-stack processing
config = PipelineConfig(
    reference_channels=["1"],
    reference_flatten="max",       # Use max projection for reference
    stitch_flatten="best_focus",   # Use best focus for final images
    focus_config=FocusAnalyzerConfig(
        method="combined",         # Combined focus metrics
        roi=(100, 100, 200, 200)  # Optional ROI for focus detection
    )
)

# Create and run the pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/plate_folder")
```

### Custom Image Preprocessing

```python
import numpy as np
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Define custom preprocessing functions
def enhance_contrast(image):
    """Enhance contrast using percentile normalization."""
    p_low, p_high = np.percentile(image, (2, 98))
    return np.clip((image - p_low) * (65535 / (p_high - p_low)), 0, 65535).astype(np.uint16)

# Configure with custom preprocessing
config = PipelineConfig(
    reference_channels=["1"],
    preprocessing_funcs={"1": enhance_contrast}
)

# Create and run the pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/plate_folder")
```

For more examples, including command-line usage and advanced configurations, see the [Examples](docs/examples.md) documentation.

## Architecture

EZStitcher is built on a modular, object-oriented architecture that separates concerns and enables flexible workflows.

### Core Components

<img src="docs/images/architecture.png" alt="EZStitcher Architecture" width="600"/>

- **PipelineOrchestrator**: Central coordinator that manages the entire processing workflow
- **MicroscopeHandler**: Handles microscope-specific functionality through composition
- **Stitcher**: Performs image stitching with subpixel precision
- **FocusAnalyzer**: Provides multiple focus detection algorithms for Z-stacks
- **ImagePreprocessor**: Handles image normalization, filtering, and compositing
- **FileSystemManager**: Manages file operations and directory structure
- **ImageLocator**: Locates and organizes images in various directory structures

### Configuration System

Each component has a corresponding configuration class that encapsulates its settings:

```python
from ezstitcher.core.config import PipelineConfig

# Create a configuration with sensible defaults
config = PipelineConfig()

# Override specific settings
config.reference_channels = ["1", "2"]
config.well_filter = ["A01", "B02"]
```

### Microscope Support

The architecture includes a plugin system for different microscope types:

- **FilenameParser**: Interface for parsing microscope-specific filenames
- **MetadataHandler**: Interface for extracting metadata from microscope files
- **MicroscopeHandler**: Composition-based handler that delegates to specific implementations

New microscope types can be added by implementing these interfaces.

## Development

### Running Tests

```bash
# Run all tests with pytest
python -m pytest

# Run tests with coverage report
python -m pytest --cov=ezstitcher --cov-report=html

# Run specific test modules
python -m pytest tests/integration/test_pipeline_orchestrator.py
```

### Contributing

Contributions are welcome! See the [Contributing Guide](docs/contributing.md) for details on how to contribute to EZStitcher.

### Documentation

The documentation is built with Sphinx and hosted on Read the Docs. To build the documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

Then open `docs/_build/html/index.html` in your browser.

## License

MIT
