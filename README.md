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

### Pipeline Architecture

EZStitcher uses a flexible pipeline architecture composed of three main components:

1. **PipelineOrchestrator**: Coordinates the execution of multiple pipelines across wells
2. **Pipeline**: A sequence of processing steps
3. **Step**: A single processing operation (with specialized subclasses)

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
from ezstitcher.core.utils import stack

# Create configuration with worker settings
# - num_workers: Number of worker threads for parallel processing
# - reference_composite_weights: Weights for creating composite images for position generation
config = PipelineConfig(
    reference_composite_weights={"1": 1.0},  # Use channel 1 for position generation
    num_workers=2                           # Use 2 worker threads for parallel processing
)

# Create orchestrator with configuration and plate path
# The orchestrator manages the execution of pipelines across wells
orchestrator = PipelineOrchestrator(config=config, plate_path="path/to/plate")

# The orchestrator automatically creates a workspace directory with symlinks to protect original data
# Steps will create their output directories as needed

# Create position generation pipeline
# This pipeline processes images and generates position files for stitching
position_pipeline = Pipeline(
    steps=[
        # Step 1: Flatten Z-stacks using maximum projection
        # This reduces 3D z-stacks to 2D images for position calculation
        Step(name="Z-Stack Flattening",
             func=(IP.create_projection, {'method': 'max_projection'}),  # Function with parameters
             variable_components=['z_index'],  # Process each z-index separately
             input_dir=dirs['input'],
             output_dir=dirs['processed']),

        # Step 2: Enhance images for better feature detection
        # Apply sharpening followed by normalization to improve contrast
        Step(name="Image Enhancement",
             func=[stack(IP.sharpen),           # First sharpen the images
                  IP.stack_percentile_normalize],  # Then normalize intensity
        ),

        # Step 3: Generate position files for stitching
        # This specialized step calculates the relative positions of tiles
        PositionGenerationStep(
            output_dir=dirs['positions']  # Save position files here
        )
    ],
    name="Position Generation Pipeline"
)

# Create image assembly pipeline
# This pipeline processes and stitches images using the generated positions
assembly_pipeline = Pipeline(
    steps=[
        # Step 1: Flatten Z-stacks for final images
        # This reduces 3D z-stacks to 2D images for final stitching
        Step(name="Z-Stack Flattening",
             func=(IP.create_projection, {'method': 'max_projection'}),  # Function with parameters
             variable_components=['z_index'],  # Process each z-index separately
             input_dir=dirs['input'],
             output_dir=dirs['post_processed']
        ),

        # Step 2: Normalize channel intensities
        # This improves image quality and consistency across tiles
        Step(name="Channel Processing",
             func=IP.stack_percentile_normalize,  # Normalize intensity based on percentiles
        ),

        # Step 3: Stitch images using the generated positions
        # This specialized step combines tiles into a single large image
        ImageStitchingStep(
            name="Stitch Images",
            positions_dir=dirs['positions'],  # Use position files from here
            output_dir=dirs['stitched']       # Save stitched images here
        )
    ],
    name="Image Assembly Pipeline"
)

# Run the orchestrator with both pipelines
# The pipelines will be executed sequentially for each well
success = orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])
```

### Z-Stack Processing with Best Focus

```python
# Create best focus pipeline
focus_pipeline = Pipeline(
    steps=[
        # Step 1: Clean images for focus detection
        Step(name="Cleaning",
             func=[IP.tophat],
             input_dir=dirs['input'],
             output_dir=dirs['focus']),

        # Step 2: Apply best focus
        Step(name="Focus",
             func=(IP.create_projection, {'method': 'best_focus'}),
             variable_components=['z_index']),

        # Step 3: Stitch focused images
        ImageStitchingStep(
            positions_dir=dirs['positions'],
            output_dir=dirs['stitched']),
    ],
    name="Focused Image Assembly Pipeline"
)
```

### Channel-Specific Processing

```python
# Define channel-specific processing functions
def process_dapi(stack):
    """Process DAPI channel images."""
    stack = IP.stack_percentile_normalize(stack, low_percentile=0.1, high_percentile=99.9)
    return [IP.tophat(img) for img in stack]

def process_calcein(stack):
    """Process Calcein channel images."""
    return [IP.tophat(img) for img in stack]

# Create pipeline with channel-specific processing
pipeline = Pipeline(
    steps=[
        # Step with channel-specific processing
        Step(name="Channel Processing",
             func={"1": process_dapi, "2": process_calcein},  # Dictionary mapping channels to functions
             variable_components=['channel'],
             group_by='channel'  # Group by channel for channel-specific processing
        )
    ],
    name="Channel-Specific Processing Pipeline"
)
```

For more examples, see the [User Guide](docs/source/user_guide/index.rst) which contains comprehensive usage examples.

## Architecture

EZStitcher is built on a modular, object-oriented architecture that separates concerns and enables flexible workflows.

### Pipeline Architecture

The pipeline architecture is composed of three main components:

1. **PipelineOrchestrator**: Coordinates the execution of multiple pipelines across wells
2. **Pipeline**: A sequence of processing steps
3. **Step**: A single processing operation (with specialized subclasses)

This hierarchical design allows for complex workflows to be built from simple, reusable components.

```
┌─────────────────────────────────────────┐
│            PipelineOrchestrator         │
│                                         │
│  ┌─────────┐    ┌─────────┐             │
│  │ Pipeline│    │ Pipeline│    ...      │
│  │         │    │         │             │
│  │ ┌─────┐ │    │ ┌─────┐ │             │
│  │ │Step │ │    │ │Step │ │             │
│  │ └─────┘ │    │ └─────┘ │             │
│  │ ┌─────┐ │    │ ┌─────┐ │             │
│  │ │Step │ │    │ │Step │ │             │
│  │ └─────┘ │    │ └─────┘ │             │
│  │   ...   │    │   ...   │             │
│  └─────────┘    └─────────┘             │
└─────────────────────────────────────────┘
```

The pipeline architecture supports three patterns for processing functions:

1. **Single Function**: A callable that takes a list of images and returns a list of processed images
2. **List of Functions**: A sequence of functions applied one after another to the images
3. **Dictionary of Functions**: A mapping from component values (like channel numbers) to functions or lists of functions

### Core Components

- **MicroscopeHandler**: Handles microscope-specific functionality through composition
- **Stitcher**: Performs image stitching with subpixel precision
- **FocusAnalyzer**: Provides multiple focus detection algorithms for Z-stacks
- **ImagePreprocessor**: Handles image normalization, filtering, and compositing
- **FileSystemManager**: Manages file operations and directory structure
- **ImageLocator**: Locates and organizes images in various directory structures

### Configuration System

Each component has a corresponding configuration class that encapsulates its settings:

```python
from ezstitcher.core.config import PipelineConfig, StitcherConfig

# Create a configuration with sensible defaults
config = PipelineConfig(
    reference_composite_weights={"1": 0.5, "2": 0.5},  # Use channels 1 and 2 with equal weights
    num_workers=2,
    stitcher=StitcherConfig(
        tile_overlap=10.0,
        max_shift=50
    )
)
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
