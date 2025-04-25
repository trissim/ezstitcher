<div align="center">
  <img src="https://raw.githubusercontent.com/trissim/ezstitcher/main/docs/source/_static/ezstitcher_logo.png" alt="EZStitcher Logo" width="400"/>
</div>

# EZStitcher

[![PyPI version](https://badge.fury.io/py/ezstitcher.svg)](https://badge.fury.io/py/ezstitcher)
[![Documentation Status](https://readthedocs.org/projects/ezstitcher/badge/?version=latest)](https://ezstitcher.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://raw.githubusercontent.com/trissim/ezstitcher/main/.github/badges/coverage.svg)](https://trissim.github.io/ezstitcher/coverage/)

## Powerful Microscopy Image Processing Made Simple

EZStitcher is a high-performance Python library that transforms complex microscopy image processing into simple, intuitive workflows. Built on top of the robust [Ashlar](https://github.com/labsyspharm/ashlar) stitching engine, it provides a flexible pipeline architecture that makes processing large microscopy datasets effortless.

## 🚀 Key Features

- **Intelligent Z-Stack Processing**
  - Advanced focus detection and quality metrics
  - Multiple projection methods (max, mean, best-focus)
  - Per-plane 3D stitching support

- **Multi-Channel Excellence**
  - Process multiple fluorescence channels independently
  - Create channel-specific processing pipelines
  - Generate composite images with custom weighting

- **Automated Workflow Management**
  - Smart microscope format detection
  - Automatic directory management
  - Built-in multithreading support

- **Research-Ready Architecture**
  - Clean, object-oriented API
  - Extensible pipeline system
  - Seamless integration with other Python tools
  - Comprehensive testing suite

## 🎯 Supported Microscopes

- ImageXpress
- Opera Phenix
- Extensible architecture for adding new microscope types

## ⚡ Quick Start

```bash
# Install with pyenv (recommended)
pyenv install 3.11
pyenv global 3.11

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install EZStitcher
pip install ezstitcher
```

## 📊 Basic Usage

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.image_processor import ImageProcessor as IP
from pathlib import Path

# Initialize configuration and orchestrator
config = PipelineConfig(num_workers=2)  # Use 2 worker threads
orchestrator = PipelineOrchestrator(
    config=config,
    plate_path=Path("/path/to/plate")
)

# Define a complete processing pipeline
pipeline = Pipeline(
    input_dir=orchestrator.workspace_path,
    steps=[
        Step(
            name="Normalize Images",
            func=IP.stack_percentile_normalize
        ),
        PositionGenerationStep(),
        ImageStitchingStep()
    ],
    name="Basic Processing Pipeline"
)

# Execute with automatic directory management
success = orchestrator.run(pipelines=[pipeline])
```

## 📊 Advanced Usage Example

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.image_processor import ImageProcessor as IP
from ezstitcher.core.utils import stack
from n2v.models import N2V
from basicpy import BaSiC
from pathlib import Path
import numpy as np

# Custom processing functions
def n2v_process(images, model_path):
    """Apply Noise2Void denoising to images"""
    model = N2V(None, model_path, 'N2V')
    return [model.predict(img, 'N2V') for img in images]

def basic_process(images):
    """Apply BaSiC illumination correction"""
    basic = BaSiC()
    basic.fit(np.stack(images))
    return list(basic.transform(np.stack(images)))

def generate_position_pipeline(orchestrator, n2v_model_path):
    """Generate pipeline for position file creation"""
    return Pipeline(
        steps=[
            Step(func=IP.stack_percentile_normalize,
                 input_dir=orchestrator.workspace_path),
            Step(func=(IP.create_projection, {'method': 'max_projection'}),
                variable_components=['z_index']),
            Step(func=IP.create_composite,
                variable_components=['channel']),
            PositionGenerationStep()
        ])

def generate_stitching_pipeline(orchestrator, n2v_model_path):
    """Generate pipeline for image stitching"""
    return Pipeline(
        steps=[
            Step(func=(stack(n2v_process), {'model_path': n2v_model_path}),
                 input_dir=orchestrator.workspace_path),
            Step(func=stack(basic_process)),
            Step(func=IP.stack_percentile_normalize),
            Step(func=IP.stack_histogram_match),
            ImageStitchingStep(positions_file='positions.json')
        ])

# Process a plate with both pipelines
orchestrator.run(pipelines=[
    generate_position_pipeline(orchestrator, n2v_model_path),
    generate_stitching_pipeline(orchestrator, n2v_model_path)
])
```

## 📚 Documentation

Comprehensive documentation is available at [Read the Docs](https://ezstitcher.readthedocs.io/en/latest/), including:

- Detailed tutorials and examples
- Advanced usage patterns
- API reference
- Best practices
- Performance optimization guides

## 🤝 Contributing

We welcome contributions! Check out our [Contributing Guide](https://ezstitcher.readthedocs.io/en/latest/development/contributing.html) to get started.

## 📄 License

EZStitcher is released under the MIT License. See the LICENSE file for details.
