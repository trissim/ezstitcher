<div align="center">
  <img src="https://raw.githubusercontent.com/trissim/ezstitcher/main/docs/source/_static/ezstitcher_logo.png" alt="EZStitcher Logo" width="400"/>
</div>

# EZStitcher

[![PyPI version](https://badge.fury.io/py/ezstitcher.svg)](https://badge.fury.io/py/ezstitcher)
[![Documentation Status](https://readthedocs.org/projects/ezstitcher/badge/?version=latest)](https://ezstitcher.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://raw.githubusercontent.com/trissim/ezstitcher/main/.github/badges/coverage.svg)](https://trissim.github.io/ezstitcher/coverage/)

## High-Performance Microscopy Image Stitching

EZStitcher is a Python library that simplifies complex microscopy image processing workflows. Built on the robust [Ashlar](https://github.com/labsyspharm/ashlar) stitching engine, it provides an intuitive pipeline architecture for processing large microscopy datasets.

## 🚀 Key Features

- **Simplified Interface for Non-Coders**
  - One-liner function for common workflows
  - Auto-detection of Z-stacks and channels

- **Intelligent Z-Stack Processing**
  - Advanced focus detection with quality metrics
  - Multiple projection methods (max, mean, best-focus)
  - Per-plane 3D stitching capability

- **Multi-Channel Support**
  - Independent fluorescence channel processing
  - Channel-specific processing pipelines
  - Custom weighted composite generation

- **Flexible Pipeline Architecture**
  - AutoPipelineFactory for quick workflow creation
  - Customizable processing steps
  - Modular design for easy extension

## 🎯 Supported Microscopes

- ImageXpress
- Opera Phenix
- Extensible architecture for additional microscope types

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

### Simplified Interface (Recommended for Beginners)

```python
from ezstitcher import stitch_plate

# Stitch a plate with a single function call
stitch_plate("path/to/microscopy/data")
```

### Using AutoPipelineFactory

```python
from ezstitcher.core import AutoPipelineFactory
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(plate_path="path/to/images")
factory = AutoPipelineFactory(input_dir=orchestrator.workspace_path)
pipelines = factory.create_pipelines()
orchestrator.run(pipelines=pipelines)
```

## 🔧 Advanced Usage Example

```python
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.specialized_steps import ZFlatStep, CompositeStep
from n2v.models import N2V
from basicpy import BaSiC
import numpy as np

def n2v_process(images, model_path):
    model = N2V(None, model_path, 'N2V')
    return [model.predict(img, 'N2V') for img in images]

def basic_process(images):
    basic = BaSiC()
    basic.fit(np.stack(images))
    return list(basic.transform(np.stack(images)))

# Position generation pipeline
position_pipeline = Pipeline(
    input_dir=orchestrator.workspace_path,
    steps=[
        ZFlatStep(),
        CompositeStep(),
        PositionGenerationStep()
    ]
)

# Assembly pipeline with advanced processing
assembly_pipeline = Pipeline(
    input_dir=orchestrator.workspace_path,
    steps=[
        # N2V denoising
        Step(func=(n2v_process, {'model_path': 'path/to/model.h5'})),

        # BaSiC flatfield correction
        Step(func=basic_process),

        # Normalize
        Step(func=IP.stack_percentile_normalize),

        # Stitch
        ImageStitchingStep()
    ]
)

orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])
```

## 📚 Documentation

Our comprehensive documentation is organized into several sections:

- [Quick Start Guide](https://ezstitcher.readthedocs.io/en/latest/getting_started/quick_start.html): Get up and running in minutes
- [Basic Usage](https://ezstitcher.readthedocs.io/en/latest/user_guide/basic_usage.html): Simple one-liner interface for common tasks
- [Intermediate Usage](https://ezstitcher.readthedocs.io/en/latest/user_guide/intermediate_usage.html): Custom pipelines with pre-built steps
- [Advanced Usage](https://ezstitcher.readthedocs.io/en/latest/user_guide/advanced_usage.html): Create custom steps and extend functionality
- [Core Concepts](https://ezstitcher.readthedocs.io/en/latest/concepts/index.html): Understanding EZStitcher's architecture
- [API Reference](https://ezstitcher.readthedocs.io/en/latest/api/index.html): Detailed API documentation

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](https://ezstitcher.readthedocs.io/en/latest/development/contributing.html) to get started.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
