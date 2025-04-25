# EZStitcher

EZStitcher is a Python library designed to simplify the processing and stitching of microscopy images. It provides a flexible pipeline architecture that allows researchers to easily process large microscopy datasets, create composite images, flatten Z-stacks, and stitch tiled images together. The stitching process is powered by the robust [Ashlar](https://github.com/labsyspharm/ashlar) backend.

## Key Features

*   Multi-channel fluorescence support: Process and stitch multiple fluorescence channels
*   Z-stack Handling & Focus Detection: Process 3D image stacks with various projection methods, advanced focus detection, and support for per-plane 3D stitching.
*   Flexible Preprocessing: Apply custom image processing functions within the pipeline.
*   Microscope Support & Auto-Detection: Supports ImageXpress and Opera Phenix formats with automatic detection of microscope type and image organization.
*   Parallel Processing: Built-in multithreading support (`PipelineConfig`) for faster execution on multi-core systems.
*   Extensible & Integratable: Clean, object-oriented API facilitates customization and integration with other Python microscopy/image analysis packages.

## Installation

The package is typically installed by cloning the Git repository and running the following command within the repository's root directory (after setting up a suitable Python 3.11 environment and virtual environment):

```bash
pip install -e .
```

This command installs the package in editable mode and handles dependencies listed in `requirements.txt`.

## Basic Usage

The following example demonstrates a basic pipeline that normalizes images and then stitches them. Intermediate output directories for steps are managed automatically by default, typically within structured intermediate folders. The final pipeline output directory also defaults to a location based on the input plate name (e.g., `[plate_name]_stitched` next to the original plate folder). You can specify custom `output_dir` for individual steps or the overall pipeline if manual control over output locations is needed.

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.image_processor import ImageProcessor as IP
from pathlib import Path

# Create configuration (e.g., single-threaded)
config = PipelineConfig(num_workers=1)

# Path to your plate folder (replace with actual path)
plate_path = Path("/path/to/your/plate")

# Create orchestrator
orchestrator = PipelineOrchestrator(
    config=config,
    plate_path=plate_path
)

# Define the pipeline steps
# The final output directory defaults to '[plate_name]_stitched' next to the plate folder
pipeline = Pipeline(
    input_dir=orchestrator.workspace_path,  # Use workspace managed by orchestrator
    steps=[
        # Step 1: Normalize image intensities
        # Output is implicitly stored in an automatically managed intermediate directory
        Step(
            name="Normalize Images",
            func=IP.stack_percentile_normalize
        ),
        # Step 2: Generate positions for stitching (uses output from Step 1)
        PositionGenerationStep(),
        # Step 3: Stitch images (uses output from Step 2 by default)
        ImageStitchingStep()
    ],
    name="Basic Processing Pipeline"
)

# Run the pipeline
success = orchestrator.run(pipelines=[pipeline])


```

## Core Concepts

EZStitcher uses a hierarchical pipeline architecture: the `PipelineOrchestrator` coordinates plate-level operations and manages the execution of `Pipeline`s (sequences of processing `Step`s) across multiple wells.

## Documentation

For more detailed information, please refer to the full documentation hosted on [Read the Docs](https://ezstitcher.readthedocs.io/en/latest/) (replace with actual link if different). The source files for the documentation are located in the `docs/source` directory, with the main index page at `docs/source/index.rst`.