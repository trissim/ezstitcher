# PipelineOrchestrator

## Overview

The `PipelineOrchestrator` is the central coordinator in EZStitcher's pipeline architecture. It manages the execution of multiple pipelines across wells, handling plate and well detection, directory structure management, multithreaded execution, and error handling.

## Class Definition

```python
class PipelineOrchestrator:
    def __init__(self, plate_path=None, workspace_path=None, config=None, fs_manager=None, image_preprocessor=None, focus_analyzer=None):
        """
        Initialize the pipeline orchestrator.

        Args:
            plate_path: Path to the plate folder (optional, can be provided later in run())
            workspace_path: Path to the workspace folder (optional, defaults to plate_path.parent/plate_path.name_workspace)
            config: Pipeline configuration (optional, a default configuration will be created if not provided)
            fs_manager: File system manager (optional, a new instance will be created if not provided)
            image_preprocessor: Image preprocessor (optional, a new instance will be created if not provided)
            focus_analyzer: Focus analyzer (optional, a new instance will be created if not provided)
        """
        # ...
```

## Key Responsibilities

### 1. Plate and Well Detection

The orchestrator automatically detects the plate structure and available wells:

```python
def detect_plate_structure(self, plate_path: Union[str, Path]) -> None:
    """
    Detect the plate structure and available wells.

    Args:
        plate_path: Path to the plate folder
    """
    # ...
```

### 2. Directory Structure Management

The orchestrator creates a workspace with symlinks to protect original data. Directory paths are automatically resolved between steps, with each step's output directory becoming the next step's input directory.

> **Note:** The `setup_directories()` method has been removed. Directory paths are now automatically resolved between steps. See the [Directory Structure](../source/concepts/directory_structure.rst) documentation for details.

### 3. Multithreaded Execution

The orchestrator can process multiple wells in parallel using a thread pool:

```python
def run(self, plate_path: Optional[Union[str, Path]] = None, pipelines: Optional[List[Pipeline]] = None) -> bool:
    """
    Run the pipeline on the specified plate.

    Args:
        plate_path: Path to the plate folder (optional if provided in __init__)
        pipelines: List of pipelines to run (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    # ...
```

### 4. Pipeline Execution

The orchestrator executes pipelines for each well:

```python
def process_well(self, well: str, pipelines: List[Pipeline]) -> bool:
    """
    Process a single well with the specified pipelines.

    Args:
        well: Well identifier
        pipelines: List of pipelines to run

    Returns:
        bool: True if successful, False otherwise
    """
    # ...
```

## Configuration

The orchestrator is configured using a `PipelineConfig` object:

```python
from ezstitcher.core.config import PipelineConfig, StitcherConfig

config = PipelineConfig(
    reference_composite_weights={"1": 0.5, "2": 0.5},  # Use channels 1 and 2 with equal weights
    well_filter=["A01", "B02"],    # Only process these wells
    stitcher=StitcherConfig(
        tile_overlap=10.0,         # 10% overlap between tiles
        max_shift=50               # Maximum shift in pixels
    ),
    num_workers=4                  # Use 4 worker threads
)
```

## Example Usage

Here's a simple example of using the `PipelineOrchestrator` with custom pipelines:

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
from ezstitcher.core.utils import stack

# Create configuration
config = PipelineConfig(
    reference_composite_weights={"1": 1.0},  # Use channel 1 for position generation
    num_workers=2  # Use 2 worker threads
)

# Create orchestrator
# All parameters except plate_path are optional and will be created if not provided
orchestrator = PipelineOrchestrator(
    plate_path="path/to/plate",      # Path to the plate folder
    config=config                    # Pipeline configuration
)

# Create position generation pipeline
position_pipeline = Pipeline(
    steps=[
        # Step 1: Flatten Z-stacks
        Step(name="Z-Stack Flattening",
             func=(IP.create_projection, {'method': 'max_projection'}),  # Function tuple with parameters
             variable_components=['z_index'],
             input_dir=orchestrator.workspace_path),  # First step uses workspace_path
             # output_dir is automatically determined

        # Step 2: Process channels
        Step(name="Image Enhancement",
             func=[stack(IP.sharpen),
                  IP.stack_percentile_normalize],
             # input_dir is automatically set to previous step's output
             # output_dir is automatically determined
        ),

        # Step 3: Generate positions
        PositionGenerationStep(
            name="Generate Positions"
            # input_dir is automatically set to previous step's output
            # output_dir is automatically determined
        )
    ],
    name="Position Generation Pipeline"
)

# Create image assembly pipeline
assembly_pipeline = Pipeline(
    steps=[
        # Step 1: Flatten Z-stacks
        Step(name="Z-Stack Flattening",
             func=(IP.create_projection, {'method': 'max_projection'}),  # Function tuple with parameters
             variable_components=['z_index'],
             input_dir=orchestrator.workspace_path  # First step uses workspace_path
             # output_dir is automatically determined
        ),

        # Step 2: Process channels
        Step(name="Channel Processing",
             func=IP.stack_percentile_normalize,
             # input_dir is automatically set to previous step's output
             # output_dir is automatically determined
        ),

        # Step 3: Stitch images
        ImageStitchingStep(
            name="Stitch Images"
            # input_dir is automatically set to previous step's output
            # positions_dir is automatically determined
            # output_dir is automatically determined
        )
    ],
    name="Image Assembly Pipeline"
)

# Run the orchestrator with the pipelines
success = orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])
```

For more examples, see the [User Guide](../source/user_guide/index.rst) which contains comprehensive usage examples.

## Related Classes

- [Pipeline](pipeline.md): Documentation on the Pipeline class
- [Steps](steps.md): Documentation on the Step class and its subclasses
