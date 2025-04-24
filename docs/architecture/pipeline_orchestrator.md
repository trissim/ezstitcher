# PipelineOrchestrator

## Overview

The `PipelineOrchestrator` is the central coordinator in EZStitcher's pipeline architecture. It manages the execution of multiple pipelines across wells, handling plate and well detection, directory structure management, multithreaded execution, and error handling.

## Class Definition

```python
class PipelineOrchestrator:
    def __init__(self, config: PipelineConfig, plate_path: Optional[Union[str, Path]] = None):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Pipeline configuration
            plate_path: Path to the plate folder (optional, can be provided later in run())
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

The orchestrator creates and manages the directory structure for processing:

```python
def setup_directories(self) -> Dict[str, Path]:
    """
    Set up directory structure for processing.

    Returns:
        dict: Dictionary of directories
    """
    # ...
```

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
    reference_channels=["1", "2"],  # Use channels 1 and 2 as reference
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
    reference_channels=["1"],
    num_workers=2  # Use 2 worker threads
)

# Create orchestrator
orchestrator = PipelineOrchestrator(config=config, plate_path="path/to/plate")

# Get directories
dirs = orchestrator.setup_directories()

# Create position generation pipeline
position_pipeline = Pipeline(
    steps=[
        # Step 1: Flatten Z-stacks
        Step(name="Z-Stack Flattening",
             func=IP.create_projection,
             variable_components=['z_index'],
             processing_args={'method': 'max_projection'},
             input_dir=dirs['input'],  
             output_dir=dirs['processed']),  

        # Step 2: Process channels
        Step(name="Image Enhancement",
             func=[stack(IP.sharpen),
                  IP.stack_percentile_normalize],
        ),

        # Step 3: Generate positions
        PositionGenerationStep(
            name="Generate Positions",
            output_dir=dirs['positions']
        )
    ],
    name="Position Generation Pipeline"
)

# Create image assembly pipeline
assembly_pipeline = Pipeline(
    steps=[
        # Step 1: Flatten Z-stacks
        Step(name="Z-Stack Flattening",
             func=IP.create_projection,
             variable_components=['z_index'],
             processing_args={'method': 'max_projection'},
             input_dir=dirs['input'],
             output_dir=dirs['post_processed']
        ),

        # Step 2: Process channels
        Step(name="Channel Processing",
             func=IP.stack_percentile_normalize,
        ),

        # Step 3: Stitch images
        ImageStitchingStep(
            name="Stitch Images",
            positions_dir=dirs['positions'],
            output_dir=dirs['stitched']
        )
    ],
    name="Image Assembly Pipeline"
)

# Run the orchestrator with the pipelines
success = orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])
```

For more examples, see the integration tests in the `tests/integration` directory.

## Related Classes

- [Pipeline](pipeline.md): Documentation on the Pipeline class
- [Steps](steps.md): Documentation on the Step class and its subclasses
