# Pipeline

## Overview

The `Pipeline` class represents a sequence of processing steps that are executed in order. It provides methods for managing steps, passing context between steps, and handling input/output directories.

## Class Definition

```python
class Pipeline:
    """
    A sequence of processing steps.

    A Pipeline encapsulates a sequence of processing steps that are executed
    in order. Each step can modify the processing context, which is passed
    from step to step.
    """

    def __init__(self, steps: List[Step] = None, name: str = None):
        """
        Initialize a pipeline.

        Args:
            steps: Initial list of steps
            name: Human-readable name for the pipeline
        """
        # ...
```

## Key Responsibilities

### 1. Step Management

The pipeline manages a sequence of processing steps:

```python
def add_step(self, step: Step) -> 'Pipeline':
    """
    Add a step to the pipeline.

    Args:
        step: The step to add

    Returns:
        Self, for method chaining
    """
    # ...
```

### 2. Pipeline Execution

The pipeline executes its steps in sequence, passing a context object between them:

```python
def run(self, input_dir=None, output_dir=None, well_filter=None, microscope_handler=None, orchestrator=None, positions_file=None):
    """
    Execute the pipeline.

    This method can either:
    1. Take individual parameters and create a ProcessingContext internally, or
    2. Take a pre-configured ProcessingContext object (when called from PipelineOrchestrator)

    The orchestrator parameter is required as it provides access to the microscope handler and other components.

    Args:
        input_dir: Optional input directory override
        output_dir: Optional output directory override
        well_filter: Optional well filter override
        microscope_handler: Optional microscope handler override
        orchestrator: Optional PipelineOrchestrator instance (required)
        positions_file: Optional positions file to use for stitching

    Returns:
        The results of the pipeline execution
    """
    # ...
```

### 3. Input/Output Directory Management

The pipeline can manage input and output directories for its steps:

```python
@property
def input_dir(self) -> Optional[Path]:
    """Get the input directory for the pipeline."""
    # ...

@input_dir.setter
def input_dir(self, value: Union[str, Path]) -> None:
    """Set the input directory for the pipeline."""
    # ...

@property
def output_dir(self) -> Optional[Path]:
    """Get the output directory for the pipeline."""
    # ...

@output_dir.setter
def output_dir(self, value: Union[str, Path]) -> None:
    """Set the output directory for the pipeline."""
    # ...
```

## Example Usage

Here's a simple example of creating and using a `Pipeline`:

```python
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep
from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
from ezstitcher.core.utils import stack
from pathlib import Path

# Create a pipeline with steps
pipeline = Pipeline(
    steps=[
        # Step 1: Flatten Z-stacks
        Step(name="Z-Stack Flattening",
             func=IP.create_projection,
             variable_components=['z_index'],
             processing_args={'method': 'max_projection'},
             input_dir=Path("path/to/input"),
             output_dir=Path("path/to/output")),

        # Step 2: Process channels
        Step(name="Image Enhancement",
             func=[stack(IP.sharpen),
                  IP.stack_percentile_normalize],
        ),

        # Step 3: Generate positions
        PositionGenerationStep(
            name="Generate Positions",
            output_dir=Path("path/to/positions")
        )
    ],
    name="Position Generation Pipeline"
)

# Create a processing context
from ezstitcher.core.pipeline import ProcessingContext
context = ProcessingContext(
    input_dir=Path("path/to/input"),
    output_dir=Path("path/to/output"),
    well_filter=["A01", "B02"],
    orchestrator=orchestrator,  # Reference to the PipelineOrchestrator
    # Additional attributes can be added as kwargs
    positions_file=Path("path/to/positions.csv"),
    custom_parameter=42
)

# Run the pipeline
result_context = pipeline.run(context)
```

For more examples, see the [User Guide](../source/user_guide/index.rst) which contains comprehensive usage examples.

## Related Classes

- [PipelineOrchestrator](pipeline_orchestrator.md): Documentation on the PipelineOrchestrator class
- [Steps](steps.md): Documentation on the Step class and its subclasses
