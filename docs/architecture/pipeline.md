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
def run(self, context: 'ProcessingContext') -> 'ProcessingContext':
    """
    Run the pipeline with the given context.

    This method executes each step in sequence, passing the context
    from step to step.

    Args:
        context: The processing context

    Returns:
        The updated processing context
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
from ezstitcher.core.processing_pipeline import ProcessingContext
context = ProcessingContext(
    well="A01",
    microscope_handler=microscope_handler,  # MicroscopeHandler instance
    input_dir=Path("path/to/input"),
    output_dir=Path("path/to/output")
)

# Run the pipeline
result_context = pipeline.run(context)
```

For more examples, see the integration tests in the `tests/integration` directory.

## Related Classes

- [PipelineOrchestrator](pipeline_orchestrator.md): Documentation on the PipelineOrchestrator class
- [Steps](steps.md): Documentation on the Step class and its subclasses
