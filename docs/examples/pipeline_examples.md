# Pipeline Examples

This document provides examples of using EZStitcher's pipeline architecture for common microscopy image processing tasks.

## Basic Pipeline Example

Here's a basic example of creating and running a pipeline:

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step
from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP

# Create configuration
config = PipelineConfig(
    reference_channels=["1"],
    num_workers=1  # Single-threaded for simplicity
)

# Create orchestrator
orchestrator = PipelineOrchestrator(config=config, plate_path="path/to/plate")

# Get directories
dirs = orchestrator.setup_directories()

# Create a simple pipeline
pipeline = Pipeline(
    steps=[
        # Step 1: Flatten Z-stacks
        Step(name="Z-Stack Flattening",
             func=IP.create_projection,
             variable_components=['z_index'],
             processing_args={'method': 'max_projection'},
             input_dir=dirs['input'],  
             output_dir=dirs['processed']),  

        # Step 2: Process channels
        Step(name="Channel Processing",
             func=IP.stack_percentile_normalize,
             variable_components=['channel'],
             group_by='channel'
        )
    ],
    name="Simple Processing Pipeline"
)

# Run the pipeline
success = orchestrator.run(pipelines=[pipeline])
```

## Position Generation and Stitching Example

Here's an example of creating pipelines for position generation and image stitching:

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

## Z-Stack Processing with Best Focus

Here's an example of processing Z-stacks with best focus detection:

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
    num_workers=2
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
        Step(name="Feature Enhancement",
             func=stack(IP.sharpen),
             variable_components=['site']),

        # Step 3: Generate positions
        PositionGenerationStep(
            name="Generate Positions",
            output_dir=dirs['positions']
        )
    ],
    name="Position Generation Pipeline"
)

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
             func=IP.create_projection,
             variable_components=['z_index'],
             processing_args={'method': 'best_focus'}),

        # Step 3: Stitch focused images
        ImageStitchingStep(
            name="Stitch Focused Images",
            positions_dir=dirs['positions'],
            output_dir=dirs['stitched']),
    ],
    name="Focused Image Assembly Pipeline"
)

# Run the orchestrator with the pipelines
success = orchestrator.run(pipelines=[position_pipeline, focus_pipeline])
```

## Channel-Specific Processing

Here's an example of applying different processing functions to different channels:

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step
from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
from ezstitcher.core.utils import stack

# Define channel-specific processing functions
def process_dapi(stack):
    """Process DAPI channel images."""
    stack = IP.stack_percentile_normalize(stack, low_percentile=0.1, high_percentile=99.9)
    return [IP.tophat(img) for img in stack]

def process_calcein(stack):
    """Process Calcein channel images."""
    return [IP.tophat(img) for img in stack]

# Create configuration
config = PipelineConfig(
    reference_channels=["1", "2"],
    num_workers=2
)

# Create orchestrator
orchestrator = PipelineOrchestrator(config=config, plate_path="path/to/plate")

# Get directories
dirs = orchestrator.setup_directories()

# Create pipeline with channel-specific processing
pipeline = Pipeline(
    steps=[
        # Step 1: Flatten Z-stacks
        Step(name="Z-Stack Flattening",
             func=IP.create_projection,
             variable_components=['z_index'],
             processing_args={'method': 'max_projection'},
             input_dir=dirs['input'],  
             output_dir=dirs['processed']),  

        # Step 2: Channel-specific processing
        Step(name="Channel Processing",
             func={"1": process_dapi, "2": process_calcein},  # Dictionary mapping channels to functions
             variable_components=['channel'],
             group_by='channel'  # Group by channel for channel-specific processing
        )
    ],
    name="Channel-Specific Processing Pipeline"
)

# Run the orchestrator with the pipeline
success = orchestrator.run(pipelines=[pipeline])
```

## More Examples

For more examples, see the integration tests in the `tests/integration` directory, particularly:

- `test_pipeline_architecture`: Basic pipeline architecture example
- `test_zstack_pipeline_architecture`: Z-stack processing example
- `test_zstack_pipeline_architecture_focus`: Z-stack processing with best focus example
