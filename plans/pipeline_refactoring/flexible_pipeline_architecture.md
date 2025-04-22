# Flexible Pipeline Architecture

**Status**: In Progress
**Progress**: 0%
**Last Updated**: 2025-04-22
**Dependencies**: None

## 1. Overview

This plan outlines the implementation of a new Pipeline architecture for EZStitcher that builds on the strengths of the current `process_patterns_with_variable_components` method while adding an object-oriented core with a functional interface. The design aims to preserve the flexibility of the current implementation while making it more modular and expressive.

## 2. Goals

- Preserve the flexibility of the current `process_patterns_with_variable_components` method
- Support any grouping strategy and function type (single, list, dict)
- Create a modular, object-oriented core with a functional interface
- Enable pipeline composition and extension
- Maintain explicit data flow with input/output paths
- Minimize boilerplate code and unnecessary abstraction

## 3. Current Implementation Analysis

The current `process_patterns_with_variable_components` method has several strengths:

1. **Flexible Grouping**: Can process by any grouping strategy (`variable_components`)
2. **Function Flexibility**: Supports single functions, lists of functions, and dictionaries mapping channels to functions
3. **Result Organization**: Can group results by any dimension (`group_by`)
4. **Well Filtering**: Can filter by wells
5. **Argument Passing**: Can pass additional arguments to processing functions
6. **Pattern Preparation**: Uses `_prepare_patterns_and_functions` to handle different function types
7. **Tile Processing**: Uses `process_tiles` to apply functions to image stacks

These strengths should be preserved and enhanced in the new architecture. The current implementation already has a solid foundation for handling different function types and grouping strategies, which we'll build upon.

## 4. Core Architecture

### 4.1 Key Design Principles

1. **Preserve Flexibility**: Maintain the ability to process by any grouping with any function type
2. **Functional Interface**: Provide a simple, declarative API for defining pipelines
3. **Object-Oriented Core**: Use objects to encapsulate state and behavior
4. **Composition Over Inheritance**: Compose pipelines from simple steps
5. **Explicit Data Flow**: Make input/output paths clear

### 4.2 Core Components

#### `Step` Class
A container for a processing operation:

```python
class Step:
    """A processing step in a pipeline"""

    def __init__(
        self,
        func,
        variable_components=None,
        group_by=None,
        input_dir=None,
        output_dir=None,
        well_filter=None,
        processing_args=None,
        name=None
    ):
        """Initialize a processing step"""
        self.func = func
        self.variable_components = variable_components or []
        self.group_by = group_by
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.well_filter = well_filter
        self.processing_args = processing_args or {}
        self.name = name or self._generate_name()

    def _generate_name(self):
        """Generate a descriptive name based on the function"""
        # Implementation

    def process(self, context):
        """Process the step with the given context"""
        # Implementation that mirrors process_patterns_with_variable_components
```

#### `Pipeline` Class
A container for a sequence of steps:

```python
class Pipeline:
    """A sequence of processing steps"""

    def __init__(
        self,
        steps=None,
        input_dir=None,
        output_dir=None,
        well_filter=None,
        name=None
    ):
        """Initialize a pipeline"""
        self.steps = list(steps) if steps else []
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.well_filter = well_filter
        self.name = name or f"Pipeline({len(self.steps)} steps)"
        self._config = {}

    def add_step(self, step, output_dir=None):
        """Add a step to the pipeline"""
        # Implementation

    def set_input(self, input_dir):
        """Set the input directory"""
        # Implementation

    def set_output(self, output_dir):
        """Set the output directory"""
        # Implementation

    def set_well_filter(self, well_filter):
        """Set the well filter"""
        # Implementation

    def set_config(self, **kwargs):
        """Set configuration parameters"""
        # Implementation

    def clone(self):
        """Create a copy of this pipeline"""
        # Implementation

    def run(self, input_dir=None, output_dir=None, well_filter=None):
        """Execute the pipeline"""
        # Implementation
```

#### `ProcessingContext` Class
A container for processing state:

```python
class ProcessingContext:
    """Maintains state during pipeline execution"""

    def __init__(
        self,
        input_dir=None,
        output_dir=None,
        well_filter=None,
        config=None,
        **kwargs
    ):
        """Initialize the processing context"""
        # Implementation
```

#### Functional API
Simple functions that create steps and pipelines:

```python
def step(
    func,
    variable_components=None,
    group_by=None,
    **kwargs
):
    """Create a processing step"""
    return Step(
        func,
        variable_components,
        group_by,
        **kwargs
    )

def pipeline(*steps, **kwargs):
    """Create a processing pipeline"""
    if len(steps) == 1 and isinstance(steps[0], list):
        return Pipeline(steps[0], **kwargs)
    return Pipeline(steps, **kwargs)
```

## 5. Implementation Plan

### 5.1 Phase 1: Core Framework (Week 1)

1. Implement the `Step` class
   - Mirror the functionality of `process_patterns_with_variable_components`
   - Support all function types and grouping strategies
   - Maintain the same flexibility for processing arguments

2. Implement the `Pipeline` class
   - Container for steps with fluent interface
   - Methods for adding steps, setting paths, and configuration
   - Clone method for pipeline extension

3. Implement the `ProcessingContext` class
   - Container for processing state
   - Methods for file management and result organization

4. Implement the functional API
   - `step()` function for creating steps
   - `pipeline()` function for creating pipelines

### 5.2 Phase 2: Processing Logic (Week 2)

1. Implement the core processing logic in `Step.process()`
   - Mirror the logic in `process_patterns_with_variable_components`
   - Support all function types and grouping strategies
   - Maintain the same result organization options

2. Implement the pipeline execution logic in `Pipeline.run()`
   - Initialize context with input/output paths and well filter
   - Execute steps in sequence
   - Organize and return results

3. Create utility functions for common operations
   - Functions for file grouping
   - Functions for result organization
   - Functions for well filtering

### 5.3 Phase 3: Integration (Week 3)

1. Leverage existing components directly
   - Use `ImagePreprocessor` methods directly as processing functions
   - Use `Stitcher` methods directly for position generation and stitching
   - Use `FocusAnalyzer` directly for focus detection

2. Update `PipelineOrchestrator` to use the new architecture
   - Refactor `process_reference_images` to use pipelines
   - Refactor `process_final_images` to use pipelines
   - Refactor `generate_positions` to use pipelines
   - Refactor `stitch_images` to use pipelines
   - Update `process_well` to create and run pipelines and collect results

3. Implement backward compatibility
   - Keep the existing `process_patterns_with_variable_components` method
   - Implement it using the new Pipeline architecture internally
   - Ensure existing code continues to work without changes

### 5.4 Phase 4: Testing and Documentation (Week 4)

1. Create unit tests
   - Test step creation and processing
   - Test pipeline creation and execution
   - Test different function types and grouping strategies

2. Create integration tests
   - Test end-to-end workflows
   - Test with real image data

3. Update documentation
   - Create API documentation
   - Create usage examples
   - Update user guide

## 6. Detailed Implementation

### 6.1 Step Processing Logic

The core of the `Step.process()` method will mirror the logic in `process_patterns_with_variable_components`:

```python
def process(self, context):
    """Process the step with the given context"""

    # Use provided values or fall back to context values
    input_dir = self.input_dir or context.input_dir
    output_dir = self.output_dir or context.output_dir
    well_filter = self.well_filter or context.well_filter

    # Get file patterns with variable components
    patterns = get_file_patterns(
        input_dir,
        self.variable_components,
        well_filter
    )

    # Process each pattern
    results = {}

    for well, well_patterns in patterns.items():
        if well_filter and well not in well_filter:
            continue

        well_results = process_patterns(
            well_patterns,
            self.func,
            self.group_by,
            self.processing_args
        )

        results[well] = well_results

    # Update context with results
    context.results = results

    return context
```

### 6.2 Pattern Processing Logic

The `process_patterns` function will handle different function types and grouping strategies:

```python
def process_patterns(patterns, func, group_by=None, processing_args=None):
    """Process patterns with the given function"""

    processing_args = processing_args or {}

    # Group patterns if needed
    if group_by:
        grouped_patterns = group_patterns_by(patterns, group_by)
    else:
        grouped_patterns = {"all": patterns}

    # Process each group
    results = {}

    for group_key, group_patterns in grouped_patterns.items():
        # Handle different function types
        if isinstance(func, dict):
            # Function mapping (e.g., by channel)
            if group_by == 'channel' and group_key in func:
                # Direct mapping for this channel
                group_func = func[group_key]
                if isinstance(group_func, list):
                    # Apply multiple functions in sequence
                    result = group_patterns
                    for f in group_func:
                        result = f(result, **processing_args)
                    results[group_key] = result
                else:
                    # Apply single function
                    results[group_key] = group_func(group_patterns, **processing_args)
            else:
                # Apply different functions to different channels within this group
                group_results = {}
                for channel, channel_func in func.items():
                    channel_patterns = [p for p in group_patterns if channel in p]
                    if channel_patterns:
                        if isinstance(channel_func, list):
                            # Apply multiple functions in sequence
                            result = channel_patterns
                            for f in channel_func:
                                result = f(result, **processing_args)
                            group_results[channel] = result
                        else:
                            # Apply single function
                            group_results[channel] = channel_func(channel_patterns, **processing_args)
                results[group_key] = group_results
        elif isinstance(func, list):
            # Apply multiple functions in sequence
            result = group_patterns
            for f in func:
                result = f(result, **processing_args)
            results[group_key] = result
        else:
            # Apply single function
            results[group_key] = func(group_patterns, **processing_args)

    return results
```

### 6.3 Pipeline Execution Logic

The `Pipeline.run()` method will execute steps in sequence:

```python
def run(self, input_dir=None, output_dir=None, well_filter=None):
    """Execute the pipeline and return results"""

    # Use provided values or fall back to instance values
    effective_input = input_dir or self.input_dir
    effective_output = output_dir or self.output_dir
    effective_well_filter = well_filter or self.well_filter

    if not effective_input:
        raise ValueError("Input directory must be specified")

    # Initialize context
    context = ProcessingContext(
        input_dir=effective_input,
        output_dir=effective_output,
        well_filter=effective_well_filter,
        config=self._config
    )

    # Execute each step
    for step in self.steps:
        context = step.process(context)

    return context.results
```

### 6.4 Example Usage

```python
from ezstitcher.core.pipeline import step, pipeline
from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP

# Define processing functions
def dapi_process(images, **kwargs):
    """Process DAPI channel images"""
    # Implementation
    return processed_images

def calcein_process(images, **kwargs):
    """Process Calcein channel images"""
    # Implementation
    return processed_images

# Define paths
workspace_folder = "/path/to/workspace"
process_folder = "/path/to/processed"
positions_folder = "/path/to/positions"
stitched_folder = "/path/to/stitched"

# Create reference pipeline
reference_pipeline = (
    pipeline(
        # Flatten Z-stacks
        step(
            func=IP.create_projection,
            variable_components=['z_index'],
            processing_args={'method': 'max_projection'},
            name="Z-Stack Flattening"
        ),

        # Process channels
        step(
            func={"1": dapi_process, "2": calcein_process},
            variable_components=['site'],
            group_by='channel',
            name="Channel Processing"
        ),

        # Create composites
        step(
            func=IP.create_composite,
            variable_components=['channel'],
            group_by='site',
            processing_args={'weights': {"1": 0.7, "2": 0.3}},
            name="Composite Creation"
        )
    )
    .set_input(workspace_folder)
    .set_output(process_folder)
    .set_well_filter(["A01", "B02"])
)

# Create position generation pipeline
position_pipeline = (
    reference_pipeline.clone()
    .add_step(
        step(
            func=Stitcher.generate_positions,
            name="Position Generation"
        )
    )
    .set_output(positions_folder)
)

# Create assembly pipeline
assembly_pipeline = (
    pipeline(
        # Final tile enhancement
        step(
            func=[IP.stack_percentile_normalize, IP.stack_match_histogram, IP.tophat],
            variable_components=['site'],
            name="Final Tile Enhancement"
        ),

        # Image assembly
        step(
            func=Stitcher.assemble_image,
            name="Image Assembly"
        )
    )
    .set_input(workspace_folder)
    .set_output(stitched_folder)
    .set_well_filter(["A01", "B02"])
)

# Execute the pipelines
position_pipeline.run()
assembly_pipeline.run()
```

## 7. Integration with PipelineOrchestrator

The refactored `PipelineOrchestrator` will use the new Pipeline architecture:

```python
def process_reference_images(self, well, dirs):
    """Process reference images using the new Pipeline architecture"""

    # Create reference pipeline
    reference_pipeline = (
        pipeline(
            # Flatten Z-stacks
            step(
                func=self.image_preprocessor.create_projection,
                variable_components=['z_index'],
                processing_args={
                    'method': self.config.reference_flatten,
                    'focus_analyzer': self.focus_analyzer
                }
            ),

            # Process channels
            step(
                func=self.config.reference_processing,
                variable_components=['site'],
                group_by='channel'
            ),

            # Create composites
            step(
                func=self.image_preprocessor.create_composite,
                variable_components=['channel'],
                group_by='site',
                processing_args={'weights': self.config.reference_composite_weights}
            )
        )
        .set_input(dirs['input'])
        .set_output(dirs['processed'])
        .set_well_filter([well])
    )

    return reference_pipeline.run()
```

## 8. Benefits of This Approach

1. **Preserves Flexibility**: Maintains the power of `process_patterns_with_variable_components`
2. **Modular Design**: Encapsulates processing logic in reusable components
3. **Expressive API**: Makes pipeline structure clear and easy to understand
4. **Functional Interface**: Provides a simple, declarative API
5. **Object-Oriented Core**: Uses objects to encapsulate state and behavior

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Complex processing logic | Mirror existing logic that is already proven to work |
| Integration with existing components | Create adapter functions and thorough testing |
| Performance overhead | Benchmark and optimize critical paths |
| Learning curve for users | Provide clear documentation and examples |

## 10. Success Criteria

The refactoring will be considered successful if:

1. All existing functionality is preserved
2. The new API is more intuitive and requires less code
3. Test coverage is maintained or improved
4. Performance is comparable or better
5. The codebase is more maintainable and extensible

## 11. Future Extensions

Once the core architecture is in place, we can consider:

1. Adding more utility functions for common operations
2. Implementing pipeline visualization tools
3. Adding pipeline serialization/deserialization
4. Creating a GUI for pipeline construction
5. Supporting distributed processing for large datasets
