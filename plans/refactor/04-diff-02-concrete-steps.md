# Diff 2: Implement Concrete Processing Steps

Status: Not Started
Progress: 0%
Last Updated: 2023-05-15
Dependencies: [plans/refactor/03-diff-01-core-interfaces.md]

## Overview

This diff implements the concrete processing steps that will be used in the pipeline. These steps encapsulate the specific processing operations that were previously hardcoded in the `process_reference_images` and `process_final_images` methods.

## Files to Create

### 1. `ezstitcher/core/processing/steps/__init__.py`

```python
"""
Processing steps for ezstitcher.

This module contains concrete implementations of processing steps.
"""

from ezstitcher.core.processing.steps.zstack import ZStackFlatteningStep
from ezstitcher.core.processing.steps.channel import ChannelProcessingStep
from ezstitcher.core.processing.steps.composite import CompositeCreationStep

__all__ = [
    'ZStackFlatteningStep',
    'ChannelProcessingStep',
    'CompositeCreationStep',
]
```

### 2. `ezstitcher/core/processing/steps/zstack.py`

```python
"""
Z-stack flattening step for processing pipeline.
"""

from typing import Optional

from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.step import ProcessingStep


class ZStackFlatteningStep(ProcessingStep):
    """Flattens Z-stacks using the specified method."""

    def __init__(self, method: str, focus_analyzer=None):
        """
        Initialize the Z-stack flattening step.

        Args:
            method: Flattening method (max_projection, mean_projection, best_focus)
            focus_analyzer: Focus analyzer for best_focus method
        """
        self.method = method
        self.focus_analyzer = focus_analyzer

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Process the input context and return the updated context.

        Args:
            context: Processing context object

        Returns:
            Updated context object
        """
        # Process Z-stacks
        result = context.pipeline_orchestrator.process_patterns_with_variable_components(
            input_dir=context.input_dir,
            output_dir=context.output_dir,
            well_filter=[context.well],
            variable_components=['z_index'],
            processing_funcs=context.pipeline_orchestrator.image_preprocessor.create_projection,
            processing_args={
                'method': self.method,
                'focus_analyzer': self.focus_analyzer
            }
        ).get(context.well, [])

        # Update context with result
        context.set_result('flattened_files', result)
        return context
```

### 3. `ezstitcher/core/processing/steps/channel.py`

```python
"""
Channel processing step for processing pipeline.
"""

from typing import Dict, List, Callable, Optional, Any

from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.step import ProcessingStep


class ChannelProcessingStep(ProcessingStep):
    """Processes images for each channel using the specified functions."""

    def __init__(self, processing_funcs: Optional[Dict[str, List[Callable]]] = None):
        """
        Initialize the channel processing step.

        Args:
            processing_funcs: Dictionary mapping channels to processing functions
        """
        self.processing_funcs = processing_funcs or {}

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Process the input context and return the updated context.

        Args:
            context: Processing context object

        Returns:
            Updated context object
        """
        # Process images
        result = context.pipeline_orchestrator.process_patterns_with_variable_components(
            input_dir=context.input_dir,
            output_dir=context.output_dir,
            well_filter=[context.well],
            variable_components=['site'],
            group_by='channel',
            processing_funcs=self.processing_funcs
        ).get(context.well, [])

        # Update context with result
        context.set_result('processed_files', result)
        return context
```

### 4. `ezstitcher/core/processing/steps/composite.py`

```python
"""
Composite creation step for processing pipeline.
"""

from typing import Dict, Optional

from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.step import ProcessingStep


class CompositeCreationStep(ProcessingStep):
    """Creates composite images from multiple channels."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the composite creation step.

        Args:
            weights: Dictionary mapping channels to weights
        """
        self.weights = weights

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Process the input context and return the updated context.

        Args:
            context: Processing context object

        Returns:
            Updated context object
        """
        # Create composites
        result = context.pipeline_orchestrator.process_patterns_with_variable_components(
            input_dir=context.input_dir,
            output_dir=context.output_dir,
            well_filter=[context.well],
            variable_components=['channel'],
            group_by='site',
            processing_funcs=context.pipeline_orchestrator.image_preprocessor.create_composite,
            processing_args={'weights': self.weights}
        ).get(context.well, [])

        # Update context with result
        context.set_result('composite_files', result)
        return context
```

## Implementation Notes

1. The `ZStackFlatteningStep` class encapsulates the logic for flattening Z-stacks using the specified method. It uses the `create_projection` method from the image preprocessor.

2. The `ChannelProcessingStep` class encapsulates the logic for processing images for each channel using the specified functions. It uses the `process_patterns_with_variable_components` method from the pipeline orchestrator.

3. The `CompositeCreationStep` class encapsulates the logic for creating composite images from multiple channels. It uses the `create_composite` method from the image preprocessor.

4. Each step follows the same pattern:
   - Initialize with the required parameters
   - Implement the `process` method to process the context
   - Update the context with the results
   - Return the updated context

5. The `__init__.py` file exports the public API of the steps module.

## Testing Plan

1. Create unit tests for each step:
   - Test `ZStackFlatteningStep` with different flattening methods
   - Test `ChannelProcessingStep` with different processing functions
   - Test `CompositeCreationStep` with different weights

2. Ensure that the steps can be imported from the module:
   ```python
   from ezstitcher.core.processing.steps import ZStackFlatteningStep, ChannelProcessingStep, CompositeCreationStep
   ```

## Validation Criteria

1. All unit tests pass
2. The steps can be imported from the module
3. The steps can be used to implement the processing strategies in the next diff
