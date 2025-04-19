# Diff 3: Implement Processing Strategies

Status: Not Started
Progress: 0%
Last Updated: 2023-05-15
Dependencies: [plans/refactor/04-diff-02-concrete-steps.md]

## Overview

This diff implements the concrete processing strategies that will create and configure the pipeline. These strategies encapsulate the logic for creating a pipeline with the appropriate steps for different types of processing.

## Files to Create

### 1. `ezstitcher/core/processing/strategies/__init__.py`

```python
"""
Processing strategies for ezstitcher.

This module contains concrete implementations of processing strategies.
"""

from ezstitcher.core.processing.strategies.reference import ReferenceProcessingStrategy
from ezstitcher.core.processing.strategies.final import FinalProcessingStrategy

__all__ = [
    'ReferenceProcessingStrategy',
    'FinalProcessingStrategy',
]
```

### 2. `ezstitcher/core/processing/strategies/reference.py`

```python
"""
Reference processing strategy for processing pipeline.
"""

from typing import Dict, List, Callable, Optional, Any

from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.pipeline import ProcessingPipeline
from ezstitcher.core.processing.steps import ZStackFlatteningStep, ChannelProcessingStep, CompositeCreationStep


class ReferenceProcessingStrategy:
    """Strategy for processing reference images."""

    def __init__(self, config: PipelineConfig, image_preprocessor: ImagePreprocessor, focus_analyzer: FocusAnalyzer):
        """
        Initialize the reference processing strategy.

        Args:
            config: Pipeline configuration
            image_preprocessor: Image preprocessor
            focus_analyzer: Focus analyzer
        """
        self.config = config
        self.image_preprocessor = image_preprocessor
        self.focus_analyzer = focus_analyzer

    def create_pipeline(self, context: ProcessingContext) -> ProcessingPipeline:
        """
        Create a processing pipeline for the given context.

        Args:
            context: Processing context object

        Returns:
            Processing pipeline
        """
        # Create pipeline
        pipeline = ProcessingPipeline()

        # Get reference processing functions from config
        processing_funcs = {}
        for channel in context.channels:
            channel_funcs = context.pipeline_orchestrator._get_processing_functions(
                getattr(self.config, 'reference_processing', None),
                channel
            )
            if channel_funcs:
                processing_funcs[channel] = channel_funcs

        # Add Z-stack flattening step if needed
        if self.config.reference_flatten:
            pipeline.add_step(ZStackFlatteningStep(
                self.config.reference_flatten,
                self.focus_analyzer
            ))

        # Add channel processing step
        pipeline.add_step(ChannelProcessingStep(processing_funcs))

        # Add composite creation step
        pipeline.add_step(CompositeCreationStep(
            self.config.reference_composite_weights
        ))

        return pipeline
```

### 3. `ezstitcher/core/processing/strategies/final.py`

```python
"""
Final processing strategy for processing pipeline.
"""

from typing import Dict, List, Callable, Optional, Any

from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.pipeline import ProcessingPipeline
from ezstitcher.core.processing.steps import ZStackFlatteningStep, ChannelProcessingStep


class FinalProcessingStrategy:
    """Strategy for processing final images."""

    def __init__(self, config: PipelineConfig, image_preprocessor: ImagePreprocessor, focus_analyzer: FocusAnalyzer):
        """
        Initialize the final processing strategy.

        Args:
            config: Pipeline configuration
            image_preprocessor: Image preprocessor
            focus_analyzer: Focus analyzer
        """
        self.config = config
        self.image_preprocessor = image_preprocessor
        self.focus_analyzer = focus_analyzer

    def create_pipeline(self, context: ProcessingContext) -> ProcessingPipeline:
        """
        Create a processing pipeline for the given context.

        Args:
            context: Processing context object

        Returns:
            Processing pipeline
        """
        # Create pipeline
        pipeline = ProcessingPipeline()

        # Get final processing functions from config
        processing_funcs = {}
        for channel in context.channels:
            channel_funcs = context.pipeline_orchestrator._get_processing_functions(
                getattr(self.config, 'final_processing', None),
                channel
            )
            if channel_funcs:
                processing_funcs[channel] = channel_funcs
            else:
                processing_funcs[channel] = []

        # Add channel processing step
        pipeline.add_step(ChannelProcessingStep(processing_funcs))

        # Add Z-stack flattening step if needed
        if self.config.stitch_flatten:
            pipeline.add_step(ZStackFlatteningStep(
                self.config.stitch_flatten,
                self.focus_analyzer
            ))

        return pipeline
```

## Implementation Notes

1. The `ReferenceProcessingStrategy` class encapsulates the logic for creating a pipeline for processing reference images. It creates a pipeline with the following steps:
   - Z-stack flattening (if needed)
   - Channel processing
   - Composite creation

2. The `FinalProcessingStrategy` class encapsulates the logic for creating a pipeline for processing final images. It creates a pipeline with the following steps:
   - Channel processing
   - Z-stack flattening (if needed)

3. Each strategy follows the same pattern:
   - Initialize with the required parameters (config, image preprocessor, focus analyzer)
   - Implement the `create_pipeline` method to create a pipeline for the given context
   - Configure the pipeline with the appropriate steps
   - Return the configured pipeline

4. The `__init__.py` file exports the public API of the strategies module.

## Testing Plan

1. Create unit tests for each strategy:
   - Test `ReferenceProcessingStrategy` with different configurations
   - Test `FinalProcessingStrategy` with different configurations

2. Ensure that the strategies can be imported from the module:
   ```python
   from ezstitcher.core.processing.strategies import ReferenceProcessingStrategy, FinalProcessingStrategy
   ```

## Validation Criteria

1. All unit tests pass
2. The strategies can be imported from the module
3. The strategies can be used to refactor the `PipelineOrchestrator` class in the next diff
