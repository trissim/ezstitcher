# Diff 1: Create Core Interfaces and Abstract Classes

Status: Not Started
Progress: 0%
Last Updated: 2023-05-15
Dependencies: [plans/refactor/02-implementation-plan.md]

## Overview

This diff implements the core interfaces and abstract classes for the processing pipeline refactoring. These classes will serve as the foundation for the rest of the implementation.

## Files to Create

### 1. `ezstitcher/core/processing/__init__.py`

```python
"""
Processing module for ezstitcher.

This module contains classes for processing images in a pipeline.
"""

from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.pipeline import ProcessingPipeline
from ezstitcher.core.processing.step import ProcessingStep
from ezstitcher.core.processing.strategy import ProcessingStrategy

__all__ = [
    'ProcessingContext',
    'ProcessingPipeline',
    'ProcessingStep',
    'ProcessingStrategy',
]
```

### 2. `ezstitcher/core/processing/context.py`

```python
"""
Context object for processing operations.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional


class ProcessingContext:
    """Context object for processing operations."""

    def __init__(self, pipeline_orchestrator, well, dirs, channels=None):
        """
        Initialize the processing context.

        Args:
            pipeline_orchestrator: Pipeline orchestrator
            well: Well identifier
            dirs: Dictionary of directories
            channels: List of channels to process
        """
        self.pipeline_orchestrator = pipeline_orchestrator
        self.well = well
        self.dirs = dirs
        self.channels = channels or []
        self.results = {}

    @property
    def input_dir(self) -> Path:
        """Get the input directory."""
        return self.dirs['input']

    @property
    def output_dir(self) -> Path:
        """Get the output directory."""
        return self.dirs['processed']

    @property
    def post_processed_dir(self) -> Path:
        """Get the post-processed directory."""
        return self.dirs['post_processed']

    @property
    def positions_dir(self) -> Path:
        """Get the positions directory."""
        return self.dirs['positions']

    @property
    def stitched_dir(self) -> Path:
        """Get the stitched directory."""
        return self.dirs['stitched']

    def set_result(self, key: str, value: Any) -> None:
        """
        Set a result value.

        Args:
            key: Result key
            value: Result value
        """
        self.results[key] = value

    def get_result(self, key: str, default: Any = None) -> Any:
        """
        Get a result value.

        Args:
            key: Result key
            default: Default value if key is not found

        Returns:
            Result value or default
        """
        return self.results.get(key, default)
```

### 3. `ezstitcher/core/processing/step.py`

```python
"""
Abstract base class for processing steps.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from ezstitcher.core.processing.context import ProcessingContext


class ProcessingStep(ABC):
    """Abstract base class for processing steps in the pipeline."""

    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Process the input context and return the updated context.

        Args:
            context: Processing context object

        Returns:
            Updated context object
        """
        pass
```

### 4. `ezstitcher/core/processing/strategy.py`

```python
"""
Protocol for processing strategies.
"""

from typing import Protocol, runtime_checkable

from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.pipeline import ProcessingPipeline


@runtime_checkable
class ProcessingStrategy(Protocol):
    """Protocol for processing strategies."""

    def create_pipeline(self, context: ProcessingContext) -> ProcessingPipeline:
        """
        Create a processing pipeline for the given context.

        Args:
            context: Processing context object

        Returns:
            Processing pipeline
        """
        ...
```

### 5. `ezstitcher/core/processing/pipeline.py`

```python
"""
Pipeline for processing images.
"""

from typing import List

from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.step import ProcessingStep


class ProcessingPipeline:
    """Pipeline for processing images."""

    def __init__(self):
        """Initialize the pipeline."""
        self.steps = []

    def add_step(self, step: ProcessingStep) -> 'ProcessingPipeline':
        """
        Add a processing step to the pipeline.

        Args:
            step: Processing step to add

        Returns:
            Self for chaining
        """
        self.steps.append(step)
        return self

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Process the input context through all steps in the pipeline.

        Args:
            context: Processing context object

        Returns:
            Updated context object
        """
        for step in self.steps:
            context = step.process(context)
        return context
```

## Implementation Notes

1. The `ProcessingContext` class provides a clean interface for passing data between processing steps. It encapsulates the pipeline orchestrator, well, directories, channels, and results.

2. The `ProcessingStep` abstract base class defines the interface for processing steps. Each step must implement the `process` method, which takes a context object and returns an updated context object.

3. The `ProcessingStrategy` protocol defines the interface for processing strategies. Each strategy must implement the `create_pipeline` method, which takes a context object and returns a pipeline.

4. The `ProcessingPipeline` class provides a way to chain processing steps together. It has methods for adding steps and processing a context through all steps.

5. The `__init__.py` file exports the public API of the processing module.

## Testing Plan

1. Create unit tests for each class:
   - Test `ProcessingContext` properties and methods
   - Test `ProcessingPipeline` with mock steps
   - Test that `ProcessingStep` and `ProcessingStrategy` can be properly subclassed

2. Ensure that the classes can be imported from the module:
   ```python
   from ezstitcher.core.processing import ProcessingContext, ProcessingPipeline, ProcessingStep, ProcessingStrategy
   ```

## Validation Criteria

1. All unit tests pass
2. The classes can be imported from the module
3. The classes can be used to implement the concrete processing steps and strategies in the next diffs
