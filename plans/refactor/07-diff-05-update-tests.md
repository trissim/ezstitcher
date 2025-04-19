# Diff 5: Update Tests

Status: Not Started
Progress: 0%
Last Updated: 2023-05-15
Dependencies: [plans/refactor/06-diff-04-refactor-orchestrator.md]

## Overview

This diff updates the existing tests to work with the refactored code and adds new tests for the new classes. The tests ensure that the refactored code works correctly and maintains backward compatibility.

## Files to Create

### 1. `tests/core/processing/test_context.py`

```python
"""
Tests for the ProcessingContext class.
"""

import pytest
from pathlib import Path

from ezstitcher.core.processing.context import ProcessingContext


class TestProcessingContext:
    """Tests for the ProcessingContext class."""

    def test_init(self):
        """Test initialization."""
        # Create a mock pipeline orchestrator
        pipeline_orchestrator = object()

        # Create a mock dirs dictionary
        dirs = {
            'input': Path('/input'),
            'processed': Path('/processed'),
            'post_processed': Path('/post_processed'),
            'positions': Path('/positions'),
            'stitched': Path('/stitched'),
        }

        # Create a context
        context = ProcessingContext(
            pipeline_orchestrator=pipeline_orchestrator,
            well='A01',
            dirs=dirs,
            channels=['1', '2']
        )

        # Check that the attributes are set correctly
        assert context.pipeline_orchestrator is pipeline_orchestrator
        assert context.well == 'A01'
        assert context.dirs is dirs
        assert context.channels == ['1', '2']
        assert context.results == {}

    def test_properties(self):
        """Test properties."""
        # Create a mock pipeline orchestrator
        pipeline_orchestrator = object()

        # Create a mock dirs dictionary
        dirs = {
            'input': Path('/input'),
            'processed': Path('/processed'),
            'post_processed': Path('/post_processed'),
            'positions': Path('/positions'),
            'stitched': Path('/stitched'),
        }

        # Create a context
        context = ProcessingContext(
            pipeline_orchestrator=pipeline_orchestrator,
            well='A01',
            dirs=dirs,
            channels=['1', '2']
        )

        # Check that the properties return the correct values
        assert context.input_dir == Path('/input')
        assert context.output_dir == Path('/processed')
        assert context.post_processed_dir == Path('/post_processed')
        assert context.positions_dir == Path('/positions')
        assert context.stitched_dir == Path('/stitched')

    def test_results(self):
        """Test result storage and retrieval."""
        # Create a mock pipeline orchestrator
        pipeline_orchestrator = object()

        # Create a mock dirs dictionary
        dirs = {
            'input': Path('/input'),
            'processed': Path('/processed'),
            'post_processed': Path('/post_processed'),
            'positions': Path('/positions'),
            'stitched': Path('/stitched'),
        }

        # Create a context
        context = ProcessingContext(
            pipeline_orchestrator=pipeline_orchestrator,
            well='A01',
            dirs=dirs,
            channels=['1', '2']
        )

        # Set a result
        context.set_result('test_key', 'test_value')

        # Check that the result is stored correctly
        assert context.results == {'test_key': 'test_value'}

        # Get the result
        result = context.get_result('test_key')

        # Check that the result is retrieved correctly
        assert result == 'test_value'

        # Get a non-existent result
        result = context.get_result('non_existent_key')

        # Check that the default value (None) is returned
        assert result is None

        # Get a non-existent result with a default value
        result = context.get_result('non_existent_key', 'default_value')

        # Check that the specified default value is returned
        assert result == 'default_value'
```

### 2. `tests/core/processing/test_pipeline.py`

```python
"""
Tests for the ProcessingPipeline class.
"""

import pytest
from unittest.mock import Mock

from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.pipeline import ProcessingPipeline
from ezstitcher.core.processing.step import ProcessingStep


class MockStep(ProcessingStep):
    """Mock processing step for testing."""

    def __init__(self, name):
        """Initialize with a name for identification."""
        self.name = name
        self.called = False

    def process(self, context):
        """Record that the step was called and return the context."""
        self.called = True
        context.set_result(f'step_{self.name}', True)
        return context


class TestProcessingPipeline:
    """Tests for the ProcessingPipeline class."""

    def test_init(self):
        """Test initialization."""
        # Create a pipeline
        pipeline = ProcessingPipeline()

        # Check that the steps list is empty
        assert pipeline.steps == []

    def test_add_step(self):
        """Test adding a step."""
        # Create a pipeline
        pipeline = ProcessingPipeline()

        # Create a mock step
        step = MockStep('test')

        # Add the step to the pipeline
        result = pipeline.add_step(step)

        # Check that the step was added
        assert pipeline.steps == [step]

        # Check that the method returns the pipeline for chaining
        assert result is pipeline

    def test_process(self):
        """Test processing a context through the pipeline."""
        # Create a pipeline
        pipeline = ProcessingPipeline()

        # Create mock steps
        step1 = MockStep('step1')
        step2 = MockStep('step2')
        step3 = MockStep('step3')

        # Add the steps to the pipeline
        pipeline.add_step(step1)
        pipeline.add_step(step2)
        pipeline.add_step(step3)

        # Create a mock context
        context = Mock(spec=ProcessingContext)
        context.get_result.return_value = None
        context.set_result.return_value = None

        # Process the context
        result = pipeline.process(context)

        # Check that all steps were called
        assert step1.called
        assert step2.called
        assert step3.called

        # Check that the result is the context
        assert result is context

        # Check that the steps were called in order
        context.set_result.assert_any_call('step_step1', True)
        context.set_result.assert_any_call('step_step2', True)
        context.set_result.assert_any_call('step_step3', True)
```

### 3. `tests/core/processing/steps/test_zstack.py`

```python
"""
Tests for the ZStackFlatteningStep class.
"""

import pytest
from unittest.mock import Mock, patch

from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.steps.zstack import ZStackFlatteningStep


class TestZStackFlatteningStep:
    """Tests for the ZStackFlatteningStep class."""

    def test_init(self):
        """Test initialization."""
        # Create a step
        step = ZStackFlatteningStep('max_projection')

        # Check that the attributes are set correctly
        assert step.method == 'max_projection'
        assert step.focus_analyzer is None

        # Create a step with a focus analyzer
        focus_analyzer = object()
        step = ZStackFlatteningStep('best_focus', focus_analyzer)

        # Check that the attributes are set correctly
        assert step.method == 'best_focus'
        assert step.focus_analyzer is focus_analyzer

    def test_process(self):
        """Test processing a context."""
        # Create a mock context
        context = Mock(spec=ProcessingContext)
        context.well = 'A01'
        context.input_dir = '/input'
        context.output_dir = '/output'
        context.pipeline_orchestrator = Mock()
        context.pipeline_orchestrator.process_patterns_with_variable_components.return_value = {'A01': ['file1.tif', 'file2.tif']}
        context.pipeline_orchestrator.image_preprocessor = Mock()

        # Create a step
        step = ZStackFlatteningStep('max_projection')

        # Process the context
        result = step.process(context)

        # Check that the process_patterns_with_variable_components method was called with the correct arguments
        context.pipeline_orchestrator.process_patterns_with_variable_components.assert_called_once_with(
            input_dir='/input',
            output_dir='/output',
            well_filter=['A01'],
            variable_components=['z_index'],
            processing_funcs=context.pipeline_orchestrator.image_preprocessor.create_projection,
            processing_args={
                'method': 'max_projection',
                'focus_analyzer': None
            }
        )

        # Check that the result was stored in the context
        context.set_result.assert_called_once_with('flattened_files', ['file1.tif', 'file2.tif'])

        # Check that the result is the context
        assert result is context
```

### 4. `tests/core/processing/steps/test_channel.py`

```python
"""
Tests for the ChannelProcessingStep class.
"""

import pytest
from unittest.mock import Mock, patch

from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.steps.channel import ChannelProcessingStep


class TestChannelProcessingStep:
    """Tests for the ChannelProcessingStep class."""

    def test_init(self):
        """Test initialization."""
        # Create a step
        step = ChannelProcessingStep()

        # Check that the attributes are set correctly
        assert step.processing_funcs == {}

        # Create a step with processing functions
        processing_funcs = {'1': [lambda x: x]}
        step = ChannelProcessingStep(processing_funcs)

        # Check that the attributes are set correctly
        assert step.processing_funcs is processing_funcs

    def test_process(self):
        """Test processing a context."""
        # Create a mock context
        context = Mock(spec=ProcessingContext)
        context.well = 'A01'
        context.input_dir = '/input'
        context.output_dir = '/output'
        context.pipeline_orchestrator = Mock()
        context.pipeline_orchestrator.process_patterns_with_variable_components.return_value = {'A01': ['file1.tif', 'file2.tif']}

        # Create a step
        processing_funcs = {'1': [lambda x: x]}
        step = ChannelProcessingStep(processing_funcs)

        # Process the context
        result = step.process(context)

        # Check that the process_patterns_with_variable_components method was called with the correct arguments
        context.pipeline_orchestrator.process_patterns_with_variable_components.assert_called_once_with(
            input_dir='/input',
            output_dir='/output',
            well_filter=['A01'],
            variable_components=['site'],
            group_by='channel',
            processing_funcs=processing_funcs
        )

        # Check that the result was stored in the context
        context.set_result.assert_called_once_with('processed_files', ['file1.tif', 'file2.tif'])

        # Check that the result is the context
        assert result is context
```

### 5. `tests/core/processing/steps/test_composite.py`

```python
"""
Tests for the CompositeCreationStep class.
"""

import pytest
from unittest.mock import Mock, patch

from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.steps.composite import CompositeCreationStep


class TestCompositeCreationStep:
    """Tests for the CompositeCreationStep class."""

    def test_init(self):
        """Test initialization."""
        # Create a step
        step = CompositeCreationStep()

        # Check that the attributes are set correctly
        assert step.weights is None

        # Create a step with weights
        weights = {'1': 0.7, '2': 0.3}
        step = CompositeCreationStep(weights)

        # Check that the attributes are set correctly
        assert step.weights is weights

    def test_process(self):
        """Test processing a context."""
        # Create a mock context
        context = Mock(spec=ProcessingContext)
        context.well = 'A01'
        context.input_dir = '/input'
        context.output_dir = '/output'
        context.pipeline_orchestrator = Mock()
        context.pipeline_orchestrator.process_patterns_with_variable_components.return_value = {'A01': ['file1.tif', 'file2.tif']}
        context.pipeline_orchestrator.image_preprocessor = Mock()

        # Create a step
        weights = {'1': 0.7, '2': 0.3}
        step = CompositeCreationStep(weights)

        # Process the context
        result = step.process(context)

        # Check that the process_patterns_with_variable_components method was called with the correct arguments
        context.pipeline_orchestrator.process_patterns_with_variable_components.assert_called_once_with(
            input_dir='/input',
            output_dir='/output',
            well_filter=['A01'],
            variable_components=['channel'],
            group_by='site',
            processing_funcs=context.pipeline_orchestrator.image_preprocessor.create_composite,
            processing_args={'weights': weights}
        )

        # Check that the result was stored in the context
        context.set_result.assert_called_once_with('composite_files', ['file1.tif', 'file2.tif'])

        # Check that the result is the context
        assert result is context
```

### 6. `tests/core/processing/strategies/test_reference.py`

```python
"""
Tests for the ReferenceProcessingStrategy class.
"""

import pytest
from unittest.mock import Mock, patch

from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.pipeline import ProcessingPipeline
from ezstitcher.core.processing.steps import ZStackFlatteningStep, ChannelProcessingStep, CompositeCreationStep
from ezstitcher.core.processing.strategies.reference import ReferenceProcessingStrategy


class TestReferenceProcessingStrategy:
    """Tests for the ReferenceProcessingStrategy class."""

    def test_init(self):
        """Test initialization."""
        # Create mock dependencies
        config = Mock(spec=PipelineConfig)
        image_preprocessor = Mock(spec=ImagePreprocessor)
        focus_analyzer = Mock(spec=FocusAnalyzer)

        # Create a strategy
        strategy = ReferenceProcessingStrategy(config, image_preprocessor, focus_analyzer)

        # Check that the attributes are set correctly
        assert strategy.config is config
        assert strategy.image_preprocessor is image_preprocessor
        assert strategy.focus_analyzer is focus_analyzer

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        # Create mock dependencies
        config = Mock(spec=PipelineConfig)
        config.reference_flatten = 'max_projection'
        config.reference_composite_weights = {'1': 0.7, '2': 0.3}
        config.reference_processing = {'1': [lambda x: x]}
        image_preprocessor = Mock(spec=ImagePreprocessor)
        focus_analyzer = Mock(spec=FocusAnalyzer)

        # Create a strategy
        strategy = ReferenceProcessingStrategy(config, image_preprocessor, focus_analyzer)

        # Create a mock context
        context = Mock(spec=ProcessingContext)
        context.channels = ['1', '2']
        context.pipeline_orchestrator = Mock()
        context.pipeline_orchestrator._get_processing_functions.return_value = [lambda x: x]

        # Create a pipeline
        pipeline = strategy.create_pipeline(context)

        # Check that the pipeline is a ProcessingPipeline
        assert isinstance(pipeline, ProcessingPipeline)

        # Check that the pipeline has the correct steps
        assert len(pipeline.steps) == 3
        assert isinstance(pipeline.steps[0], ZStackFlatteningStep)
        assert isinstance(pipeline.steps[1], ChannelProcessingStep)
        assert isinstance(pipeline.steps[2], CompositeCreationStep)

        # Check that the steps have the correct attributes
        assert pipeline.steps[0].method == 'max_projection'
        assert pipeline.steps[0].focus_analyzer is focus_analyzer
        assert pipeline.steps[1].processing_funcs == {'1': [lambda x: x], '2': [lambda x: x]}
        assert pipeline.steps[2].weights == {'1': 0.7, '2': 0.3}
```

### 7. `tests/core/processing/strategies/test_final.py`

```python
"""
Tests for the FinalProcessingStrategy class.
"""

import pytest
from unittest.mock import Mock, patch

from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.processing.context import ProcessingContext
from ezstitcher.core.processing.pipeline import ProcessingPipeline
from ezstitcher.core.processing.steps import ZStackFlatteningStep, ChannelProcessingStep
from ezstitcher.core.processing.strategies.final import FinalProcessingStrategy


class TestFinalProcessingStrategy:
    """Tests for the FinalProcessingStrategy class."""

    def test_init(self):
        """Test initialization."""
        # Create mock dependencies
        config = Mock(spec=PipelineConfig)
        image_preprocessor = Mock(spec=ImagePreprocessor)
        focus_analyzer = Mock(spec=FocusAnalyzer)

        # Create a strategy
        strategy = FinalProcessingStrategy(config, image_preprocessor, focus_analyzer)

        # Check that the attributes are set correctly
        assert strategy.config is config
        assert strategy.image_preprocessor is image_preprocessor
        assert strategy.focus_analyzer is focus_analyzer

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        # Create mock dependencies
        config = Mock(spec=PipelineConfig)
        config.stitch_flatten = 'max_projection'
        config.final_processing = {'1': [lambda x: x]}
        image_preprocessor = Mock(spec=ImagePreprocessor)
        focus_analyzer = Mock(spec=FocusAnalyzer)

        # Create a strategy
        strategy = FinalProcessingStrategy(config, image_preprocessor, focus_analyzer)

        # Create a mock context
        context = Mock(spec=ProcessingContext)
        context.channels = ['1', '2']
        context.pipeline_orchestrator = Mock()
        context.pipeline_orchestrator._get_processing_functions.side_effect = lambda funcs, channel: [lambda x: x] if channel == '1' else None

        # Create a pipeline
        pipeline = strategy.create_pipeline(context)

        # Check that the pipeline is a ProcessingPipeline
        assert isinstance(pipeline, ProcessingPipeline)

        # Check that the pipeline has the correct steps
        assert len(pipeline.steps) == 2
        assert isinstance(pipeline.steps[0], ChannelProcessingStep)
        assert isinstance(pipeline.steps[1], ZStackFlatteningStep)

        # Check that the steps have the correct attributes
        assert pipeline.steps[0].processing_funcs == {'1': [lambda x: x], '2': []}
        assert pipeline.steps[1].method == 'max_projection'
        assert pipeline.steps[1].focus_analyzer is focus_analyzer
```

### 8. `tests/core/test_processing_pipeline.py` (Update)

```diff
import pytest
from unittest.mock import Mock, patch

from ezstitcher.core.processing_pipeline import PipelineOrchestrator
+ from ezstitcher.core.processing.context import ProcessingContext

# ... (existing tests remain unchanged)

class TestPipelineOrchestrator:
    # ... (existing tests remain unchanged)

+   def test_process_reference_images(self):
+       """Test processing reference images."""
+       # Create a mock orchestrator
+       orchestrator = Mock(spec=PipelineOrchestrator)
+       orchestrator.config.reference_channels = ['1', '2']
+       orchestrator.process_reference_images = PipelineOrchestrator.process_reference_images.__get__(orchestrator)
+
+       # Create mock dependencies
+       dirs = {
+           'input': '/input',
+           'processed': '/processed',
+           'post_processed': '/post_processed',
+           'positions': '/positions',
+           'stitched': '/stitched',
+       }
+
+       # Process reference images
+       result = orchestrator.process_reference_images('A01', dirs)
+
+       # Check that the result is a ProcessingContext
+       assert isinstance(result, ProcessingContext)
+       assert result.well == 'A01'
+       assert result.dirs is dirs
+       assert result.channels == ['1', '2']
+
+   def test_process_final_images(self):
+       """Test processing final images."""
+       # Create a mock orchestrator
+       orchestrator = Mock(spec=PipelineOrchestrator)
+       orchestrator._get_available_channels.return_value = ['1', '2']
+       orchestrator.process_final_images = PipelineOrchestrator.process_final_images.__get__(orchestrator)
+
+       # Create mock dependencies
+       dirs = {
+           'input': '/input',
+           'processed': '/processed',
+           'post_processed': '/post_processed',
+           'positions': '/positions',
+           'stitched': '/stitched',
+       }
+
+       # Process final images
+       result = orchestrator.process_final_images('A01', dirs)
+
+       # Check that the result is a ProcessingContext
+       assert isinstance(result, ProcessingContext)
+       assert result.well == 'A01'
+       assert result.dirs is dirs
+       assert result.channels == ['1', '2']
+
+   def test_process_well(self):
+       """Test processing a well."""
+       # Create a mock orchestrator
+       orchestrator = Mock(spec=PipelineOrchestrator)
+       orchestrator.process_reference_images.return_value = Mock(spec=ProcessingContext)
+       orchestrator.process_final_images.return_value = Mock(spec=ProcessingContext)
+       orchestrator.generate_positions.return_value = ('/positions/A01.csv', 'pattern')
+       orchestrator.process_well = PipelineOrchestrator.process_well.__get__(orchestrator)
+
+       # Create mock dependencies
+       dirs = {
+           'input': '/input',
+           'processed': '/processed',
+           'post_processed': '/post_processed',
+           'positions': '/positions',
+           'stitched': '/stitched',
+       }
+
+       # Process a well
+       orchestrator.process_well('A01', dirs)
+
+       # Check that the methods were called with the correct arguments
+       orchestrator.process_reference_images.assert_called_once_with('A01', dirs)
+       orchestrator.generate_positions.assert_called_once_with('A01', dirs)
+       orchestrator.process_final_images.assert_called_once_with('A01', dirs)
+       orchestrator.stitch_images.assert_called_once_with('A01', dirs, '/positions/A01.csv')
```

## Implementation Notes

1. The tests for the new classes ensure that they work correctly and maintain backward compatibility.

2. The tests for `ProcessingContext` verify that the context object correctly stores and retrieves data.

3. The tests for `ProcessingPipeline` verify that the pipeline correctly processes a context through a sequence of steps.

4. The tests for the processing steps verify that they correctly process a context and update it with the results.

5. The tests for the processing strategies verify that they correctly create a pipeline with the appropriate steps.

6. The updated tests for `PipelineOrchestrator` verify that the refactored methods work correctly and maintain backward compatibility.

## Testing Plan

1. Run the new tests to ensure that the new classes work correctly.

2. Run the updated tests to ensure that the refactored code works correctly and maintains backward compatibility.

3. Run the existing integration tests to ensure that the refactored code works correctly with the rest of the system.

## Validation Criteria

1. All unit tests pass
2. All integration tests pass
3. The refactored code produces the same results as the original code
4. The code is more modular, flexible, and testable
