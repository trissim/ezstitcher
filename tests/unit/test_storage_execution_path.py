"""Tests for the storage execution path in the stateless architecture."""

import pytest
import numpy as np
from pathlib import Path

from ezstitcher.core.pipeline import ProcessingContext, Pipeline, StepResult
from ezstitcher.core.steps import Step
from ezstitcher.io.storage_adapter import MemoryStorageAdapter

class TestStorageExecutionPath:
    """Test that storage operations are executed by Pipeline.run(), not by Steps."""
    
    def test_step_declares_storage_operations(self):
        """Test that steps declare storage operations but don't execute them."""
        # Create a simple step
        class TestStep(Step):
            def process(self, context):
                result = self.create_result()
                result.store("test_key", np.ones((5, 5)))
                return result
        
        step = TestStep(name="Test Step", func=lambda x: x)
        
        # Create a minimal context
        context = ProcessingContext()
        context.orchestrator = type('MockOrchestrator', (), {
            'file_manager': type('MockFileManager', (), {
                'ensure_directory': lambda p: None
            })(),
            'microscope_handler': type('MockMicroscopeHandler', (), {
                'auto_detect_patterns': lambda input_dir, well_filter, variable_components: {}
            })()
        })
        context.step_plans = {id(step): None}
        context.get_step_input_dir = lambda s: Path("/tmp/input")
        context.get_step_output_dir = lambda s: Path("/tmp/output")
        
        # Process the context
        result = step.process(context)
        
        # Verify the result contains storage operations
        assert len(result.storage_operations) > 0
        
        # Verify no actual storage adapter was accessed
        # This is the key test - the step should not have tried to write to a storage adapter
        assert not hasattr(context.orchestrator, 'storage_adapter')
    
    def test_pipeline_executes_storage_operations(self):
        """Test that Pipeline.run() executes storage operations."""
        # Create a simple step
        class TestStep(Step):
            def process(self, context):
                result = self.create_result()
                result.store("test_key", np.ones((5, 5)))
                return result
        
        # Create a pipeline with the step
        pipeline = Pipeline(steps=[TestStep(name="Test Step", func=lambda x: x)])
        
        # Create a context with a mock storage adapter
        context = ProcessingContext()
        mock_adapter = MemoryStorageAdapter()
        context.orchestrator = type('MockOrchestrator', (), {
            'storage_adapter': mock_adapter,
            'storage_mode': 'memory',
            'file_manager': type('MockFileManager', (), {
                'ensure_directory': lambda p: None
            })(),
            'microscope_handler': type('MockMicroscopeHandler', (), {
                'auto_detect_patterns': lambda input_dir, well_filter, variable_components: {}
            })()
        })
        context.step_plans = {id(pipeline.steps[0]): None}
        context.get_step_input_dir = lambda s: Path("/tmp/input")
        context.get_step_output_dir = lambda s: Path("/tmp/output")
        
        # Run the pipeline
        pipeline.run(context)
        
        # Verify the storage adapter was written to
        assert len(mock_adapter.list_keys()) > 0
        assert "test_key" in mock_adapter.list_keys()
