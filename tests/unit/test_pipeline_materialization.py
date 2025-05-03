"""Tests for the Pipeline.run materialization triggers."""

import pytest
from unittest.mock import MagicMock, patch, call
import numpy as np
from pathlib import Path

from ezstitcher.core.pipeline import Pipeline, Step, ProcessingContext, StepResult
from ezstitcher.io.materialization import MaterializationManager, MaterializationPolicy


class TestPipelineMaterialization:
    """Tests for the Pipeline.run materialization triggers."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a step that needs materialization
        self.step_with_materialization = MagicMock(spec=Step)
        self.step_with_materialization.name = "Step With Materialization"
        self.step_with_materialization.needs_materialization.return_value = True
        self.step_with_materialization.process.return_value = StepResult()
        
        # Create a step that doesn't need materialization
        self.step_without_materialization = MagicMock(spec=Step)
        self.step_without_materialization.name = "Step Without Materialization"
        self.step_without_materialization.needs_materialization.return_value = False
        self.step_without_materialization.process.return_value = StepResult()
        
        # Create a pipeline with both steps
        self.pipeline = Pipeline(
            steps=[self.step_with_materialization, self.step_without_materialization],
            name="Test Pipeline"
        )
        
        # Create a mock orchestrator with materialization manager
        self.mock_orchestrator = MagicMock()
        self.mock_orchestrator.materialization_manager = MagicMock(spec=MaterializationManager)
        
        # Create a mock context
        self.mock_context = MagicMock(spec=ProcessingContext)
        self.mock_context.orchestrator = self.mock_orchestrator
        self.mock_context.well_filter = ["A01"]
        self.mock_context.get_step_input_dir.return_value = Path("/path/to/input")
        self.mock_context.get_step_output_dir.return_value = Path("/path/to/output")
        
    def test_pipeline_run_with_materialization(self):
        """Test Pipeline.run with materialization."""
        # Run the pipeline
        self.pipeline.run(self.mock_context)
        
        # Verify that needs_materialization was called for both steps
        self.step_with_materialization.needs_materialization.assert_called_once_with(self.mock_context)
        self.step_without_materialization.needs_materialization.assert_called_once_with(self.mock_context)
        
        # Verify that prepare_for_step was called only for the step that needs materialization
        self.mock_orchestrator.materialization_manager.prepare_for_step.assert_called_once_with(
            self.step_with_materialization, "A01", Path("/path/to/input")
        )
        
        # Verify that execute_pending_operations was called
        self.mock_orchestrator.materialization_manager.execute_pending_operations.assert_called_once()
        
        # Verify that process was called for both steps
        self.step_with_materialization.process.assert_called_once_with(self.mock_context)
        self.step_without_materialization.process.assert_called_once_with(self.mock_context)
        
        # Verify that cleanup_operations was called
        self.mock_orchestrator.materialization_manager.cleanup_operations.assert_called_once()
        
    def test_pipeline_run_without_materialization_manager(self):
        """Test Pipeline.run without materialization manager."""
        # Remove the materialization manager from the orchestrator
        self.mock_orchestrator.materialization_manager = None
        
        # Run the pipeline
        with patch('ezstitcher.io.materialization.MaterializationManager') as mock_manager_class:
            mock_manager = MagicMock(spec=MaterializationManager)
            mock_manager_class.return_value = mock_manager
            
            self.pipeline.run(self.mock_context)
            
            # Verify that MaterializationManager was created
            mock_manager_class.assert_called_once_with(self.mock_context)
            
            # Verify that needs_materialization was called for both steps
            self.step_with_materialization.needs_materialization.assert_called_once_with(self.mock_context)
            self.step_without_materialization.needs_materialization.assert_called_once_with(self.mock_context)
            
            # Verify that prepare_for_step was called only for the step that needs materialization
            mock_manager.prepare_for_step.assert_called_once_with(
                self.step_with_materialization, "A01", Path("/path/to/input")
            )
            
            # Verify that execute_pending_operations was called
            mock_manager.execute_pending_operations.assert_called_once()
            
            # Verify that process was called for both steps
            self.step_with_materialization.process.assert_called_once_with(self.mock_context)
            self.step_without_materialization.process.assert_called_once_with(self.mock_context)
            
            # Verify that cleanup_operations was called
            mock_manager.cleanup_operations.assert_called_once()
            
    def test_pipeline_run_without_well_filter(self):
        """Test Pipeline.run without well filter."""
        # Remove the well filter from the context
        self.mock_context.well_filter = []
        
        # Run the pipeline
        self.pipeline.run(self.mock_context)
        
        # Verify that needs_materialization was called for both steps
        self.step_with_materialization.needs_materialization.assert_called_once_with(self.mock_context)
        self.step_without_materialization.needs_materialization.assert_called_once_with(self.mock_context)
        
        # Verify that prepare_for_step was not called
        self.mock_orchestrator.materialization_manager.prepare_for_step.assert_not_called()
        
        # Verify that execute_pending_operations was not called
        self.mock_orchestrator.materialization_manager.execute_pending_operations.assert_not_called()
        
        # Verify that process was called for both steps
        self.step_with_materialization.process.assert_called_once_with(self.mock_context)
        self.step_without_materialization.process.assert_called_once_with(self.mock_context)
        
        # Verify that cleanup_operations was called
        self.mock_orchestrator.materialization_manager.cleanup_operations.assert_called_once()
        
    def test_pipeline_run_without_input_dir(self):
        """Test Pipeline.run without input directory."""
        # Make get_step_input_dir return None for the first step
        self.mock_context.get_step_input_dir.side_effect = [None, Path("/path/to/input")]
        
        # Run the pipeline
        self.pipeline.run(self.mock_context)
        
        # Verify that needs_materialization was called for both steps
        self.step_with_materialization.needs_materialization.assert_called_once_with(self.mock_context)
        self.step_without_materialization.needs_materialization.assert_called_once_with(self.mock_context)
        
        # Verify that prepare_for_step was not called
        self.mock_orchestrator.materialization_manager.prepare_for_step.assert_not_called()
        
        # Verify that execute_pending_operations was not called
        self.mock_orchestrator.materialization_manager.execute_pending_operations.assert_not_called()
        
        # Verify that process was called for both steps
        self.step_with_materialization.process.assert_called_once_with(self.mock_context)
        self.step_without_materialization.process.assert_called_once_with(self.mock_context)
        
        # Verify that cleanup_operations was called
        self.mock_orchestrator.materialization_manager.cleanup_operations.assert_called_once()
        
    def test_pipeline_run_with_step_result_update(self):
        """Test Pipeline.run with step result update."""
        # Create a step result with context updates
        step_result = StepResult()
        step_result.add_result("key1", "value1")
        step_result.update_context("attr1", "value1")
        
        # Make the first step return the step result
        self.step_with_materialization.process.return_value = step_result
        
        # Run the pipeline
        self.pipeline.run(self.mock_context)
        
        # Verify that update_from_step_result was called with the step result
        self.mock_context.update_from_step_result.assert_called_with(step_result)
        
    def test_pipeline_run_with_non_dict_step_result(self):
        """Test Pipeline.run with non-dict step result."""
        # Make the first step return a non-dict result
        self.step_with_materialization.process.return_value = "not a dict"
        
        # Run the pipeline
        with patch('ezstitcher.core.pipeline.logger') as mock_logger:
            self.pipeline.run(self.mock_context)
            
            # Verify that a warning was logged
            mock_logger.warning.assert_called_once()
            
            # Verify that update_from_step_result was not called for the first step
            assert self.mock_context.update_from_step_result.call_count == 1
            
    def test_pipeline_run_with_step_exception(self):
        """Test Pipeline.run with step exception."""
        # Make the first step raise an exception
        self.step_with_materialization.process.side_effect = Exception("Test exception")
        
        # Run the pipeline
        with pytest.raises(Exception):
            self.pipeline.run(self.mock_context)
            
        # Verify that cleanup_operations was called
        self.mock_orchestrator.materialization_manager.cleanup_operations.assert_called_once()
