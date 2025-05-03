"""Integration tests for Pipeline with MaterializationManager."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.io.materialization import MaterializationManager, MaterializationPolicy
from ezstitcher.io.overlay import OverlayMode

# Import fixtures
from tests.fixtures.materialization_fixtures import (
    temp_test_dir, overlay_dir, mock_file_manager, mock_microscope_handler
)


class TestPipelineMaterialization:
    """Integration tests for Pipeline with MaterializationManager."""
    
    def test_pipeline_with_materialization_manager(self, temp_test_dir, overlay_dir):
        """Test a pipeline with MaterializationManager."""
        # Create a real PipelineOrchestrator
        orchestrator = PipelineOrchestrator(
            plate_path=temp_test_dir,
            workspace_path=temp_test_dir / "workspace",
            storage_mode="memory",
            overlay_mode=OverlayMode.AUTO,
            materialization_context="testing"
        )
        
        # Replace file_manager and microscope_handler with mocks
        orchestrator.file_manager = mock_file_manager
        orchestrator.microscope_handler = mock_microscope_handler
        
        # Create a simple test step
        class TestStep(Step):
            requires_fs_input = True
            
            def process(self, context):
                result = self.create_result()
                result.results['test'] = np.ones((10, 10))
                return result
        
        # Create a pipeline with the test step
        pipeline = Pipeline(
            steps=[TestStep()],
            name="Test Pipeline"
        )
        
        # Create a context
        context = orchestrator.create_context(pipeline, well_filter=["A01"])
        
        # Verify that the MaterializationManager was created
        assert hasattr(orchestrator, 'materialization_manager')
        assert isinstance(orchestrator.materialization_manager, MaterializationManager)
        
        # Mock the prepare_for_step and execute_pending_operations methods
        orchestrator.materialization_manager.prepare_for_step = MagicMock(return_value={})
        orchestrator.materialization_manager.execute_pending_operations = MagicMock(return_value=0)
        orchestrator.materialization_manager.cleanup_operations = MagicMock(return_value=0)
        
        # Run the pipeline
        result_context = pipeline.run(context)
        
        # Verify that the MaterializationManager methods were called
        orchestrator.materialization_manager.prepare_for_step.assert_called()
        orchestrator.materialization_manager.execute_pending_operations.assert_called()
        orchestrator.materialization_manager.cleanup_operations.assert_called()
        
        # Verify that the step was executed
        assert 'test' in result_context.results
        assert np.array_equal(result_context.results['test'], np.ones((10, 10)))
        
    def test_pipeline_with_multiple_steps(self, temp_test_dir, overlay_dir):
        """Test a pipeline with multiple steps and MaterializationManager."""
        # Create a real PipelineOrchestrator
        orchestrator = PipelineOrchestrator(
            plate_path=temp_test_dir,
            workspace_path=temp_test_dir / "workspace",
            storage_mode="memory",
            overlay_mode=OverlayMode.AUTO,
            materialization_context="testing"
        )
        
        # Replace file_manager and microscope_handler with mocks
        orchestrator.file_manager = mock_file_manager
        orchestrator.microscope_handler = mock_microscope_handler
        
        # Create test steps with different materialization requirements
        class TestStep1(Step):
            requires_fs_input = True
            
            def process(self, context):
                result = self.create_result()
                result.results['step1'] = np.ones((10, 10))
                return result
                
        class TestStep2(Step):
            requires_fs_input = False
            
            def process(self, context):
                result = self.create_result()
                result.results['step2'] = np.ones((10, 10)) * 2
                return result
                
        class TestStep3(Step):
            requires_fs_output = True
            
            def process(self, context):
                result = self.create_result()
                result.results['step3'] = np.ones((10, 10)) * 3
                return result
        
        # Create a pipeline with the test steps
        pipeline = Pipeline(
            steps=[TestStep1(), TestStep2(), TestStep3()],
            name="Test Pipeline"
        )
        
        # Create a context
        context = orchestrator.create_context(pipeline, well_filter=["A01"])
        
        # Mock the needs_materialization method to track which steps need materialization
        original_needs_materialization = MaterializationManager.needs_materialization
        materialization_calls = []
        
        def mock_needs_materialization(self, step):
            materialization_calls.append(step)
            return original_needs_materialization(self, step)
            
        with patch.object(MaterializationManager, 'needs_materialization', mock_needs_materialization):
            # Run the pipeline
            result_context = pipeline.run(context)
            
            # Verify that needs_materialization was called for each step
            assert len(materialization_calls) == 3
            
            # Verify that the steps were executed
            assert 'step1' in result_context.results
            assert 'step2' in result_context.results
            assert 'step3' in result_context.results
            
            # Verify that the correct steps needed materialization
            assert materialization_calls[0].requires_fs_input  # TestStep1
            assert not materialization_calls[1].requires_fs_input  # TestStep2
            assert materialization_calls[2].requires_fs_output  # TestStep3
