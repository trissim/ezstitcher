"""Integration tests for MaterializationManager error handling."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest
import logging

from ezstitcher.core.steps import Step
from ezstitcher.core.pipeline import Pipeline, ProcessingContext
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.io.overlay import OverlayMode, OverlayOperation
from ezstitcher.io.materialization import (
    MaterializationManager, MaterializationPolicy, FailureMode, MaterializationError
)


class TestMaterializationErrorHandling:
    """Integration tests for MaterializationManager error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.test_dir = Path("/tmp/materialization_test")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a step with requires_fs_input=True
        class TestStep(Step):
            requires_fs_input = True
            
            def process(self, context):
                result = self.create_result()
                result.add_result("test_key", "test_value")
                return result
                
        # Store the step class
        self.TestStep = TestStep
        
    def teardown_method(self):
        """Clean up after tests."""
        # Remove the temporary directory
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def create_orchestrator(self, failure_mode=FailureMode.FAIL_FAST):
        """Create a PipelineOrchestrator with mocked components."""
        # Create a real orchestrator with mocked components
        orchestrator = PipelineOrchestrator(
            storage_mode="memory",
            overlay_mode=OverlayMode.AUTO
        )
        
        # Mock the microscope_handler to avoid actual pattern detection
        orchestrator.microscope_handler = MagicMock()
        orchestrator.microscope_handler.parser = MagicMock()
        orchestrator.microscope_handler.auto_detect_patterns.return_value = {"A01": ["*.tif"]}
        orchestrator.microscope_handler.parser.path_list_from_pattern.return_value = [
            Path(self.test_dir) / "image1.tif",
            Path(self.test_dir) / "image2.tif"
        ]
        
        # Mock the file_manager to avoid actual file operations
        orchestrator.file_manager = MagicMock()
        orchestrator.file_manager.ensure_directory.return_value = True
        
        # Mock the storage_adapter to avoid actual storage operations
        orchestrator.storage_adapter = MagicMock()
        orchestrator.storage_adapter.exists.return_value = True
        orchestrator.storage_adapter.register_for_overlay.return_value = Path("/tmp/overlay/file.tif")
        orchestrator.storage_adapter.execute_overlay_operation.return_value = True
        orchestrator.storage_adapter.cleanup_overlay_operations.return_value = 2
        orchestrator.storage_adapter.overlay_operations = {
            "key1": MagicMock(),
            "key2": MagicMock()
        }
        
        # Initialize the materialization manager with the specified failure mode
        policy = MaterializationPolicy(failure_mode=failure_mode)
        orchestrator.materialization_manager = MaterializationManager(None, policy)
            
        return orchestrator
        
    def create_context(self, orchestrator):
        """Create a ProcessingContext with the given orchestrator."""
        context = ProcessingContext()
        context.orchestrator = orchestrator
        context.well_filter = ["A01"]
        
        # Set up directory paths
        context.get_step_input_dir = MagicMock(return_value=self.test_dir)
        context.get_step_output_dir = MagicMock(return_value=self.test_dir / "output")
        
        # Update the materialization manager with the context
        if hasattr(orchestrator, 'materialization_manager'):
            orchestrator.materialization_manager.context = context
            
        return context
        
    def test_fail_fast_on_register_file(self):
        """Test FAIL_FAST behavior when register_file fails."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with FAIL_FAST policy
        orchestrator = self.create_orchestrator(failure_mode=FailureMode.FAIL_FAST)
        context = self.create_context(orchestrator)
        
        # Make the storage adapter raise an exception
        orchestrator.storage_adapter.register_for_overlay.side_effect = Exception("Test exception")
        
        # Run the pipeline and check for exception
        with pytest.raises(MaterializationError):
            pipeline.run(context)
            
    def test_log_and_continue_on_register_file(self):
        """Test LOG_AND_CONTINUE behavior when register_file fails."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with LOG_AND_CONTINUE policy
        orchestrator = self.create_orchestrator(failure_mode=FailureMode.LOG_AND_CONTINUE)
        context = self.create_context(orchestrator)
        
        # Make the storage adapter raise an exception
        orchestrator.storage_adapter.register_for_overlay.side_effect = Exception("Test exception")
        
        # Run the pipeline (should not raise an exception)
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            pipeline.run(context)
            
            # Check that the error was logged
            assert mock_logger.error.called
            
    def test_fallback_to_disk_on_register_file(self):
        """Test FALLBACK_TO_DISK behavior when register_file fails."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with FALLBACK_TO_DISK policy
        orchestrator = self.create_orchestrator(failure_mode=FailureMode.FALLBACK_TO_DISK)
        context = self.create_context(orchestrator)
        
        # Make the storage adapter raise an exception
        orchestrator.storage_adapter.register_for_overlay.side_effect = Exception("Test exception")
        
        # Run the pipeline (should not raise an exception)
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            pipeline.run(context)
            
            # Check that the error was logged
            assert mock_logger.error.called
            
    def test_fail_fast_on_execute_operations(self):
        """Test FAIL_FAST behavior when execute_overlay_operation fails."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with FAIL_FAST policy
        orchestrator = self.create_orchestrator(failure_mode=FailureMode.FAIL_FAST)
        context = self.create_context(orchestrator)
        
        # Make the storage adapter raise an exception during execution
        orchestrator.storage_adapter.execute_overlay_operation.side_effect = Exception("Test exception")
        
        # Run the pipeline and check for exception
        with pytest.raises(MaterializationError):
            pipeline.run(context)
            
    def test_log_and_continue_on_execute_operations(self):
        """Test LOG_AND_CONTINUE behavior when execute_overlay_operation fails."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with LOG_AND_CONTINUE policy
        orchestrator = self.create_orchestrator(failure_mode=FailureMode.LOG_AND_CONTINUE)
        context = self.create_context(orchestrator)
        
        # Make the storage adapter raise an exception during execution
        orchestrator.storage_adapter.execute_overlay_operation.side_effect = Exception("Test exception")
        
        # Run the pipeline (should not raise an exception)
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            pipeline.run(context)
            
            # Check that the error was logged
            assert mock_logger.error.called
            
    def test_fallback_to_disk_on_execute_operations(self):
        """Test FALLBACK_TO_DISK behavior when execute_overlay_operation fails."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with FALLBACK_TO_DISK policy
        orchestrator = self.create_orchestrator(failure_mode=FailureMode.FALLBACK_TO_DISK)
        context = self.create_context(orchestrator)
        
        # Make the storage adapter raise an exception during execution
        orchestrator.storage_adapter.execute_overlay_operation.side_effect = Exception("Test exception")
        
        # Run the pipeline (should not raise an exception)
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            pipeline.run(context)
            
            # Check that the error was logged
            assert mock_logger.error.called
            
    def test_fail_fast_on_cleanup_operations(self):
        """Test FAIL_FAST behavior when cleanup_overlay_operations fails."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with FAIL_FAST policy
        orchestrator = self.create_orchestrator(failure_mode=FailureMode.FAIL_FAST)
        context = self.create_context(orchestrator)
        
        # Make the storage adapter raise an exception during cleanup
        orchestrator.storage_adapter.cleanup_overlay_operations.side_effect = Exception("Test exception")
        
        # Run the pipeline and check for exception
        with pytest.raises(MaterializationError):
            pipeline.run(context)
            
    def test_log_and_continue_on_cleanup_operations(self):
        """Test LOG_AND_CONTINUE behavior when cleanup_overlay_operations fails."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with LOG_AND_CONTINUE policy
        orchestrator = self.create_orchestrator(failure_mode=FailureMode.LOG_AND_CONTINUE)
        context = self.create_context(orchestrator)
        
        # Make the storage adapter raise an exception during cleanup
        orchestrator.storage_adapter.cleanup_overlay_operations.side_effect = Exception("Test exception")
        
        # Run the pipeline (should not raise an exception)
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            pipeline.run(context)
            
            # Check that the error was logged
            assert mock_logger.error.called
            
    def test_fallback_to_disk_on_cleanup_operations(self):
        """Test FALLBACK_TO_DISK behavior when cleanup_overlay_operations fails."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with FALLBACK_TO_DISK policy
        orchestrator = self.create_orchestrator(failure_mode=FailureMode.FALLBACK_TO_DISK)
        context = self.create_context(orchestrator)
        
        # Make the storage adapter raise an exception during cleanup
        orchestrator.storage_adapter.cleanup_overlay_operations.side_effect = Exception("Test exception")
        
        # Run the pipeline (should not raise an exception)
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            pipeline.run(context)
            
            # Check that the error was logged
            assert mock_logger.error.called
            
    def test_error_handling_with_multiple_steps(self):
        """Test error handling with multiple steps."""
        # Create a pipeline with multiple steps
        pipeline = Pipeline(
            steps=[
                self.TestStep(func=lambda x: x, name="Step 1"),
                self.TestStep(func=lambda x: x, name="Step 2"),
                self.TestStep(func=lambda x: x, name="Step 3")
            ],
            name="Multi-Step Pipeline"
        )
        
        # Create an orchestrator with LOG_AND_CONTINUE policy
        orchestrator = self.create_orchestrator(failure_mode=FailureMode.LOG_AND_CONTINUE)
        context = self.create_context(orchestrator)
        
        # Make the storage adapter raise an exception for the second step only
        def side_effect(*args, **kwargs):
            if orchestrator.materialization_manager.prepare_for_step.call_count == 2:
                raise Exception("Test exception for step 2")
            return Path("/tmp/overlay/file.tif")
            
        orchestrator.storage_adapter.register_for_overlay.side_effect = side_effect
        
        # Run the pipeline (should not raise an exception)
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            pipeline.run(context)
            
            # Check that the error was logged
            assert mock_logger.error.called
            
            # Check that all steps were processed
            assert orchestrator.materialization_manager.prepare_for_step.call_count == 3
