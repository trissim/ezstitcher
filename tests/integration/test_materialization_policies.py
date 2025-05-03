"""Integration tests for MaterializationManager policies."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

from ezstitcher.core.steps import Step
from ezstitcher.core.pipeline import Pipeline, ProcessingContext
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.io.overlay import OverlayMode, OverlayOperation
from ezstitcher.io.materialization import (
    MaterializationManager, MaterializationPolicy, FailureMode, MaterializationError,
    MaterializationMethod
)


class TestMaterializationPolicies:
    """Integration tests for MaterializationManager policies."""

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
        
    def create_orchestrator(self, policy=None):
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
        
        # Initialize the materialization manager with the specified policy
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
        
    def test_default_policy(self):
        """Test the default policy."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with default policy
        orchestrator = self.create_orchestrator(policy=None)
        context = self.create_context(orchestrator)
        
        # Run the pipeline
        pipeline.run(context)
        
        # Check the default policy values
        assert orchestrator.materialization_manager.policy.method == MaterializationMethod.COPY
        assert orchestrator.materialization_manager.policy.failure_mode == FailureMode.FAIL_FAST
        assert orchestrator.materialization_manager.policy.force_memory is False
        assert orchestrator.materialization_manager.policy.force_disk is False
        
    def test_copy_method_policy(self):
        """Test the COPY method policy."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with COPY method policy
        policy = MaterializationPolicy(method=MaterializationMethod.COPY)
        orchestrator = self.create_orchestrator(policy=policy)
        context = self.create_context(orchestrator)
        
        # Run the pipeline
        pipeline.run(context)
        
        # Check that the storage adapter was called with the right method
        for call_args in orchestrator.storage_adapter.register_for_overlay.call_args_list:
            assert call_args[1].get('method', MaterializationMethod.COPY) == MaterializationMethod.COPY
            
    def test_symlink_method_policy(self):
        """Test the SYMLINK method policy."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with SYMLINK method policy
        policy = MaterializationPolicy(method=MaterializationMethod.SYMLINK)
        orchestrator = self.create_orchestrator(policy=policy)
        context = self.create_context(orchestrator)
        
        # Run the pipeline
        pipeline.run(context)
        
        # Check that the storage adapter was called with the right method
        for call_args in orchestrator.storage_adapter.register_for_overlay.call_args_list:
            assert call_args[1].get('method', MaterializationMethod.COPY) == MaterializationMethod.SYMLINK
            
    def test_force_memory_policy(self):
        """Test the force_memory policy."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with force_memory=True policy
        policy = MaterializationPolicy(force_memory=True)
        orchestrator = self.create_orchestrator(policy=policy)
        context = self.create_context(orchestrator)
        
        # Run the pipeline
        pipeline.run(context)
        
        # Check that materialization was not triggered
        assert not orchestrator.storage_adapter.register_for_overlay.called
        assert not orchestrator.storage_adapter.execute_overlay_operation.called
        
    def test_force_disk_policy(self):
        """Test the force_disk policy."""
        # Create a pipeline with a step that doesn't require filesystem access
        class NoFlagsStep(Step):
            def process(self, context):
                result = self.create_result()
                result.add_result("test_key", "test_value")
                return result
                
        # Create a pipeline with the step
        pipeline = Pipeline(
            steps=[NoFlagsStep(func=lambda x: x, name="No Flags Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with force_disk=True policy
        policy = MaterializationPolicy(force_disk=True)
        orchestrator = self.create_orchestrator(policy=policy)
        context = self.create_context(orchestrator)
        
        # Run the pipeline
        pipeline.run(context)
        
        # Check that materialization was triggered even though the step doesn't require it
        assert orchestrator.storage_adapter.register_for_overlay.called
        assert orchestrator.storage_adapter.execute_overlay_operation.called
        
    def test_testing_context_policy(self):
        """Test the policy for testing context."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with testing context policy
        policy = MaterializationPolicy.for_context("testing")
        orchestrator = self.create_orchestrator(policy=policy)
        context = self.create_context(orchestrator)
        
        # Run the pipeline
        pipeline.run(context)
        
        # Check the testing policy values
        assert orchestrator.materialization_manager.policy.method == MaterializationMethod.COPY
        assert orchestrator.materialization_manager.policy.failure_mode == FailureMode.LOG_AND_CONTINUE
        assert orchestrator.materialization_manager.policy.force_memory is True
        assert orchestrator.materialization_manager.policy.force_disk is False
        
        # Check that materialization was not triggered (force_memory=True)
        assert not orchestrator.storage_adapter.register_for_overlay.called
        assert not orchestrator.storage_adapter.execute_overlay_operation.called
        
    def test_benchmark_context_policy(self):
        """Test the policy for benchmark context."""
        # Create a pipeline with a step that doesn't require filesystem access
        class NoFlagsStep(Step):
            def process(self, context):
                result = self.create_result()
                result.add_result("test_key", "test_value")
                return result
                
        # Create a pipeline with the step
        pipeline = Pipeline(
            steps=[NoFlagsStep(func=lambda x: x, name="No Flags Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with benchmark context policy
        policy = MaterializationPolicy.for_context("benchmark")
        orchestrator = self.create_orchestrator(policy=policy)
        context = self.create_context(orchestrator)
        
        # Run the pipeline
        pipeline.run(context)
        
        # Check the benchmark policy values
        assert orchestrator.materialization_manager.policy.method == MaterializationMethod.COPY
        assert orchestrator.materialization_manager.policy.failure_mode == FailureMode.FAIL_FAST
        assert orchestrator.materialization_manager.policy.force_memory is False
        assert orchestrator.materialization_manager.policy.force_disk is True
        
        # Check that materialization was triggered (force_disk=True)
        assert orchestrator.storage_adapter.register_for_overlay.called
        assert orchestrator.storage_adapter.execute_overlay_operation.called
        
    def test_production_context_policy(self):
        """Test the policy for production context."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with production context policy
        policy = MaterializationPolicy.for_context("production")
        orchestrator = self.create_orchestrator(policy=policy)
        context = self.create_context(orchestrator)
        
        # Run the pipeline
        pipeline.run(context)
        
        # Check the production policy values
        assert orchestrator.materialization_manager.policy.method == MaterializationMethod.SYMLINK
        assert orchestrator.materialization_manager.policy.failure_mode == FailureMode.FAIL_FAST
        assert orchestrator.materialization_manager.policy.force_memory is False
        assert orchestrator.materialization_manager.policy.force_disk is False
        
        # Check that materialization was triggered with SYMLINK method
        for call_args in orchestrator.storage_adapter.register_for_overlay.call_args_list:
            assert call_args[1].get('method', MaterializationMethod.COPY) == MaterializationMethod.SYMLINK
            
    def test_unknown_context_policy(self):
        """Test the policy for an unknown context."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )
        
        # Create an orchestrator with an unknown context policy
        policy = MaterializationPolicy.for_context("unknown")
        orchestrator = self.create_orchestrator(policy=policy)
        context = self.create_context(orchestrator)
        
        # Run the pipeline
        pipeline.run(context)
        
        # Check that the default policy was used
        assert orchestrator.materialization_manager.policy.method == MaterializationMethod.COPY
        assert orchestrator.materialization_manager.policy.failure_mode == FailureMode.FAIL_FAST
        assert orchestrator.materialization_manager.policy.force_memory is False
        assert orchestrator.materialization_manager.policy.force_disk is False
