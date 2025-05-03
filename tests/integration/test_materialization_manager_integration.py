"""Integration tests for the MaterializationManager."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.pipeline import Pipeline, ProcessingContext
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.io.overlay import OverlayMode
from ezstitcher.io.materialization import (
    MaterializationManager, MaterializationPolicy, FailureMode, MaterializationError
)


class TestMaterializationManagerIntegration:
    """Integration tests for the MaterializationManager."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.test_dir = Path("/tmp/materialization_test")
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a step with requires_fs_input=True
        class InputStep(Step):
            requires_fs_input = True

            def process(self, context):
                result = self.create_result()
                result.add_result("test_key", "test_value")
                return result

        # Create a step with requires_fs_output=True
        class OutputStep(Step):
            requires_fs_output = True

            def process(self, context):
                result = self.create_result()
                result.add_result("test_key", "test_value")
                return result

        # Create a step with both flags
        class BothStep(Step):
            requires_fs_input = True
            requires_fs_output = True

            def process(self, context):
                result = self.create_result()
                result.add_result("test_key", "test_value")
                return result

        # Create a step with no flags
        class NoFlagsStep(Step):
            def process(self, context):
                result = self.create_result()
                result.add_result("test_key", "test_value")
                return result

        # Store the step classes
        self.InputStep = InputStep
        self.OutputStep = OutputStep
        self.BothStep = BothStep
        self.NoFlagsStep = NoFlagsStep

    def teardown_method(self):
        """Clean up after tests."""
        # Remove the temporary directory
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_orchestrator(self, storage_mode="memory", overlay_mode=OverlayMode.AUTO,
                           materialization_context=None):
        """Create a PipelineOrchestrator with mocked components."""
        # Create a real orchestrator with mocked components
        orchestrator = PipelineOrchestrator(
            storage_mode=storage_mode,
            overlay_mode=overlay_mode,
            materialization_context=materialization_context
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

        # Initialize the materialization manager
        if storage_mode != "legacy":
            policy = MaterializationPolicy.for_context(materialization_context) if materialization_context else None
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

    def test_materialization_with_requires_fs_input(self):
        """Test materialization with a step that requires filesystem input."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.InputStep(func=lambda x: x, name="Input Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator and context
        orchestrator = self.create_orchestrator(storage_mode="memory")
        context = self.create_context(orchestrator)

        # Spy on the materialization manager methods
        with patch.object(orchestrator.materialization_manager, 'prepare_for_step', wraps=orchestrator.materialization_manager.prepare_for_step) as mock_prepare:
            with patch.object(orchestrator.materialization_manager, 'execute_pending_operations', wraps=orchestrator.materialization_manager.execute_pending_operations) as mock_execute:
                # Run the pipeline
                pipeline.run(context)

                # Check that prepare_for_step was called
                assert mock_prepare.called

                # Check that execute_pending_operations was called
                assert mock_execute.called

    def test_materialization_with_requires_fs_output(self):
        """Test materialization with a step that requires filesystem output."""
        # Create a pipeline with a step that requires filesystem output
        pipeline = Pipeline(
            steps=[self.OutputStep(func=lambda x: x, name="Output Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator and context
        orchestrator = self.create_orchestrator(storage_mode="memory")
        context = self.create_context(orchestrator)

        # Spy on the materialization manager methods
        with patch.object(orchestrator.materialization_manager, 'prepare_for_step', wraps=orchestrator.materialization_manager.prepare_for_step) as mock_prepare:
            with patch.object(orchestrator.materialization_manager, 'execute_pending_operations', wraps=orchestrator.materialization_manager.execute_pending_operations) as mock_execute:
                # Run the pipeline
                pipeline.run(context)

                # Check that prepare_for_step was called
                assert mock_prepare.called

                # Check that execute_pending_operations was called
                assert mock_execute.called

    def test_materialization_with_both_flags(self):
        """Test materialization with a step that requires both input and output filesystem access."""
        # Create a pipeline with a step that requires both input and output filesystem access
        pipeline = Pipeline(
            steps=[self.BothStep(func=lambda x: x, name="Both Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator and context
        orchestrator = self.create_orchestrator(storage_mode="memory")
        context = self.create_context(orchestrator)

        # Spy on the materialization manager methods
        with patch.object(orchestrator.materialization_manager, 'prepare_for_step', wraps=orchestrator.materialization_manager.prepare_for_step) as mock_prepare:
            with patch.object(orchestrator.materialization_manager, 'execute_pending_operations', wraps=orchestrator.materialization_manager.execute_pending_operations) as mock_execute:
                # Run the pipeline
                pipeline.run(context)

                # Check that prepare_for_step was called
                assert mock_prepare.called

                # Check that execute_pending_operations was called
                assert mock_execute.called

    def test_materialization_with_no_flags(self):
        """Test materialization with a step that doesn't require filesystem access."""
        # Create a pipeline with a step that doesn't require filesystem access
        pipeline = Pipeline(
            steps=[self.NoFlagsStep(func=lambda x: x, name="No Flags Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator and context
        orchestrator = self.create_orchestrator(storage_mode="memory")
        context = self.create_context(orchestrator)

        # Spy on the materialization manager methods
        with patch.object(orchestrator.materialization_manager, 'prepare_for_step', wraps=orchestrator.materialization_manager.prepare_for_step) as mock_prepare:
            with patch.object(orchestrator.materialization_manager, 'execute_pending_operations', wraps=orchestrator.materialization_manager.execute_pending_operations) as mock_execute:
                # Run the pipeline
                pipeline.run(context)

                # Check that prepare_for_step was not called
                assert not mock_prepare.called

                # Check that execute_pending_operations was not called
                assert not mock_execute.called

    def test_materialization_with_mixed_pipeline(self):
        """Test materialization with a pipeline containing mixed steps."""
        # Create a pipeline with mixed steps
        pipeline = Pipeline(
            steps=[
                self.NoFlagsStep(func=lambda x: x, name="No Flags Step"),
                self.InputStep(func=lambda x: x, name="Input Step"),
                self.NoFlagsStep(func=lambda x: x, name="Another No Flags Step"),
                self.OutputStep(func=lambda x: x, name="Output Step")
            ],
            name="Mixed Pipeline"
        )

        # Create an orchestrator and context
        orchestrator = self.create_orchestrator(storage_mode="memory")
        context = self.create_context(orchestrator)

        # Spy on the materialization manager methods
        with patch.object(orchestrator.materialization_manager, 'prepare_for_step', wraps=orchestrator.materialization_manager.prepare_for_step) as mock_prepare:
            with patch.object(orchestrator.materialization_manager, 'execute_pending_operations', wraps=orchestrator.materialization_manager.execute_pending_operations) as mock_execute:
                # Run the pipeline
                pipeline.run(context)

                # Check that prepare_for_step was called twice (for InputStep and OutputStep)
                assert mock_prepare.call_count == 2

                # Check that execute_pending_operations was called twice
                assert mock_execute.call_count == 2

    def test_materialization_with_position_generation_step(self):
        """Test materialization with a PositionGenerationStep."""
        # Create a pipeline with a PositionGenerationStep
        pipeline = Pipeline(
            steps=[PositionGenerationStep(func=lambda x: x, name="Position Generation")],
            name="Position Generation Pipeline"
        )

        # Create an orchestrator and context
        orchestrator = self.create_orchestrator(storage_mode="memory")
        context = self.create_context(orchestrator)

        # Mock the generate_positions method
        orchestrator.generate_positions = MagicMock(return_value=(Path("/tmp/positions/A01.csv"), "*.tif"))

        # Spy on the materialization manager methods
        with patch.object(orchestrator.materialization_manager, 'prepare_for_step', wraps=orchestrator.materialization_manager.prepare_for_step) as mock_prepare:
            with patch.object(orchestrator.materialization_manager, 'execute_pending_operations', wraps=orchestrator.materialization_manager.execute_pending_operations) as mock_execute:
                # Run the pipeline
                pipeline.run(context)

                # Check that prepare_for_step was called
                assert mock_prepare.called

                # Check that execute_pending_operations was called
                assert mock_execute.called

    def test_materialization_with_image_stitching_step(self):
        """Test materialization with an ImageStitchingStep."""
        # Create a pipeline with an ImageStitchingStep
        pipeline = Pipeline(
            steps=[ImageStitchingStep(func=lambda x: x, name="Image Stitching")],
            name="Image Stitching Pipeline"
        )

        # Create an orchestrator and context
        orchestrator = self.create_orchestrator(storage_mode="memory")
        context = self.create_context(orchestrator)

        # Set up positions_dir in context
        context.positions_dir = self.test_dir / "positions"

        # Mock the stitch_images method
        orchestrator.stitch_images = MagicMock(return_value=[Path("/tmp/stitched/A01_stitched.tif")])

        # Spy on the materialization manager methods
        with patch.object(orchestrator.materialization_manager, 'prepare_for_step', wraps=orchestrator.materialization_manager.prepare_for_step) as mock_prepare:
            with patch.object(orchestrator.materialization_manager, 'execute_pending_operations', wraps=orchestrator.materialization_manager.execute_pending_operations) as mock_execute:
                # Run the pipeline
                pipeline.run(context)

                # Check that prepare_for_step was called
                assert mock_prepare.called

                # Check that execute_pending_operations was called
                assert mock_execute.called

    def test_materialization_with_legacy_storage_mode(self):
        """Test materialization with legacy storage mode."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.InputStep(func=lambda x: x, name="Input Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator with legacy storage mode
        orchestrator = self.create_orchestrator(storage_mode="legacy")
        context = self.create_context(orchestrator)

        # Run the pipeline
        pipeline.run(context)

        # Check that materialization_manager is not present
        assert not hasattr(orchestrator, 'materialization_manager')

    def test_materialization_with_disabled_overlay_mode(self):
        """Test materialization with disabled overlay mode."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.InputStep(func=lambda x: x, name="Input Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator with disabled overlay mode
        orchestrator = self.create_orchestrator(storage_mode="memory", overlay_mode=OverlayMode.DISABLED)
        context = self.create_context(orchestrator)

        # Spy on the materialization manager methods
        with patch.object(orchestrator.materialization_manager, 'prepare_for_step', wraps=orchestrator.materialization_manager.prepare_for_step) as mock_prepare:
            with patch.object(orchestrator.materialization_manager, 'execute_pending_operations', wraps=orchestrator.materialization_manager.execute_pending_operations) as mock_execute:
                # Run the pipeline
                pipeline.run(context)

                # Check that prepare_for_step was not called
                assert not mock_prepare.called

                # Check that execute_pending_operations was not called
                assert not mock_execute.called

    def test_materialization_with_testing_context(self):
        """Test materialization with testing context."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.InputStep(func=lambda x: x, name="Input Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator with testing context
        orchestrator = self.create_orchestrator(storage_mode="memory", materialization_context="testing")
        context = self.create_context(orchestrator)

        # Spy on the materialization manager methods
        with patch.object(orchestrator.materialization_manager, 'prepare_for_step', wraps=orchestrator.materialization_manager.prepare_for_step) as mock_prepare:
            with patch.object(orchestrator.materialization_manager, 'execute_pending_operations', wraps=orchestrator.materialization_manager.execute_pending_operations) as mock_execute:
                # Run the pipeline
                pipeline.run(context)

                # Check that prepare_for_step was not called (force_memory=True in testing context)
                assert not mock_prepare.called

                # Check that execute_pending_operations was not called
                assert not mock_execute.called

    def test_materialization_with_benchmark_context(self):
        """Test materialization with benchmark context."""
        # Create a pipeline with a step that doesn't require filesystem access
        pipeline = Pipeline(
            steps=[self.NoFlagsStep(func=lambda x: x, name="No Flags Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator with benchmark context
        orchestrator = self.create_orchestrator(storage_mode="memory", materialization_context="benchmark")
        context = self.create_context(orchestrator)

        # Spy on the materialization manager methods
        with patch.object(orchestrator.materialization_manager, 'prepare_for_step', wraps=orchestrator.materialization_manager.prepare_for_step) as mock_prepare:
            with patch.object(orchestrator.materialization_manager, 'execute_pending_operations', wraps=orchestrator.materialization_manager.execute_pending_operations) as mock_execute:
                # Run the pipeline
                pipeline.run(context)

                # Check that prepare_for_step was called (force_disk=True in benchmark context)
                assert mock_prepare.called

                # Check that execute_pending_operations was called
                assert mock_execute.called

    def test_materialization_with_fail_fast_policy(self):
        """Test materialization with fail_fast policy."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.InputStep(func=lambda x: x, name="Input Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator
        orchestrator = self.create_orchestrator(storage_mode="memory")
        context = self.create_context(orchestrator)

        # Set the failure mode to FAIL_FAST
        orchestrator.materialization_manager.policy.failure_mode = FailureMode.FAIL_FAST

        # Make the storage adapter raise an exception
        orchestrator.storage_adapter.register_for_overlay.side_effect = Exception("Test exception")

        # Run the pipeline and check for exception
        with pytest.raises(MaterializationError):
            pipeline.run(context)

    def test_materialization_with_log_and_continue_policy(self):
        """Test materialization with log_and_continue policy."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.InputStep(func=lambda x: x, name="Input Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator
        orchestrator = self.create_orchestrator(storage_mode="memory")
        context = self.create_context(orchestrator)

        # Set the failure mode to LOG_AND_CONTINUE
        orchestrator.materialization_manager.policy.failure_mode = FailureMode.LOG_AND_CONTINUE

        # Make the storage adapter raise an exception
        orchestrator.storage_adapter.register_for_overlay.side_effect = Exception("Test exception")

        # Run the pipeline (should not raise an exception)
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            pipeline.run(context)

            # Check that the error was logged
            assert mock_logger.error.called

    def test_materialization_with_fallback_to_disk_policy(self):
        """Test materialization with fallback_to_disk policy."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.InputStep(func=lambda x: x, name="Input Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator
        orchestrator = self.create_orchestrator(storage_mode="memory")
        context = self.create_context(orchestrator)

        # Set the failure mode to FALLBACK_TO_DISK
        orchestrator.materialization_manager.policy.failure_mode = FailureMode.FALLBACK_TO_DISK

        # Make the storage adapter raise an exception
        orchestrator.storage_adapter.register_for_overlay.side_effect = Exception("Test exception")

        # Run the pipeline (should not raise an exception)
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            pipeline.run(context)

            # Check that the error was logged
            assert mock_logger.error.called

    def test_materialization_cleanup_after_pipeline(self):
        """Test that materialization is cleaned up after the pipeline completes."""
        # Create a pipeline with a step that requires filesystem input
        pipeline = Pipeline(
            steps=[self.InputStep(func=lambda x: x, name="Input Step")],
            name="Test Pipeline"
        )

        # Create an orchestrator and context
        orchestrator = self.create_orchestrator(storage_mode="memory")
        context = self.create_context(orchestrator)

        # Spy on the materialization manager methods
        with patch.object(orchestrator.materialization_manager, 'cleanup_operations', wraps=orchestrator.materialization_manager.cleanup_operations) as mock_cleanup:
            # Run the pipeline
            pipeline.run(context)

            # Check that cleanup_operations was called
            assert mock_cleanup.called


