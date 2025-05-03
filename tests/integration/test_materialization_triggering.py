"""Integration tests for materialization triggering in Pipeline.run()."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from ezstitcher.core.steps import Step
from ezstitcher.core.pipeline import Pipeline, ProcessingContext
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.io.overlay import OverlayMode


class TestMaterializationTriggering:
    """Test that materialization is triggered in Pipeline.run()."""

    def test_materialization_triggered_for_requires_legacy_fs(self, tmp_path):
        """Test that materialization is triggered for steps with requires_legacy_fs=True."""
        # Create a step with requires_legacy_fs=True
        class TestStep(Step):
            requires_legacy_fs = True

            def process(self, context):
                result = self.create_result()
                result.add_result("test_key", "test_value")
                return result

        # Create a pipeline with the step
        pipeline = Pipeline(
            steps=[TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )

        # Create a real orchestrator with mocked components
        orchestrator = PipelineOrchestrator(
            storage_mode="memory",
            overlay_mode=OverlayMode.AUTO
        )

        # Mock the needs_materialization and prepare_materialization methods
        orchestrator.needs_materialization = MagicMock(return_value=True)
        orchestrator.prepare_materialization = MagicMock(return_value={"test_path": Path("/tmp/test")})

        # Create a context with the orchestrator
        context = ProcessingContext()
        context.orchestrator = orchestrator
        context.well_filter = ["A01"]

        # Mock the microscope_handler to avoid actual pattern detection
        orchestrator.microscope_handler = MagicMock()
        orchestrator.microscope_handler.auto_detect_patterns.return_value = {"A01": []}

        # Mock the file_manager to avoid actual file operations
        orchestrator.file_manager = MagicMock()

        # Run the pipeline
        pipeline.run(context)

        # Check that needs_materialization was called
        assert orchestrator.needs_materialization.called

        # Check that prepare_materialization was called
        assert orchestrator.prepare_materialization.called

    def test_materialization_not_triggered_for_normal_step(self, tmp_path):
        """Test that materialization is not triggered for normal steps."""
        # Create a normal step without requires_legacy_fs
        class TestStep(Step):
            def process(self, context):
                result = self.create_result()
                result.add_result("test_key", "test_value")
                return result

        # Create a pipeline with the step
        pipeline = Pipeline(
            steps=[TestStep(func=lambda x: x, name="Test Step")],
            name="Test Pipeline"
        )

        # Create a real orchestrator with mocked components
        orchestrator = PipelineOrchestrator(
            storage_mode="memory",
            overlay_mode=OverlayMode.AUTO
        )

        # Mock the needs_materialization and prepare_materialization methods
        orchestrator.needs_materialization = MagicMock(return_value=False)
        orchestrator.prepare_materialization = MagicMock(return_value={})

        # Create a context with the orchestrator
        context = ProcessingContext()
        context.orchestrator = orchestrator
        context.well_filter = ["A01"]

        # Mock the microscope_handler to avoid actual pattern detection
        orchestrator.microscope_handler = MagicMock()
        orchestrator.microscope_handler.auto_detect_patterns.return_value = {"A01": []}

        # Mock the file_manager to avoid actual file operations
        orchestrator.file_manager = MagicMock()

        # Run the pipeline
        pipeline.run(context)

        # Check that needs_materialization was called
        assert orchestrator.needs_materialization.called

        # Check that prepare_materialization was not called
        assert not orchestrator.prepare_materialization.called
