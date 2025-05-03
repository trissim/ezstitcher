"""Tests for PipelineOrchestrator initialization."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.microscope_interfaces import MicroscopeHandler


class TestOrchestratorInitialization:
    """Test the PipelineOrchestrator initialization behavior."""

    def setup_method(self):
        """Set up test environment."""
        self.config = PipelineConfig()
        self.plate_path = Path("dummy/plate")

        # Create an orchestrator without initializing it
        self.orchestrator = PipelineOrchestrator(
            plate_path=self.plate_path,
            config=self.config,
            storage_mode="legacy"
        )

        # Create a mock pipeline
        self.pipeline = MagicMock(spec=Pipeline)
        self.pipeline.steps = []

    def test_run_without_initialization(self):
        """Test that run() raises an exception if called before initialization."""
        with pytest.raises(RuntimeError, match="Orchestrator must be initialized before calling run"):
            self.orchestrator.run()

    def test_prepare_pipeline_paths_without_initialization(self):
        """Test that prepare_pipeline_paths() raises an exception if called before initialization."""
        with pytest.raises(RuntimeError, match="Orchestrator must be initialized before calling prepare_pipeline_paths"):
            self.orchestrator.prepare_pipeline_paths(self.pipeline)

    def test_create_context_without_initialization(self):
        """Test that create_context() raises an exception if called before initialization."""
        with pytest.raises(RuntimeError, match="Orchestrator must be initialized before calling create_context"):
            self.orchestrator.create_context(self.pipeline)

    def test_get_stitcher_without_initialization(self):
        """Test that get_stitcher() raises an exception if called before initialization."""
        with pytest.raises(RuntimeError, match="Orchestrator must be initialized before calling get_stitcher"):
            self.orchestrator.get_stitcher()

    def test_generate_positions_without_initialization(self):
        """Test that generate_positions() raises an exception if called before initialization."""
        with pytest.raises(RuntimeError, match="Orchestrator must be initialized before calling generate_positions"):
            self.orchestrator.generate_positions("A1", Path("input"), Path("output"))

    def test_stitch_images_without_initialization(self):
        """Test that stitch_images() raises an exception if called before initialization."""
        with pytest.raises(RuntimeError, match="Orchestrator must be initialized before calling stitch_images"):
            self.orchestrator.stitch_images("A1", Path("input"), Path("output"), Path("positions.json"))

    @patch('ezstitcher.core.pipeline_orchestrator.create_microscope_handler')
    def test_initialize_idempotent(self, mock_create_microscope_handler):
        """Test that initialize() is idempotent."""
        # Mock the microscope handler with required attributes
        mock_parser = MagicMock()
        mock_microscope_handler = MagicMock(spec=MicroscopeHandler)
        mock_microscope_handler.parser = mock_parser
        mock_create_microscope_handler.return_value = mock_microscope_handler

        # Initialize the orchestrator
        self.orchestrator.initialize()
        assert self.orchestrator._initialized is True

        # Initialize again
        self.orchestrator.initialize()
        assert self.orchestrator._initialized is True

        # Verify that create_microscope_handler was called only once
        assert mock_create_microscope_handler.call_count == 1

    @patch('ezstitcher.core.pipeline_orchestrator.create_microscope_handler')
    def test_method_chaining(self, mock_create_microscope_handler):
        """Test that initialize() supports method chaining."""
        # Mock the microscope handler with required attributes
        mock_parser = MagicMock()
        mock_microscope_handler = MagicMock(spec=MicroscopeHandler)
        mock_microscope_handler.parser = mock_parser
        mock_create_microscope_handler.return_value = mock_microscope_handler

        # Initialize the orchestrator with method chaining
        result = self.orchestrator.initialize()
        assert result is self.orchestrator

    @patch('ezstitcher.core.pipeline_orchestrator.create_microscope_handler')
    def test_component_availability_checks(self, mock_create_microscope_handler):
        """Test that component availability checks work correctly."""
        # Mock the microscope handler with required attributes
        mock_parser = MagicMock()
        mock_microscope_handler = MagicMock(spec=MicroscopeHandler)
        mock_microscope_handler.parser = mock_parser
        mock_create_microscope_handler.return_value = mock_microscope_handler

        # Before initialization, component checks should raise exceptions
        with pytest.raises(RuntimeError, match="File manager is not initialized"):
            self.orchestrator._ensure_file_manager()

        with pytest.raises(RuntimeError, match="Microscope handler is not initialized"):
            self.orchestrator._ensure_microscope_handler()

        with pytest.raises(RuntimeError, match="Stitcher is not initialized"):
            self.orchestrator._ensure_stitcher()

        # After initialization, component checks should return the components
        self.orchestrator.initialize()

        assert self.orchestrator._ensure_file_manager() is self.orchestrator.file_manager
        assert self.orchestrator._ensure_microscope_handler() is self.orchestrator.microscope_handler
        assert self.orchestrator._ensure_stitcher() is self.orchestrator.stitcher
