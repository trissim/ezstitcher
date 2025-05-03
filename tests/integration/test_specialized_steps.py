"""Integration tests for specialized steps with MaterializationManager."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.io.materialization import MaterializationManager
from ezstitcher.io.overlay import OverlayMode

# Import fixtures
from tests.fixtures.materialization_fixtures import (
    temp_test_dir, overlay_dir, mock_file_manager, mock_microscope_handler
)


class TestSpecializedSteps:
    """Integration tests for specialized steps with MaterializationManager."""
    
    def test_position_generation_step(self, temp_test_dir, overlay_dir):
        """Test PositionGenerationStep with MaterializationManager."""
        # Create a real PipelineOrchestrator
        orchestrator = PipelineOrchestrator(
            plate_path=temp_test_dir,
            workspace_path=temp_test_dir / "workspace",
            storage_mode="memory",
            overlay_mode=OverlayMode.AUTO
        )
        
        # Replace file_manager and microscope_handler with mocks
        orchestrator.file_manager = mock_file_manager
        orchestrator.microscope_handler = mock_microscope_handler
        
        # Mock generate_positions to return a positions file
        positions_file = temp_test_dir / "positions" / "A01.csv"
        orchestrator.generate_positions = MagicMock(return_value=positions_file)
        
        # Create a pipeline with PositionGenerationStep
        pipeline = Pipeline(
            steps=[PositionGenerationStep()],
            name="Position Generation Pipeline"
        )
        
        # Create a context
        context = orchestrator.create_context(pipeline, well_filter=["A01"])
        context.positions_dir = temp_test_dir / "positions"
        
        # Mock the prepare_for_step and execute_pending_operations methods
        orchestrator.materialization_manager.prepare_for_step = MagicMock(return_value={})
        orchestrator.materialization_manager.execute_pending_operations = MagicMock(return_value=0)
        
        # Run the pipeline
        result_context = pipeline.run(context)
        
        # Verify that the MaterializationManager methods were called
        orchestrator.materialization_manager.prepare_for_step.assert_called()
        orchestrator.materialization_manager.execute_pending_operations.assert_called()
        
        # Verify that the positions file was generated
        assert orchestrator.generate_positions.called
        
    def test_image_stitching_step(self, temp_test_dir, overlay_dir):
        """Test ImageStitchingStep with MaterializationManager."""
        # Create a real PipelineOrchestrator
        orchestrator = PipelineOrchestrator(
            plate_path=temp_test_dir,
            workspace_path=temp_test_dir / "workspace",
            storage_mode="memory",
            overlay_mode=OverlayMode.AUTO
        )
        
        # Replace file_manager and microscope_handler with mocks
        orchestrator.file_manager = mock_file_manager
        orchestrator.microscope_handler = mock_microscope_handler
        
        # Mock stitch_images to return a list of stitched files
        stitched_files = [temp_test_dir / "stitched" / "A01_stitched.tif"]
        orchestrator.stitch_images = MagicMock(return_value=stitched_files)
        
        # Create a pipeline with ImageStitchingStep
        pipeline = Pipeline(
            steps=[ImageStitchingStep()],
            name="Image Stitching Pipeline"
        )
        
        # Create a context
        context = orchestrator.create_context(pipeline, well_filter=["A01"])
        context.positions_dir = temp_test_dir / "positions"
        
        # Mock the prepare_for_step and execute_pending_operations methods
        orchestrator.materialization_manager.prepare_for_step = MagicMock(return_value={})
        orchestrator.materialization_manager.execute_pending_operations = MagicMock(return_value=0)
        
        # Run the pipeline
        result_context = pipeline.run(context)
        
        # Verify that the MaterializationManager methods were called
        orchestrator.materialization_manager.prepare_for_step.assert_called()
        orchestrator.materialization_manager.execute_pending_operations.assert_called()
        
        # Verify that the stitch_images method was called
        assert orchestrator.stitch_images.called
