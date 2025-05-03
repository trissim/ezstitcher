"""Integration tests for position materialization."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import ImageStitchingStep
from ezstitcher.io.materialization_resolver import MaterializationResolver
from ezstitcher.io.memory_path import MemoryPath
from ezstitcher.io.virtual_path import PhysicalPath


class TestPositionMaterialization:
    """Integration tests for position materialization."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orchestrator = MagicMock()
        orchestrator.storage_mode = "memory"
        orchestrator.overlay_mode = "auto"
        
        # Create a mock materialization manager
        orchestrator.materialization_manager = MagicMock()
        orchestrator.materialization_manager.needs_materialization = MagicMock(return_value=False)
        
        return orchestrator
        
    @pytest.fixture
    def mock_context(self):
        """Create a mock context."""
        context = MagicMock()
        context.well_filter = ["A01"]
        return context

    def test_image_stitching_step_materializes_memory_positions(self, mock_orchestrator, mock_context):
        """Test that ImageStitchingStep materializes positions when they are memory-backed."""
        # Create a pipeline with an ImageStitchingStep
        step = ImageStitchingStep(name="Image Stitching")
        
        # Create a memory-backed positions directory
        positions_dir = MemoryPath("/positions")
        mock_context.positions_dir = positions_dir
        
        # Set up the materialization manager to use our implementation
        with patch('ezstitcher.io.materialization.MaterializationManager.needs_materialization') as mock_needs_mat:
            # Call our implementation directly
            def side_effect(step):
                from ezstitcher.core.steps import ImageStitchingStep
                from ezstitcher.io.virtual_path import VirtualPath, PhysicalPath
                
                if isinstance(step, ImageStitchingStep):
                    positions_dir = mock_context.positions_dir
                    if isinstance(positions_dir, VirtualPath) and not isinstance(positions_dir, PhysicalPath):
                        return True
                return False
                
            mock_needs_mat.side_effect = side_effect
            
            # Check if materialization is needed
            needs_mat = MaterializationResolver.needs_materialization(
                step, mock_orchestrator, mock_context
            )
            
            # Should need materialization because positions directory is memory-backed
            assert needs_mat is True
            
            # Verify that the materialization manager was called with the step
            mock_orchestrator.materialization_manager.needs_materialization.assert_called_once_with(step)

    def test_image_stitching_step_does_not_materialize_disk_positions(self, mock_orchestrator, mock_context):
        """Test that ImageStitchingStep does not materialize positions when they are disk-backed."""
        # Create a pipeline with an ImageStitchingStep
        step = ImageStitchingStep(name="Image Stitching")
        
        # Create a disk-backed positions directory
        positions_dir = PhysicalPath("/positions")
        mock_context.positions_dir = positions_dir
        
        # Set up the materialization manager to use our implementation
        with patch('ezstitcher.io.materialization.MaterializationManager.needs_materialization') as mock_needs_mat:
            # Call our implementation directly
            def side_effect(step):
                from ezstitcher.core.steps import ImageStitchingStep
                from ezstitcher.io.virtual_path import VirtualPath, PhysicalPath
                
                if isinstance(step, ImageStitchingStep):
                    positions_dir = mock_context.positions_dir
                    if isinstance(positions_dir, VirtualPath) and not isinstance(positions_dir, PhysicalPath):
                        return True
                return False
                
            mock_needs_mat.side_effect = side_effect
            
            # Check if materialization is needed
            needs_mat = MaterializationResolver.needs_materialization(
                step, mock_orchestrator, mock_context
            )
            
            # Should not need materialization because positions directory is disk-backed
            assert needs_mat is False
            
            # Verify that the materialization manager was called with the step
            mock_orchestrator.materialization_manager.needs_materialization.assert_called_once_with(step)

    def test_non_image_stitching_step_uses_flags(self, mock_orchestrator, mock_context):
        """Test that non-ImageStitchingStep uses flags for materialization."""
        # Create a generic step with requires_fs_input=False
        step = MagicMock()
        step.requires_fs_input = False
        step.requires_fs_output = False
        step.force_disk_output = False
        step.requires_legacy_fs = False
        step.needs_materialization.return_value = False
        
        # Set up the materialization manager to use our implementation
        with patch('ezstitcher.io.materialization.MaterializationManager.needs_materialization') as mock_needs_mat:
            # Call our implementation directly
            def side_effect(step):
                # For non-ImageStitchingStep, just return the flag value
                return step.needs_materialization()
                
            mock_needs_mat.side_effect = side_effect
            
            # Check if materialization is needed
            needs_mat = MaterializationResolver.needs_materialization(
                step, mock_orchestrator, mock_context
            )
            
            # Should not need materialization because flags are all False
            assert needs_mat is False
            
            # Verify that the materialization manager was called with the step
            mock_orchestrator.materialization_manager.needs_materialization.assert_called_once_with(step)
