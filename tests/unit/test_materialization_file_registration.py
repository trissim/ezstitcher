"""Tests for the MaterializationManager file registration behavior."""

from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from ezstitcher.io.materialization import (
    MaterializationManager, MaterializationPolicy, FailureMode, MaterializationError
)
from ezstitcher.io.overlay import OverlayMode, OverlayOperation


class TestMaterializationFileRegistration:
    """Tests for the MaterializationManager file registration behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock context and orchestrator
        self.mock_orchestrator = MagicMock()
        self.mock_orchestrator.storage_mode = "memory"
        self.mock_orchestrator.overlay_mode = OverlayMode.AUTO
        self.mock_orchestrator.storage_adapter = MagicMock()
        self.mock_orchestrator.file_manager = MagicMock()
        self.mock_orchestrator.microscope_handler = MagicMock()
        
        self.mock_context = MagicMock()
        self.mock_context.orchestrator = self.mock_orchestrator
        
    def test_register_single_file(self):
        """Test registering a single file."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.return_value = Path("/tmp/overlay/file.tif")
        
        # Test
        result = manager.register_file("/path/to/file.tif", "A01", "/path/to")
        
        # Verify
        assert result == Path("/tmp/overlay/file.tif")
        self.mock_orchestrator.storage_adapter.exists.assert_called_once_with("overlay_A01_file.tif")
        self.mock_orchestrator.storage_adapter.register_for_overlay.assert_called_once()
        assert len(manager.pending_operations) == 1
        
    def test_register_multiple_files(self):
        """Test registering multiple files."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.side_effect = [
            Path("/tmp/overlay/file1.tif"),
            Path("/tmp/overlay/file2.tif"),
            Path("/tmp/overlay/file3.tif")
        ]
        
        # Test
        result = manager.register_files(
            ["/path/to/file1.tif", "/path/to/file2.tif", "/path/to/file3.tif"],
            "A01",
            "/path/to"
        )
        
        # Verify
        assert len(result) == 3
        assert result["/path/to/file1.tif"] == Path("/tmp/overlay/file1.tif")
        assert result["/path/to/file2.tif"] == Path("/tmp/overlay/file2.tif")
        assert result["/path/to/file3.tif"] == Path("/tmp/overlay/file3.tif")
        assert self.mock_orchestrator.storage_adapter.exists.call_count == 3
        assert self.mock_orchestrator.storage_adapter.register_for_overlay.call_count == 3
        assert len(manager.pending_operations) == 3
        
    def test_register_files_with_some_failures(self):
        """Test registering multiple files with some failures."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Make exists return True for first two files, False for third
        self.mock_orchestrator.storage_adapter.exists.side_effect = [True, True, False]
        
        # Make register_for_overlay return a path for first file, raise exception for second
        self.mock_orchestrator.storage_adapter.register_for_overlay.side_effect = [
            Path("/tmp/overlay/file1.tif"),
            Exception("Test exception")
        ]
        
        # Test with LOG_AND_CONTINUE policy
        policy = MaterializationPolicy(failure_mode=FailureMode.LOG_AND_CONTINUE)
        manager = MaterializationManager(self.mock_context, policy)
        
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            result = manager.register_files(
                ["/path/to/file1.tif", "/path/to/file2.tif", "/path/to/file3.tif"],
                "A01",
                "/path/to"
            )
            
            # Verify
            assert len(result) == 1
            assert result["/path/to/file1.tif"] == Path("/tmp/overlay/file1.tif")
            assert "/path/to/file2.tif" not in result
            assert "/path/to/file3.tif" not in result
            assert self.mock_orchestrator.storage_adapter.exists.call_count == 3
            assert self.mock_orchestrator.storage_adapter.register_for_overlay.call_count == 2
            assert len(manager.pending_operations) == 1
            assert mock_logger.error.call_count == 1
            
    def test_register_pattern(self):
        """Test registering files using a pattern."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Mock path_list_from_pattern to return files for the pattern
        self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.return_value = [
            "/path/to/file1.tif",
            "/path/to/file2.tif"
        ]
        
        # Mock exists and register_for_overlay
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.side_effect = [
            Path("/tmp/overlay/file1.tif"),
            Path("/tmp/overlay/file2.tif")
        ]
        
        # Test
        result = manager.register_pattern("*.tif", "A01", "/path/to")
        
        # Verify
        assert len(result) == 2
        assert result["/path/to/file1.tif"] == Path("/tmp/overlay/file1.tif")
        assert result["/path/to/file2.tif"] == Path("/tmp/overlay/file2.tif")
        self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.assert_called_once_with(
            "/path/to", "*.tif"
        )
        assert self.mock_orchestrator.storage_adapter.exists.call_count == 2
        assert self.mock_orchestrator.storage_adapter.register_for_overlay.call_count == 2
        assert len(manager.pending_operations) == 2
        
    def test_register_patterns(self):
        """Test registering files using multiple patterns."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Mock path_list_from_pattern to return files for each pattern
        self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.side_effect = [
            ["/path/to/file1.tif", "/path/to/file2.tif"],
            ["/path/to/file3.tif"]
        ]
        
        # Mock exists and register_for_overlay
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.side_effect = [
            Path("/tmp/overlay/file1.tif"),
            Path("/tmp/overlay/file2.tif"),
            Path("/tmp/overlay/file3.tif")
        ]
        
        # Test
        result = manager.register_patterns(["*.tif", "*.jpg"], "A01", "/path/to")
        
        # Verify
        assert len(result) == 3
        assert result["/path/to/file1.tif"] == Path("/tmp/overlay/file1.tif")
        assert result["/path/to/file2.tif"] == Path("/tmp/overlay/file2.tif")
        assert result["/path/to/file3.tif"] == Path("/tmp/overlay/file3.tif")
        assert self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.call_count == 2
        assert self.mock_orchestrator.storage_adapter.exists.call_count == 3
        assert self.mock_orchestrator.storage_adapter.register_for_overlay.call_count == 3
        assert len(manager.pending_operations) == 3
        
    def test_register_pattern_with_no_matching_files(self):
        """Test registering a pattern with no matching files."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Mock path_list_from_pattern to return no files
        self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.return_value = []
        
        # Test
        result = manager.register_pattern("*.tif", "A01", "/path/to")
        
        # Verify
        assert len(result) == 0
        self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.assert_called_once_with(
            "/path/to", "*.tif"
        )
        assert self.mock_orchestrator.storage_adapter.exists.call_count == 0
        assert self.mock_orchestrator.storage_adapter.register_for_overlay.call_count == 0
        assert len(manager.pending_operations) == 0
        
    def test_prepare_for_step_with_input_flag(self):
        """Test prepare_for_step with requires_fs_input flag."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Create a step with requires_fs_input=True
        step = MagicMock()
        step.requires_fs_input = True
        step.requires_fs_output = False
        step.force_disk_output = False
        step.requires_legacy_fs = False
        
        # Mock get_patterns_for_well to return patterns
        with patch('ezstitcher.io.materialization.get_patterns_for_well') as mock_get_patterns:
            mock_get_patterns.return_value = ["*.tif"]
            
            # Mock path_list_from_pattern to return files
            self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.return_value = [
                "/path/to/file1.tif",
                "/path/to/file2.tif"
            ]
            
            # Mock exists and register_for_overlay
            self.mock_orchestrator.storage_adapter.exists.return_value = True
            self.mock_orchestrator.storage_adapter.register_for_overlay.side_effect = [
                Path("/tmp/overlay/file1.tif"),
                Path("/tmp/overlay/file2.tif")
            ]
            
            # Test
            result = manager.prepare_for_step(step, "A01", "/path/to")
            
            # Verify
            assert len(result) == 2
            assert result["/path/to/file1.tif"] == Path("/tmp/overlay/file1.tif")
            assert result["/path/to/file2.tif"] == Path("/tmp/overlay/file2.tif")
            mock_get_patterns.assert_called_once_with(
                "A01", "/path/to", self.mock_orchestrator.microscope_handler, recursive=True
            )
            assert self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.call_count == 1
            assert self.mock_orchestrator.storage_adapter.exists.call_count == 2
            assert self.mock_orchestrator.storage_adapter.register_for_overlay.call_count == 2
            assert len(manager.pending_operations) == 2
            
    def test_prepare_for_step_with_output_flag(self):
        """Test prepare_for_step with requires_fs_output flag."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Create a step with requires_fs_output=True
        step = MagicMock()
        step.requires_fs_input = False
        step.requires_fs_output = True
        step.force_disk_output = False
        step.requires_legacy_fs = False
        
        # Test
        result = manager.prepare_for_step(step, "A01", "/path/to")
        
        # Verify
        assert len(result) == 0
        assert len(manager.pending_operations) == 0
        
    def test_prepare_for_step_with_force_disk_output_flag(self):
        """Test prepare_for_step with force_disk_output flag."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Create a step with force_disk_output=True
        step = MagicMock()
        step.requires_fs_input = False
        step.requires_fs_output = False
        step.force_disk_output = True
        step.requires_legacy_fs = False
        
        # Test
        result = manager.prepare_for_step(step, "A01", "/path/to")
        
        # Verify
        assert len(result) == 0
        assert len(manager.pending_operations) == 0
        
    def test_prepare_for_step_with_requires_legacy_fs_flag(self):
        """Test prepare_for_step with requires_legacy_fs flag."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Create a step with requires_legacy_fs=True
        step = MagicMock()
        step.requires_fs_input = False
        step.requires_fs_output = False
        step.force_disk_output = False
        step.requires_legacy_fs = True
        
        # Mock get_patterns_for_well to return patterns
        with patch('ezstitcher.io.materialization.get_patterns_for_well') as mock_get_patterns:
            mock_get_patterns.return_value = ["*.tif"]
            
            # Mock path_list_from_pattern to return files
            self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.return_value = [
                "/path/to/file1.tif",
                "/path/to/file2.tif"
            ]
            
            # Mock exists and register_for_overlay
            self.mock_orchestrator.storage_adapter.exists.return_value = True
            self.mock_orchestrator.storage_adapter.register_for_overlay.side_effect = [
                Path("/tmp/overlay/file1.tif"),
                Path("/tmp/overlay/file2.tif")
            ]
            
            # Test
            result = manager.prepare_for_step(step, "A01", "/path/to")
            
            # Verify
            assert len(result) == 2
            assert result["/path/to/file1.tif"] == Path("/tmp/overlay/file1.tif")
            assert result["/path/to/file2.tif"] == Path("/tmp/overlay/file2.tif")
            mock_get_patterns.assert_called_once_with(
                "A01", "/path/to", self.mock_orchestrator.microscope_handler, recursive=True
            )
            assert self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.call_count == 1
            assert self.mock_orchestrator.storage_adapter.exists.call_count == 2
            assert self.mock_orchestrator.storage_adapter.register_for_overlay.call_count == 2
            assert len(manager.pending_operations) == 2
