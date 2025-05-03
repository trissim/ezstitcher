"""Tests for the MaterializationManager class."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ezstitcher.io.materialization import (
    MaterializationManager, MaterializationPolicy, FailureMode, MaterializationError
)
from ezstitcher.io.overlay import OverlayMode, OverlayOperation


class TestMaterializationManager:
    """Tests for the MaterializationManager class."""

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
        
        # Create a step with flags
        self.step_with_flags = MagicMock()
        self.step_with_flags.requires_fs_input = True
        self.step_with_flags.requires_fs_output = False
        self.step_with_flags.force_disk_output = False
        self.step_with_flags.requires_legacy_fs = False
        
        # Create a step without flags
        self.step_without_flags = MagicMock()
        self.step_without_flags.requires_fs_input = False
        self.step_without_flags.requires_fs_output = False
        self.step_without_flags.force_disk_output = False
        self.step_without_flags.requires_legacy_fs = False

    def test_needs_materialization_with_legacy_storage(self):
        """Test needs_materialization with legacy storage mode."""
        # Set up
        self.mock_orchestrator.storage_mode = "legacy"
        manager = MaterializationManager(self.mock_context)
        
        # Test
        result = manager.needs_materialization(self.step_with_flags)
        
        # Verify
        assert result is False

    def test_needs_materialization_with_disabled_overlay(self):
        """Test needs_materialization with disabled overlay mode."""
        # Set up
        self.mock_orchestrator.overlay_mode = OverlayMode.DISABLED
        manager = MaterializationManager(self.mock_context)
        
        # Test
        result = manager.needs_materialization(self.step_with_flags)
        
        # Verify
        assert result is False

    def test_needs_materialization_with_force_memory(self):
        """Test needs_materialization with force_memory policy."""
        # Set up
        policy = MaterializationPolicy(force_memory=True)
        manager = MaterializationManager(self.mock_context, policy)
        
        # Test
        result = manager.needs_materialization(self.step_with_flags)
        
        # Verify
        assert result is False

    def test_needs_materialization_with_force_disk(self):
        """Test needs_materialization with force_disk policy."""
        # Set up
        policy = MaterializationPolicy(force_disk=True)
        manager = MaterializationManager(self.mock_context, policy)
        
        # Test
        result = manager.needs_materialization(self.step_without_flags)
        
        # Verify
        assert result is True

    def test_needs_materialization_with_flags(self):
        """Test needs_materialization with step flags."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Test
        result_with_flags = manager.needs_materialization(self.step_with_flags)
        result_without_flags = manager.needs_materialization(self.step_without_flags)
        
        # Verify
        assert result_with_flags is True
        assert result_without_flags is False

    def test_needs_materialization_with_requires_legacy_fs(self):
        """Test needs_materialization with requires_legacy_fs flag."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        step_with_legacy_fs = MagicMock()
        step_with_legacy_fs.requires_fs_input = False
        step_with_legacy_fs.requires_fs_output = False
        step_with_legacy_fs.force_disk_output = False
        step_with_legacy_fs.requires_legacy_fs = True
        
        # Test
        result = manager.needs_materialization(step_with_legacy_fs)
        
        # Verify
        assert result is True

    def test_handle_failure_with_fail_fast(self):
        """Test _handle_failure with FAIL_FAST policy."""
        # Set up
        policy = MaterializationPolicy(failure_mode=FailureMode.FAIL_FAST)
        manager = MaterializationManager(self.mock_context, policy)
        
        # Test
        with pytest.raises(MaterializationError):
            manager._handle_failure("Test error", Exception("Test exception"))

    def test_handle_failure_with_log_and_continue(self):
        """Test _handle_failure with LOG_AND_CONTINUE policy."""
        # Set up
        policy = MaterializationPolicy(failure_mode=FailureMode.LOG_AND_CONTINUE)
        manager = MaterializationManager(self.mock_context, policy)
        
        # Test
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            manager._handle_failure("Test error", Exception("Test exception"))
            
            # Verify
            mock_logger.error.assert_called_once()

    def test_construct_key(self):
        """Test _construct_key method."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Test
        key1 = manager._construct_key("A01", "/path/to/file.tif", "/path/to")
        key2 = manager._construct_key("A01", "/path/to/subdir/file.tif", "/path/to")
        key3 = manager._construct_key("A01", "file.tif", "/path/to")
        
        # Verify
        assert key1 == "overlay_A01_file.tif"
        assert key2 == "overlay_A01_subdir/file.tif"
        assert key3 == "overlay_A01_file.tif"

    def test_register_file(self):
        """Test register_file method."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.return_value = Path("/tmp/overlay/file.tif")
        
        # Test
        result = manager.register_file("/path/to/file.tif", "A01", "/path/to")
        
        # Verify
        assert result == Path("/tmp/overlay/file.tif")
        self.mock_orchestrator.storage_adapter.exists.assert_called_once()
        self.mock_orchestrator.storage_adapter.register_for_overlay.assert_called_once()

    def test_register_files(self):
        """Test register_files method."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Mock register_file to return a path for each file
        manager.register_file = MagicMock(side_effect=[
            Path("/tmp/overlay/file1.tif"),
            Path("/tmp/overlay/file2.tif"),
            None  # Simulate a failure for the third file
        ])
        
        # Test
        result = manager.register_files(
            ["/path/to/file1.tif", "/path/to/file2.tif", "/path/to/file3.tif"],
            "A01",
            "/path/to"
        )
        
        # Verify
        assert len(result) == 2
        assert result["/path/to/file1.tif"] == Path("/tmp/overlay/file1.tif")
        assert result["/path/to/file2.tif"] == Path("/tmp/overlay/file2.tif")
        assert "/path/to/file3.tif" not in result
        assert manager.register_file.call_count == 3

    def test_register_patterns(self):
        """Test register_patterns method."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Mock path_list_from_pattern to return files for each pattern
        self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.side_effect = [
            ["/path/to/file1.tif", "/path/to/file2.tif"],
            ["/path/to/file3.tif"]
        ]
        
        # Mock register_files to return a dictionary of paths
        manager.register_files = MagicMock(return_value={
            "/path/to/file1.tif": Path("/tmp/overlay/file1.tif"),
            "/path/to/file2.tif": Path("/tmp/overlay/file2.tif"),
            "/path/to/file3.tif": Path("/tmp/overlay/file3.tif")
        })
        
        # Test
        result = manager.register_patterns(
            ["pattern1", "pattern2"],
            "A01",
            "/path/to"
        )
        
        # Verify
        assert len(result) == 3
        assert self.mock_orchestrator.microscope_handler.parser.path_list_from_pattern.call_count == 2
        manager.register_files.assert_called_once_with(
            ["/path/to/file1.tif", "/path/to/file2.tif", "/path/to/file3.tif"],
            "A01",
            "/path/to",
            "read",
            True
        )

    def test_prepare_for_step(self):
        """Test prepare_for_step method."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Mock needs_materialization to return True
        manager.needs_materialization = MagicMock(return_value=True)
        
        # Mock get_patterns_for_well to return patterns
        with patch('ezstitcher.io.materialization.get_patterns_for_well') as mock_get_patterns:
            mock_get_patterns.return_value = ["pattern1", "pattern2"]
            
            # Mock register_patterns to return a dictionary of paths
            manager.register_patterns = MagicMock(return_value={
                "/path/to/file1.tif": Path("/tmp/overlay/file1.tif"),
                "/path/to/file2.tif": Path("/tmp/overlay/file2.tif")
            })
            
            # Test
            result = manager.prepare_for_step(self.step_with_flags, "A01", "/path/to")
            
            # Verify
            assert len(result) == 2
            manager.needs_materialization.assert_called_once_with(self.step_with_flags)
            mock_get_patterns.assert_called_once_with(
                "A01", "/path/to", self.mock_orchestrator.microscope_handler, recursive=True
            )
            manager.register_patterns.assert_called_once_with(
                ["pattern1", "pattern2"], "A01", "/path/to", "read", True
            )

    def test_execute_pending_operations(self):
        """Test execute_pending_operations method."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Add some pending operations
        manager.pending_operations = {
            "key1": MagicMock(),
            "key2": MagicMock(),
            "key3": MagicMock()
        }
        
        # Mock execute_overlay_operation to succeed for key1 and key2, fail for key3
        self.mock_orchestrator.storage_adapter.execute_overlay_operation.side_effect = [
            True,  # key1 succeeds
            True,  # key2 succeeds
            False  # key3 fails
        ]
        
        # Test
        result = manager.execute_pending_operations()
        
        # Verify
        assert result == 2
        assert self.mock_orchestrator.storage_adapter.execute_overlay_operation.call_count == 3
        assert len(manager.pending_operations) == 1
        assert "key3" in manager.pending_operations

    def test_cleanup_operations(self):
        """Test cleanup_operations method."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        
        # Mock cleanup_overlay_operations to return a count
        self.mock_orchestrator.storage_adapter.cleanup_overlay_operations.return_value = 3
        
        # Test
        result = manager.cleanup_operations()
        
        # Verify
        assert result == 3
        self.mock_orchestrator.storage_adapter.cleanup_overlay_operations.assert_called_once_with(
            self.mock_orchestrator.file_manager
        )
