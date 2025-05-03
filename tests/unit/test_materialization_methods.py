"""Tests for the MaterializationManager materialization methods."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ezstitcher.io.materialization import (
    MaterializationManager, MaterializationPolicy, FailureMode, MaterializationError,
    MaterializationMethod
)
from ezstitcher.io.overlay import OverlayMode, OverlayOperation


class TestMaterializationMethods:
    """Tests for the MaterializationManager materialization methods."""

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
        
    def test_register_file_with_copy_method(self):
        """Test register_file with COPY method."""
        # Set up
        policy = MaterializationPolicy(method=MaterializationMethod.COPY)
        manager = MaterializationManager(self.mock_context, policy)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.return_value = Path("/tmp/overlay/file.tif")
        
        # Test
        result = manager.register_file("/path/to/file.tif", "A01", "/path/to")
        
        # Verify
        assert result == Path("/tmp/overlay/file.tif")
        self.mock_orchestrator.storage_adapter.register_for_overlay.assert_called_once_with(
            "overlay_A01_file.tif", 
            operation_type="read", 
            cleanup=True, 
            method=MaterializationMethod.COPY
        )
        
    def test_register_file_with_symlink_method(self):
        """Test register_file with SYMLINK method."""
        # Set up
        policy = MaterializationPolicy(method=MaterializationMethod.SYMLINK)
        manager = MaterializationManager(self.mock_context, policy)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.return_value = Path("/tmp/overlay/file.tif")
        
        # Test
        result = manager.register_file("/path/to/file.tif", "A01", "/path/to")
        
        # Verify
        assert result == Path("/tmp/overlay/file.tif")
        self.mock_orchestrator.storage_adapter.register_for_overlay.assert_called_once_with(
            "overlay_A01_file.tif", 
            operation_type="read", 
            cleanup=True, 
            method=MaterializationMethod.SYMLINK
        )
        
    def test_register_file_with_hardlink_method(self):
        """Test register_file with HARDLINK method."""
        # Set up
        policy = MaterializationPolicy(method=MaterializationMethod.HARDLINK)
        manager = MaterializationManager(self.mock_context, policy)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.return_value = Path("/tmp/overlay/file.tif")
        
        # Test
        result = manager.register_file("/path/to/file.tif", "A01", "/path/to")
        
        # Verify
        assert result == Path("/tmp/overlay/file.tif")
        self.mock_orchestrator.storage_adapter.register_for_overlay.assert_called_once_with(
            "overlay_A01_file.tif", 
            operation_type="read", 
            cleanup=True, 
            method=MaterializationMethod.HARDLINK
        )
        
    def test_register_file_with_custom_operation_type(self):
        """Test register_file with custom operation_type."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.return_value = Path("/tmp/overlay/file.tif")
        
        # Test
        result = manager.register_file("/path/to/file.tif", "A01", "/path/to", operation_type="write")
        
        # Verify
        assert result == Path("/tmp/overlay/file.tif")
        self.mock_orchestrator.storage_adapter.register_for_overlay.assert_called_once_with(
            "overlay_A01_file.tif", 
            operation_type="write", 
            cleanup=True, 
            method=MaterializationMethod.COPY
        )
        
    def test_register_file_with_custom_cleanup(self):
        """Test register_file with custom cleanup."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.return_value = Path("/tmp/overlay/file.tif")
        
        # Test
        result = manager.register_file("/path/to/file.tif", "A01", "/path/to", cleanup=False)
        
        # Verify
        assert result == Path("/tmp/overlay/file.tif")
        self.mock_orchestrator.storage_adapter.register_for_overlay.assert_called_once_with(
            "overlay_A01_file.tif", 
            operation_type="read", 
            cleanup=False, 
            method=MaterializationMethod.COPY
        )
        
    def test_register_file_with_nonexistent_key(self):
        """Test register_file with nonexistent key."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        self.mock_orchestrator.storage_adapter.exists.return_value = False
        
        # Test
        result = manager.register_file("/path/to/file.tif", "A01", "/path/to")
        
        # Verify
        assert result is None
        self.mock_orchestrator.storage_adapter.exists.assert_called_once_with("overlay_A01_file.tif")
        self.mock_orchestrator.storage_adapter.register_for_overlay.assert_not_called()
        
    def test_register_file_with_exception(self):
        """Test register_file with exception."""
        # Set up
        manager = MaterializationManager(self.mock_context)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.side_effect = Exception("Test exception")
        
        # Test with FAIL_FAST policy
        policy = MaterializationPolicy(failure_mode=FailureMode.FAIL_FAST)
        manager = MaterializationManager(self.mock_context, policy)
        
        with pytest.raises(MaterializationError):
            manager.register_file("/path/to/file.tif", "A01", "/path/to")
            
        # Test with LOG_AND_CONTINUE policy
        policy = MaterializationPolicy(failure_mode=FailureMode.LOG_AND_CONTINUE)
        manager = MaterializationManager(self.mock_context, policy)
        
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            result = manager.register_file("/path/to/file.tif", "A01", "/path/to")
            
            # Verify
            assert result is None
            mock_logger.error.assert_called_once()
            
        # Test with FALLBACK_TO_DISK policy
        policy = MaterializationPolicy(failure_mode=FailureMode.FALLBACK_TO_DISK)
        manager = MaterializationManager(self.mock_context, policy)
        
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            result = manager.register_file("/path/to/file.tif", "A01", "/path/to")
            
            # Verify
            assert result is None
            mock_logger.error.assert_called_once()
