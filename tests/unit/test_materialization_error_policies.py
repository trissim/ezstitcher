"""Tests for the MaterializationManager error handling policies."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ezstitcher.io.materialization import (
    MaterializationManager, MaterializationPolicy, FailureMode, MaterializationError
)
from ezstitcher.io.overlay import OverlayMode, OverlayOperation


class TestMaterializationErrorPolicies:
    """Tests for the MaterializationManager error handling policies."""

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
        
    def test_fail_fast_policy_on_register_file(self):
        """Test FAIL_FAST policy on register_file."""
        # Set up
        policy = MaterializationPolicy(failure_mode=FailureMode.FAIL_FAST)
        manager = MaterializationManager(self.mock_context, policy)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.side_effect = Exception("Test exception")
        
        # Test
        with pytest.raises(MaterializationError):
            manager.register_file("/path/to/file.tif", "A01", "/path/to")
            
    def test_log_and_continue_policy_on_register_file(self):
        """Test LOG_AND_CONTINUE policy on register_file."""
        # Set up
        policy = MaterializationPolicy(failure_mode=FailureMode.LOG_AND_CONTINUE)
        manager = MaterializationManager(self.mock_context, policy)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.side_effect = Exception("Test exception")
        
        # Test
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            result = manager.register_file("/path/to/file.tif", "A01", "/path/to")
            
            # Verify
            assert result is None
            mock_logger.error.assert_called_once()
            
    def test_fallback_to_disk_policy_on_register_file(self):
        """Test FALLBACK_TO_DISK policy on register_file."""
        # Set up
        policy = MaterializationPolicy(failure_mode=FailureMode.FALLBACK_TO_DISK)
        manager = MaterializationManager(self.mock_context, policy)
        self.mock_orchestrator.storage_adapter.exists.return_value = True
        self.mock_orchestrator.storage_adapter.register_for_overlay.side_effect = Exception("Test exception")
        
        # Test
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            result = manager.register_file("/path/to/file.tif", "A01", "/path/to")
            
            # Verify
            assert result is None
            mock_logger.error.assert_called_once()
            
    def test_fail_fast_policy_on_execute_operations(self):
        """Test FAIL_FAST policy on execute_pending_operations."""
        # Set up
        policy = MaterializationPolicy(failure_mode=FailureMode.FAIL_FAST)
        manager = MaterializationManager(self.mock_context, policy)
        
        # Add some pending operations
        manager.pending_operations = {
            "key1": MagicMock(),
            "key2": MagicMock()
        }
        
        # Make execute_overlay_operation raise an exception
        self.mock_orchestrator.storage_adapter.execute_overlay_operation.side_effect = Exception("Test exception")
        
        # Test
        with pytest.raises(MaterializationError):
            manager.execute_pending_operations()
            
    def test_log_and_continue_policy_on_execute_operations(self):
        """Test LOG_AND_CONTINUE policy on execute_pending_operations."""
        # Set up
        policy = MaterializationPolicy(failure_mode=FailureMode.LOG_AND_CONTINUE)
        manager = MaterializationManager(self.mock_context, policy)
        
        # Add some pending operations
        manager.pending_operations = {
            "key1": MagicMock(),
            "key2": MagicMock()
        }
        
        # Make execute_overlay_operation raise an exception
        self.mock_orchestrator.storage_adapter.execute_overlay_operation.side_effect = Exception("Test exception")
        
        # Test
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            result = manager.execute_pending_operations()
            
            # Verify
            assert result == 0
            assert mock_logger.error.call_count == 2
            assert len(manager.pending_operations) == 2
            
    def test_fallback_to_disk_policy_on_execute_operations(self):
        """Test FALLBACK_TO_DISK policy on execute_pending_operations."""
        # Set up
        policy = MaterializationPolicy(failure_mode=FailureMode.FALLBACK_TO_DISK)
        manager = MaterializationManager(self.mock_context, policy)
        
        # Add some pending operations
        manager.pending_operations = {
            "key1": MagicMock(),
            "key2": MagicMock()
        }
        
        # Make execute_overlay_operation raise an exception
        self.mock_orchestrator.storage_adapter.execute_overlay_operation.side_effect = Exception("Test exception")
        
        # Test
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            result = manager.execute_pending_operations()
            
            # Verify
            assert result == 0
            assert mock_logger.error.call_count == 2
            assert len(manager.pending_operations) == 2
            
    def test_fail_fast_policy_on_cleanup_operations(self):
        """Test FAIL_FAST policy on cleanup_operations."""
        # Set up
        policy = MaterializationPolicy(failure_mode=FailureMode.FAIL_FAST)
        manager = MaterializationManager(self.mock_context, policy)
        
        # Make cleanup_overlay_operations raise an exception
        self.mock_orchestrator.storage_adapter.cleanup_overlay_operations.side_effect = Exception("Test exception")
        
        # Test
        with pytest.raises(MaterializationError):
            manager.cleanup_operations()
            
    def test_log_and_continue_policy_on_cleanup_operations(self):
        """Test LOG_AND_CONTINUE policy on cleanup_operations."""
        # Set up
        policy = MaterializationPolicy(failure_mode=FailureMode.LOG_AND_CONTINUE)
        manager = MaterializationManager(self.mock_context, policy)
        
        # Make cleanup_overlay_operations raise an exception
        self.mock_orchestrator.storage_adapter.cleanup_overlay_operations.side_effect = Exception("Test exception")
        
        # Test
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            result = manager.cleanup_operations()
            
            # Verify
            assert result == 0
            mock_logger.error.assert_called_once()
            
    def test_fallback_to_disk_policy_on_cleanup_operations(self):
        """Test FALLBACK_TO_DISK policy on cleanup_operations."""
        # Set up
        policy = MaterializationPolicy(failure_mode=FailureMode.FALLBACK_TO_DISK)
        manager = MaterializationManager(self.mock_context, policy)
        
        # Make cleanup_overlay_operations raise an exception
        self.mock_orchestrator.storage_adapter.cleanup_overlay_operations.side_effect = Exception("Test exception")
        
        # Test
        with patch('ezstitcher.io.materialization.logger') as mock_logger:
            result = manager.cleanup_operations()
            
            # Verify
            assert result == 0
            mock_logger.error.assert_called_once()
            
    def test_policy_for_context_testing(self):
        """Test MaterializationPolicy.for_context with testing context."""
        # Test
        policy = MaterializationPolicy.for_context("testing")
        
        # Verify
        assert policy.method == MaterializationPolicy.DEFAULT_METHOD
        assert policy.failure_mode == FailureMode.LOG_AND_CONTINUE
        assert policy.force_memory is True
        assert policy.force_disk is False
        
    def test_policy_for_context_benchmark(self):
        """Test MaterializationPolicy.for_context with benchmark context."""
        # Test
        policy = MaterializationPolicy.for_context("benchmark")
        
        # Verify
        assert policy.method == MaterializationPolicy.DEFAULT_METHOD
        assert policy.failure_mode == FailureMode.FAIL_FAST
        assert policy.force_memory is False
        assert policy.force_disk is True
        
    def test_policy_for_context_production(self):
        """Test MaterializationPolicy.for_context with production context."""
        # Test
        policy = MaterializationPolicy.for_context("production")
        
        # Verify
        assert policy.method == MaterializationMethod.SYMLINK
        assert policy.failure_mode == FailureMode.FAIL_FAST
        assert policy.force_memory is False
        assert policy.force_disk is False
        
    def test_policy_for_context_unknown(self):
        """Test MaterializationPolicy.for_context with unknown context."""
        # Test
        policy = MaterializationPolicy.for_context("unknown")
        
        # Verify
        assert policy.method == MaterializationPolicy.DEFAULT_METHOD
        assert policy.failure_mode == FailureMode.FAIL_FAST
        assert policy.force_memory is False
        assert policy.force_disk is False
