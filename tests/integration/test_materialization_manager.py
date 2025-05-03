"""
Integration tests for the MaterializationManager.
"""

import unittest
from unittest.mock import Mock, patch
import os
from pathlib import Path
import tempfile
import shutil
import numpy as np

from ezstitcher.io.materialization import MaterializationManager, MaterializationPolicy, FailureMode
from ezstitcher.io.overlay import OverlayMode
from ezstitcher.core.steps import Step


class TestMaterializationManager(unittest.TestCase):
    """Integration tests for the MaterializationManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / "input"
        self.output_dir = Path(self.test_dir) / "output"
        self.overlay_dir = Path(self.test_dir) / "overlay"
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.overlay_dir, exist_ok=True)

        # Create a test image
        self.test_image = np.ones((10, 10), dtype=np.uint8)
        
        # Create a mock step
        self.step = Mock(spec=Step)
        self.step.name = "test_step"
        self.step.requires_fs_input = True
        self.step.requires_fs_output = False
        self.step.force_disk_output = False
        self.step.requires_legacy_fs = False
        self.step.needs_materialization.return_value = True
        
        # Create a mock microscope handler
        self.microscope_handler = Mock()
        self.microscope_handler.parser = Mock()
        self.microscope_handler.parser.path_list_from_pattern.return_value = ["test_image.tif"]
        
        # Create a mock storage adapter
        self.storage_adapter = Mock()
        self.storage_adapter.exists.return_value = True
        self.storage_adapter.read.return_value = self.test_image
        self.storage_adapter.register_for_overlay.return_value = Path(self.overlay_dir) / "test_image.tif"
        self.storage_adapter.overlay_operations = {}
        self.storage_adapter.execute_overlay_operation.return_value = True
        self.storage_adapter.cleanup_overlay_operations.return_value = 1
        
        # Create a mock file manager
        self.file_manager = Mock()
        self.file_manager.exists.return_value = True
        self.file_manager.write_image.return_value = True
        
        # Create a mock orchestrator
        self.orchestrator = Mock()
        self.orchestrator.storage_mode = "memory"
        self.orchestrator.overlay_mode = OverlayMode.MEMORY_OVERLAY
        self.orchestrator.microscope_handler = self.microscope_handler
        self.orchestrator.storage_adapter = self.storage_adapter
        self.orchestrator.file_manager = self.file_manager
        self.orchestrator.overlay_root = self.overlay_dir
        
        # Create a mock context
        self.context = Mock()
        self.context.orchestrator = self.orchestrator
        
        # Create the materialization manager
        self.manager = MaterializationManager(self.context)

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_needs_materialization(self):
        """Test the needs_materialization method."""
        # Test with a step that needs materialization
        self.step.needs_materialization.return_value = True
        self.assertTrue(self.manager.needs_materialization(self.step))
        
        # Test with a step that doesn't need materialization
        self.step.needs_materialization.return_value = False
        self.assertFalse(self.manager.needs_materialization(self.step))
        
        # Test with force_memory policy
        policy = MaterializationPolicy(force_memory=True)
        self.manager.policy = policy
        self.assertFalse(self.manager.needs_materialization(self.step))
        
        # Test with force_disk policy
        policy = MaterializationPolicy(force_disk=True)
        self.manager.policy = policy
        self.assertTrue(self.manager.needs_materialization(self.step))
        
        # Test with legacy storage mode
        self.orchestrator.storage_mode = "legacy"
        self.assertFalse(self.manager.needs_materialization(self.step))
        
        # Test with overlay disabled
        self.orchestrator.storage_mode = "memory"
        self.orchestrator.overlay_mode = OverlayMode.DISABLED
        self.assertFalse(self.manager.needs_materialization(self.step))

    def test_register_file(self):
        """Test the register_file method."""
        # Test registering a file
        disk_path = self.manager.register_file(
            file_path="test_image.tif",
            well="A01",
            input_dir=self.input_dir,
            operation_type="read",
            cleanup=True
        )
        self.assertEqual(disk_path, Path(self.overlay_dir) / "test_image.tif")
        self.storage_adapter.register_for_overlay.assert_called_once()
        
        # Test registering a file that doesn't exist
        self.storage_adapter.exists.return_value = False
        disk_path = self.manager.register_file(
            file_path="nonexistent.tif",
            well="A01",
            input_dir=self.input_dir,
            operation_type="read",
            cleanup=True
        )
        self.assertIsNone(disk_path)

    def test_prepare_for_step(self):
        """Test the prepare_for_step method."""
        # Mock the get_patterns_for_well function
        with patch('ezstitcher.core.pattern_resolver.get_patterns_for_well') as mock_get_patterns:
            mock_get_patterns.return_value = ["pattern1", "pattern2"]
            
            # Test preparing for a step
            result = self.manager.prepare_for_step(
                step=self.step,
                well="A01",
                input_dir=self.input_dir
            )
            self.assertEqual(len(result), 2)  # Two patterns, one file each
            mock_get_patterns.assert_called_once_with(
                "A01", self.input_dir, self.microscope_handler, recursive=True
            )
            
            # Test with no patterns
            mock_get_patterns.return_value = []
            result = self.manager.prepare_for_step(
                step=self.step,
                well="A01",
                input_dir=self.input_dir
            )
            self.assertEqual(len(result), 0)

    def test_execute_pending_operations(self):
        """Test the execute_pending_operations method."""
        # Add a pending operation
        self.manager.pending_operations = {"key1": Mock()}
        
        # Test executing pending operations
        executed = self.manager.execute_pending_operations()
        self.assertEqual(executed, 1)
        self.storage_adapter.execute_overlay_operation.assert_called_once()
        
        # Test with no pending operations
        self.manager.pending_operations = {}
        executed = self.manager.execute_pending_operations()
        self.assertEqual(executed, 0)

    def test_cleanup_operations(self):
        """Test the cleanup_operations method."""
        # Test cleaning up operations
        cleaned = self.manager.cleanup_operations()
        self.assertEqual(cleaned, 1)
        self.storage_adapter.cleanup_overlay_operations.assert_called_once_with(self.file_manager)
        
        # Test with lazy cleanup
        policy = MaterializationPolicy(lazy_cleanup=True)
        self.manager.policy = policy
        cleaned = self.manager.cleanup_operations()
        self.assertEqual(cleaned, 0)

    def test_failure_modes(self):
        """Test different failure modes."""
        # Test FAIL_FAST mode
        policy = MaterializationPolicy(failure_mode=FailureMode.FAIL_FAST)
        self.manager.policy = policy
        self.storage_adapter.register_for_overlay.side_effect = Exception("Test exception")
        
        with self.assertRaises(Exception):
            self.manager.register_file(
                file_path="test_image.tif",
                well="A01",
                input_dir=self.input_dir,
                operation_type="read",
                cleanup=True
            )
        
        # Test LOG_AND_CONTINUE mode
        policy = MaterializationPolicy(failure_mode=FailureMode.LOG_AND_CONTINUE)
        self.manager.policy = policy
        disk_path = self.manager.register_file(
            file_path="test_image.tif",
            well="A01",
            input_dir=self.input_dir,
            operation_type="read",
            cleanup=True
        )
        self.assertIsNone(disk_path)


if __name__ == '__main__':
    unittest.main()
