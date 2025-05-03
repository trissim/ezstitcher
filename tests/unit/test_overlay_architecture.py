"""Tests for the overlay architecture support for filesystem-only backends."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from ezstitcher.core.pipeline import ProcessingContext, Pipeline, StepResult
from ezstitcher.core.steps import Step
from ezstitcher.io.storage_adapter import MemoryStorageAdapter
from ezstitcher.io.overlay import OverlayMode, OverlayOperation
from ezstitcher.io.filemanager import FileManager


class TestOverlayArchitecture:
    """Test the overlay architecture support for filesystem-only backends."""

    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.overlay_dir = self.temp_dir / "overlay"
        self.overlay_dir.mkdir(exist_ok=True)

        # Create a file manager for testing
        self.file_manager = FileManager(backend="disk")

    def teardown_method(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_overlay_operation_registration(self):
        """Test registration of overlay operations."""
        # Create a storage adapter
        adapter = MemoryStorageAdapter()
        adapter.configure_overlay(OverlayMode.AUTO, self.overlay_dir)

        # Write some data to the adapter
        test_data = np.ones((10, 10), dtype=np.uint8)
        adapter.write("test_key", test_data)

        # Register for overlay
        disk_path = adapter.register_for_overlay("test_key", operation_type="read", cleanup=True)

        # Verify the operation was registered
        assert disk_path is not None
        assert "test_key" in adapter.overlay_operations
        assert adapter.overlay_operations["test_key"].disk_path == disk_path
        assert not adapter.overlay_operations["test_key"].executed

    def test_overlay_operation_execution(self):
        """Test execution of overlay operations."""
        # Create a storage adapter
        adapter = MemoryStorageAdapter()
        adapter.configure_overlay(OverlayMode.AUTO, self.overlay_dir)

        # Write some data to the adapter
        test_data = np.ones((10, 10), dtype=np.uint8)
        adapter.write("test_key", test_data)

        # Register for overlay
        disk_path = adapter.register_for_overlay("test_key", operation_type="read", cleanup=True)

        # Ensure the directory exists
        self.file_manager.ensure_directory(disk_path.parent)

        # Execute the operation
        success = adapter.execute_overlay_operation("test_key", self.file_manager)

        # Verify the operation was executed
        assert success
        assert adapter.overlay_operations["test_key"].executed

    def test_overlay_operation_cleanup(self):
        """Test cleanup of overlay operations."""
        # Create a storage adapter
        adapter = MemoryStorageAdapter()
        adapter.configure_overlay(OverlayMode.AUTO, self.overlay_dir)

        # Write some data to the adapter
        test_data = np.ones((10, 10), dtype=np.uint8)
        adapter.write("test_key", test_data)

        # Register for overlay
        disk_path = adapter.register_for_overlay("test_key", operation_type="write", cleanup=True)

        # Ensure the directory exists
        self.file_manager.ensure_directory(disk_path.parent)

        # Execute the operation
        success = adapter.execute_overlay_operation("test_key", self.file_manager)
        assert success, "Failed to execute overlay operation"

        # Verify the file was written
        assert self.file_manager.exists(disk_path), f"File was not created at {disk_path}"

        # Clean up the operation
        cleaned = adapter.cleanup_overlay_operations(self.file_manager)

        # Verify the operation was cleaned up
        assert cleaned == 1, f"Expected 1 cleaned operation, got {cleaned}"
        assert "test_key" not in adapter.overlay_operations, "Operation was not removed from overlay_operations"
        assert not self.file_manager.exists(disk_path), f"File was not deleted at {disk_path}"

    def test_overlay_disabled_mode(self):
        """Test overlay in disabled mode."""
        # Create a storage adapter with overlay disabled
        adapter = MemoryStorageAdapter()
        adapter.configure_overlay(OverlayMode.DISABLED, self.overlay_dir)

        # Write some data to the adapter
        test_data = np.ones((10, 10), dtype=np.uint8)
        adapter.write("test_key", test_data)

        # Try to register for overlay
        disk_path = adapter.register_for_overlay("test_key", operation_type="read", cleanup=True)

        # Verify the operation was not registered
        assert disk_path is None
        assert "test_key" not in adapter.overlay_operations
