"""
Unit tests for logical key path resolution.
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import os
import shutil
import numpy as np
from pathlib import Path

from ezstitcher.io.storage_adapter import StorageAdapter, MemoryStorageAdapter
from ezstitcher.io.filemanager import FileManager
from ezstitcher.io.virtual_path import VirtualPath, PhysicalPath


class TestLogicalKeyResolution(unittest.TestCase):
    """Test logical key path resolution."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test image
        self.test_image = np.ones((10, 10), dtype=np.uint8)
        
        # Create a storage adapter
        self.storage_adapter = MemoryStorageAdapter()
        
        # Create a file manager
        self.file_manager = FileManager(
            root_dir=self.temp_dir,
            backend="memory",
            storage_adapter=self.storage_adapter
        )
        
        # Register some patterns
        self.storage_adapter.register_pattern(
            r"([^/]+)/([^/]+)/([^/]+)\.tif",
            f"{self.temp_dir}/$1/$2/$3.tif"
        )
        self.storage_adapter.register_pattern(
            r"([^/]+)_([^/]+)\.tif",
            f"{self.temp_dir}/$1/$2.tif"
        )
        
        # Create some test directories
        os.makedirs(os.path.join(self.temp_dir, "A01", "ch0"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "A02"), exist_ok=True)
        
        # Create some test files
        with open(os.path.join(self.temp_dir, "A01", "ch0", "img1.tif"), "wb") as f:
            f.write(b"test image data")
        with open(os.path.join(self.temp_dir, "A02", "ch1.tif"), "wb") as f:
            f.write(b"test image data")

    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_register_pattern(self):
        """Test registering a pattern."""
        adapter = MemoryStorageAdapter()
        adapter.register_pattern(r"test/(\d+)", "path/to/$1")
        self.assertEqual(len(adapter.key_patterns), 1)

    def test_register_mapping(self):
        """Test registering a direct mapping."""
        adapter = MemoryStorageAdapter()
        adapter.register_mapping("test_key", "/path/to/file")
        self.assertEqual(len(adapter.key_mappings), 1)
        self.assertIsInstance(adapter.key_mappings["test_key"], VirtualPath)

    def test_resolve_key_with_pattern(self):
        """Test resolving a key using a pattern."""
        # Test with a pattern that matches
        path = self.storage_adapter.resolve_key("A01/ch0/img1.tif")
        self.assertIsNotNone(path)
        self.assertEqual(str(path.to_physical_path()), 
                         str(Path(self.temp_dir) / "A01" / "ch0" / "img1.tif"))
        
        # Test with a pattern that doesn't match
        path = self.storage_adapter.resolve_key("nonexistent/key")
        self.assertIsNone(path)

    def test_resolve_key_with_mapping(self):
        """Test resolving a key using a direct mapping."""
        # Register a direct mapping
        self.storage_adapter.register_mapping(
            "direct_key",
            os.path.join(self.temp_dir, "direct_file.tif")
        )
        
        # Test with a direct mapping
        path = self.storage_adapter.resolve_key("direct_key")
        self.assertIsNotNone(path)
        self.assertEqual(str(path.to_physical_path()), 
                         str(Path(self.temp_dir) / "direct_file.tif"))

    def test_file_manager_resolve_key(self):
        """Test resolving a key through the file manager."""
        # Test with a pattern that matches
        path = self.file_manager.resolve_key("A01/ch0/img1.tif")
        self.assertIsNotNone(path)
        self.assertEqual(str(path), 
                         str(Path(self.temp_dir) / "A01" / "ch0" / "img1.tif"))
        
        # Test with a pattern that doesn't match
        path = self.file_manager.resolve_key("nonexistent/key")
        self.assertIsNone(path)

    def test_file_manager_load_from_key(self):
        """Test loading an image using a logical key."""
        # Mock the load_image method
        self.file_manager.load_image = Mock(return_value=self.test_image)
        
        # Test loading with a key that resolves to a path
        image = self.file_manager.load_from_key("A01/ch0/img1.tif")
        self.assertIsNotNone(image)
        self.file_manager.load_image.assert_called_once()
        
        # Reset the mock
        self.file_manager.load_image.reset_mock()
        
        # Test loading with a key that doesn't resolve to a path
        # but can be read from the storage adapter
        self.storage_adapter.write("memory_key", self.test_image)
        image = self.file_manager.load_from_key("memory_key")
        self.assertIsNotNone(image)
        self.file_manager.load_image.assert_not_called()
        
        # Test loading with a key that doesn't resolve and can't be read
        image = self.file_manager.load_from_key("nonexistent/key")
        self.assertIsNone(image)

    def test_file_manager_write_to_key(self):
        """Test writing an image using a logical key."""
        # Mock the save_image method
        self.file_manager.save_image = Mock(return_value=True)
        
        # Test writing with a key that resolves to a path
        result = self.file_manager.write_to_key("A01/ch0/img2.tif", self.test_image)
        self.assertTrue(result)
        self.file_manager.save_image.assert_called_once()
        
        # Reset the mock
        self.file_manager.save_image.reset_mock()
        
        # Test writing with a key that doesn't resolve to a path
        # but can be written to the storage adapter
        result = self.file_manager.write_to_key("memory_key2", self.test_image)
        self.assertTrue(result)
        self.file_manager.save_image.assert_not_called()
        self.assertTrue(self.storage_adapter.exists("memory_key2"))
        
        # Test writing with a key that doesn't resolve and can't be written
        # (this is hard to test with MemoryStorageAdapter since it accepts any key)
        with patch.object(self.storage_adapter, 'write', side_effect=Exception("Test exception")):
            result = self.file_manager.write_to_key("error_key", self.test_image)
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
