"""
Unit tests for the adapter-aware FileManager.
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
from ezstitcher.io.virtual_path import VirtualPath, PhysicalPath, VirtualPathFactory


class TestAdapterAwareFileManager(unittest.TestCase):
    """Test the adapter-aware FileManager."""

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
        
        # Create some test directories
        os.makedirs(os.path.join(self.temp_dir, "test_dir"), exist_ok=True)
        
        # Create some test files
        with open(os.path.join(self.temp_dir, "test_dir", "test.tif"), "wb") as f:
            f.write(b"test image data")
            
        # Store some test data in the storage adapter
        self.storage_adapter.write("test_key", self.test_image)

    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_load_from_virtual_path(self):
        """Test loading an image from a virtual path."""
        # Create a virtual path
        virtual_path = VirtualPathFactory.from_path(os.path.join(self.temp_dir, "test_dir", "test.tif"))
        
        # Mock the read_bytes method to return test data
        virtual_path.read_bytes = Mock(return_value=b"test image data")
        
        # Mock tifffile.imread to return a test image
        with patch("tifffile.imread", return_value=self.test_image):
            # Load the image
            image = self.file_manager.load_from_virtual_path(virtual_path)
            
            # Check that the image was loaded
            self.assertIsNotNone(image)
            self.assertEqual(image.shape, self.test_image.shape)
            
            # Check that read_bytes was called
            virtual_path.read_bytes.assert_called_once()

    def test_load_from_key(self):
        """Test loading an image using a storage key."""
        # Load the image
        image = self.file_manager.load_from_key("test_key")
        
        # Check that the image was loaded
        self.assertIsNotNone(image)
        self.assertEqual(image.shape, self.test_image.shape)
        
        # Test with a nonexistent key
        image = self.file_manager.load_from_key("nonexistent_key")
        self.assertIsNone(image)
        
        # Test with no storage adapter
        self.file_manager.storage_adapter = None
        image = self.file_manager.load_from_key("test_key")
        self.assertIsNone(image)

    def test_save_to_virtual_path(self):
        """Test saving an image to a virtual path."""
        # Create a virtual path
        virtual_path = VirtualPathFactory.from_path(os.path.join(self.temp_dir, "test_dir", "new_test.tif"))
        
        # Mock the parent method
        parent = Mock()
        virtual_path.parent = Mock(return_value=parent)
        
        # Mock the write_bytes method
        virtual_path.write_bytes = Mock()
        
        # Mock tifffile.imwrite to return test data
        with patch("tifffile.imwrite", return_value=b"test image data"):
            # Save the image
            result = self.file_manager.save_to_virtual_path(self.test_image, virtual_path)
            
            # Check that the image was saved
            self.assertTrue(result)
            
            # Check that parent.mkdir was called
            parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            
            # Check that write_bytes was called
            virtual_path.write_bytes.assert_called_once()

    def test_save_to_key(self):
        """Test saving an image using a storage key."""
        # Save the image
        result = self.file_manager.save_to_key(self.test_image, "new_test_key")
        
        # Check that the image was saved
        self.assertTrue(result)
        self.assertTrue(self.storage_adapter.exists("new_test_key"))
        
        # Test with no storage adapter
        self.file_manager.storage_adapter = None
        result = self.file_manager.save_to_key(self.test_image, "another_test_key")
        self.assertFalse(result)

    def test_virtual_path_exists(self):
        """Test checking if a virtual path exists."""
        # Create a virtual path
        virtual_path = VirtualPathFactory.from_path(os.path.join(self.temp_dir, "test_dir", "test.tif"))
        
        # Mock the exists method
        virtual_path.exists = Mock(return_value=True)
        
        # Check if the path exists
        result = self.file_manager.virtual_path_exists(virtual_path)
        
        # Check the result
        self.assertTrue(result)
        
        # Check that exists was called
        virtual_path.exists.assert_called_once()

    def test_key_exists(self):
        """Test checking if a storage key exists."""
        # Check if the key exists
        result = self.file_manager.key_exists("test_key")
        
        # Check the result
        self.assertTrue(result)
        
        # Test with a nonexistent key
        result = self.file_manager.key_exists("nonexistent_key")
        self.assertFalse(result)
        
        # Test with no storage adapter
        self.file_manager.storage_adapter = None
        result = self.file_manager.key_exists("test_key")
        self.assertFalse(result)

    def test_remove_virtual_path(self):
        """Test removing a file at a virtual path."""
        # Create a virtual path
        virtual_path = VirtualPathFactory.from_path(os.path.join(self.temp_dir, "test_dir", "test.tif"))
        
        # Mock the unlink method
        virtual_path.unlink = Mock()
        
        # Remove the file
        result = self.file_manager.remove_virtual_path(virtual_path)
        
        # Check the result
        self.assertTrue(result)
        
        # Check that unlink was called
        virtual_path.unlink.assert_called_once_with(missing_ok=True)

    def test_remove_key(self):
        """Test removing a file using a storage key."""
        # Remove the file
        result = self.file_manager.remove_key("test_key")
        
        # Check the result
        self.assertTrue(result)
        self.assertFalse(self.storage_adapter.exists("test_key"))
        
        # Test with a nonexistent key
        result = self.file_manager.remove_key("nonexistent_key")
        self.assertTrue(result)  # Idempotent behavior
        
        # Test with no storage adapter
        self.file_manager.storage_adapter = None
        result = self.file_manager.remove_key("test_key")
        self.assertFalse(result)

    def test_ensure_virtual_directory(self):
        """Test ensuring a virtual directory exists."""
        # Create a virtual path
        virtual_directory = VirtualPathFactory.from_path(os.path.join(self.temp_dir, "new_dir"))
        
        # Mock the mkdir method
        virtual_directory.mkdir = Mock()
        
        # Ensure the directory exists
        result = self.file_manager.ensure_virtual_directory(virtual_directory)
        
        # Check the result
        self.assertEqual(result, virtual_directory)
        
        # Check that mkdir was called
        virtual_directory.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_list_virtual_image_files(self):
        """Test listing image files in a virtual directory."""
        # Create a virtual path
        virtual_directory = VirtualPathFactory.from_path(os.path.join(self.temp_dir, "test_dir"))
        
        # Mock the glob method
        virtual_directory.glob = Mock(return_value=[
            VirtualPathFactory.from_path(os.path.join(self.temp_dir, "test_dir", "test.tif"))
        ])
        
        # List the files
        files = self.file_manager.list_virtual_image_files(virtual_directory)
        
        # Check the result
        self.assertEqual(len(files), 1)
        
        # Check that glob was called
        virtual_directory.glob.assert_called_once()

    def test_list_keys(self):
        """Test listing keys in the storage adapter."""
        # Store another test key
        self.storage_adapter.write("another_test_key", self.test_image)
        
        # Mock the list_keys method
        self.storage_adapter.list_keys = Mock(return_value=["test_key", "another_test_key"])
        
        # List the keys
        keys = self.file_manager.list_keys()
        
        # Check the result
        self.assertEqual(len(keys), 2)
        self.assertIn("test_key", keys)
        self.assertIn("another_test_key", keys)
        
        # Check that list_keys was called
        self.storage_adapter.list_keys.assert_called_once_with("*")
        
        # Test with no storage adapter
        self.file_manager.storage_adapter = None
        keys = self.file_manager.list_keys()
        self.assertEqual(len(keys), 0)


if __name__ == "__main__":
    unittest.main()
