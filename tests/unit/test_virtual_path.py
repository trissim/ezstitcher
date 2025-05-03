"""
Unit tests for the VirtualPath interface and implementations.
"""

import unittest
from pathlib import Path
import tempfile
import os
import shutil

from ezstitcher.io.virtual_path import VirtualPath, PhysicalPath, open_virtual
from ezstitcher.io.virtual_path_factory import VirtualPathFactory
from ezstitcher.io.virtual_path import VirtualPathResolver


class TestPhysicalPath(unittest.TestCase):
    """Test the PhysicalPath implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("test content")
        self.test_dir = os.path.join(self.temp_dir, "test_dir")
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_str(self):
        """Test the __str__ method."""
        path = PhysicalPath(self.test_file)
        self.assertEqual(str(path), str(Path(self.test_file)))

    def test_truediv(self):
        """Test the __truediv__ method."""
        path = PhysicalPath(self.temp_dir)
        result = path / "test.txt"
        self.assertEqual(str(result), str(Path(self.test_file)))

    def test_name(self):
        """Test the name method."""
        path = PhysicalPath(self.test_file)
        self.assertEqual(path.name(), "test.txt")

    def test_stem(self):
        """Test the stem method."""
        path = PhysicalPath(self.test_file)
        self.assertEqual(path.stem(), "test")

    def test_suffix(self):
        """Test the suffix method."""
        path = PhysicalPath(self.test_file)
        self.assertEqual(path.suffix(), ".txt")

    def test_parent(self):
        """Test the parent method."""
        path = PhysicalPath(self.test_file)
        self.assertEqual(str(path.parent()), str(Path(self.temp_dir)))

    def test_exists(self):
        """Test the exists method."""
        path = PhysicalPath(self.test_file)
        self.assertTrue(path.exists())
        path = PhysicalPath(os.path.join(self.temp_dir, "nonexistent.txt"))
        self.assertFalse(path.exists())

    def test_is_file(self):
        """Test the is_file method."""
        path = PhysicalPath(self.test_file)
        self.assertTrue(path.is_file())
        path = PhysicalPath(self.test_dir)
        self.assertFalse(path.is_file())

    def test_is_dir(self):
        """Test the is_dir method."""
        path = PhysicalPath(self.test_dir)
        self.assertTrue(path.is_dir())
        path = PhysicalPath(self.test_file)
        self.assertFalse(path.is_dir())

    def test_glob(self):
        """Test the glob method."""
        path = PhysicalPath(self.temp_dir)
        results = list(path.glob("*.txt"))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name(), "test.txt")

    def test_relative_to(self):
        """Test the relative_to method."""
        path = PhysicalPath(self.test_file)
        rel_path = path.relative_to(self.temp_dir)
        self.assertEqual(str(rel_path), "test.txt")

    def test_is_relative_to(self):
        """Test the is_relative_to method."""
        path = PhysicalPath(self.test_file)
        self.assertTrue(path.is_relative_to(self.temp_dir))
        self.assertFalse(path.is_relative_to("/nonexistent"))

    def test_resolve(self):
        """Test the resolve method."""
        path = PhysicalPath(self.test_file)
        resolved = path.resolve()
        self.assertEqual(str(resolved), str(Path(self.test_file).resolve()))

    def test_open(self):
        """Test the open method."""
        path = PhysicalPath(self.test_file)
        with path.open("r") as f:
            content = f.read()
        self.assertEqual(content, "test content")

    def test_read_bytes(self):
        """Test the read_bytes method."""
        path = PhysicalPath(self.test_file)
        content = path.read_bytes()
        self.assertEqual(content, b"test content")

    def test_read_text(self):
        """Test the read_text method."""
        path = PhysicalPath(self.test_file)
        content = path.read_text()
        self.assertEqual(content, "test content")

    def test_write_bytes(self):
        """Test the write_bytes method."""
        path = PhysicalPath(os.path.join(self.temp_dir, "write_bytes.txt"))
        path.write_bytes(b"test bytes")
        with open(os.path.join(self.temp_dir, "write_bytes.txt"), "rb") as f:
            content = f.read()
        self.assertEqual(content, b"test bytes")

    def test_write_text(self):
        """Test the write_text method."""
        path = PhysicalPath(os.path.join(self.temp_dir, "write_text.txt"))
        path.write_text("test text")
        with open(os.path.join(self.temp_dir, "write_text.txt"), "r") as f:
            content = f.read()
        self.assertEqual(content, "test text")

    def test_mkdir(self):
        """Test the mkdir method."""
        path = PhysicalPath(os.path.join(self.temp_dir, "new_dir"))
        path.mkdir()
        self.assertTrue(os.path.isdir(os.path.join(self.temp_dir, "new_dir")))

    def test_rmdir(self):
        """Test the rmdir method."""
        path = PhysicalPath(os.path.join(self.temp_dir, "rmdir_test"))
        os.makedirs(os.path.join(self.temp_dir, "rmdir_test"))
        path.rmdir()
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, "rmdir_test")))

    def test_unlink(self):
        """Test the unlink method."""
        path = PhysicalPath(os.path.join(self.temp_dir, "unlink_test.txt"))
        with open(os.path.join(self.temp_dir, "unlink_test.txt"), "w") as f:
            f.write("test")
        path.unlink()
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, "unlink_test.txt")))

    def test_iterdir(self):
        """Test the iterdir method."""
        # Create some files in the test directory
        for i in range(3):
            with open(os.path.join(self.test_dir, f"file{i}.txt"), "w") as f:
                f.write(f"content {i}")
        
        path = PhysicalPath(self.test_dir)
        results = list(path.iterdir())
        self.assertEqual(len(results), 3)
        names = sorted([p.name() for p in results])
        self.assertEqual(names, ["file0.txt", "file1.txt", "file2.txt"])

    def test_to_physical_path(self):
        """Test the to_physical_path method."""
        path = PhysicalPath(self.test_file)
        physical_path = path.to_physical_path()
        self.assertEqual(str(physical_path), str(Path(self.test_file)))

    def test_to_storage_key(self):
        """Test the to_storage_key method."""
        path = PhysicalPath(self.test_file)
        key = path.to_storage_key()
        self.assertEqual(key, str(Path(self.test_file)).replace("\\", "/"))

    def test_from_storage_key(self):
        """Test the from_storage_key method."""
        key = str(Path(self.test_file)).replace("\\", "/")
        path = PhysicalPath.from_storage_key(key)
        self.assertEqual(str(path), str(Path(self.test_file)))


class TestVirtualPathFactory(unittest.TestCase):
    """Test the VirtualPathFactory class."""

    def test_from_path(self):
        """Test the from_path method."""
        path = VirtualPathFactory.from_path("/test/path")
        self.assertIsInstance(path, PhysicalPath)
        self.assertEqual(str(path), "/test/path")

    def test_from_storage_key(self):
        """Test the from_storage_key method."""
        path = VirtualPathFactory.from_storage_key("test/key")
        self.assertIsInstance(path, PhysicalPath)
        self.assertEqual(str(path), "test/key")

    def test_from_uri(self):
        """Test the from_uri method."""
        path = VirtualPathFactory.from_uri("file:///test/uri")
        self.assertIsInstance(path, PhysicalPath)
        self.assertEqual(str(path), "file:///test/uri")


class TestVirtualPathResolver(unittest.TestCase):
    """Test the VirtualPathResolver class."""

    def setUp(self):
        """Set up test fixtures."""
        self.resolver = VirtualPathResolver()

    def test_resolve_virtual_path(self):
        """Test resolving a VirtualPath."""
        path = PhysicalPath("/test/path")
        resolved = self.resolver.resolve(path)
        self.assertIs(resolved, path)

    def test_resolve_string(self):
        """Test resolving a string."""
        resolved = self.resolver.resolve("/test/path")
        self.assertIsInstance(resolved, PhysicalPath)
        self.assertEqual(str(resolved), "/test/path")

    def test_resolve_path(self):
        """Test resolving a Path."""
        resolved = self.resolver.resolve(Path("/test/path"))
        self.assertIsInstance(resolved, PhysicalPath)
        self.assertEqual(str(resolved), str(Path("/test/path")))

    def test_to_physical_path(self):
        """Test converting to a physical path."""
        physical = self.resolver.to_physical_path("/test/path")
        self.assertEqual(str(physical), str(Path("/test/path")))

    def test_to_storage_key(self):
        """Test converting to a storage key."""
        key = self.resolver.to_storage_key("/test/path")
        self.assertEqual(key, str(Path("/test/path")).replace("\\", "/"))


class TestOpenVirtual(unittest.TestCase):
    """Test the open_virtual function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("test content")

    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_open_virtual_with_virtual_path(self):
        """Test opening a VirtualPath."""
        path = PhysicalPath(self.test_file)
        with open_virtual(path) as f:
            content = f.read()
        self.assertEqual(content, "test content")

    def test_open_virtual_with_string(self):
        """Test opening a string path."""
        with open_virtual(self.test_file) as f:
            content = f.read()
        self.assertEqual(content, "test content")

    def test_open_virtual_with_path(self):
        """Test opening a Path."""
        with open_virtual(Path(self.test_file)) as f:
            content = f.read()
        self.assertEqual(content, "test content")


if __name__ == "__main__":
    unittest.main()
