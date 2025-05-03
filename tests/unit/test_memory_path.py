"""
Unit tests for the MemoryPath implementation.
"""

import unittest
import pytest
import threading
import io
from pathlib import Path, PurePosixPath

from ezstitcher.io.memory_path import MemoryPath
from ezstitcher.io.virtual_path import VirtualPath
from ezstitcher.io.virtual_path_factory import VirtualPathFactory


class TestMemoryPath(unittest.TestCase):
    """Test the MemoryPath class."""

    def setUp(self):
        """Set up the test environment."""
        # Reset the in-memory storage before each test
        MemoryPath.reset_storage()
        
        # Create some test files and directories
        self.root = MemoryPath('/')
        self.test_dir = MemoryPath('/test_dir')
        self.test_dir.mkdir()
        self.test_file = MemoryPath('/test_dir/test_file.txt')
        self.test_file.write_text('Hello, world!')
        self.nested_dir = MemoryPath('/test_dir/nested_dir')
        self.nested_dir.mkdir()
        self.nested_file = MemoryPath('/test_dir/nested_dir/nested_file.txt')
        self.nested_file.write_text('Nested file content')

    def test_init(self):
        """Test the initialization of a MemoryPath."""
        # Test with string path
        path = MemoryPath('/foo/bar')
        self.assertEqual(str(path), '/foo/bar')
        
        # Test with PurePosixPath
        path = MemoryPath(PurePosixPath('/foo/bar'))
        self.assertEqual(str(path), '/foo/bar')
        
        # Test path normalization
        path = MemoryPath('/foo/bar/')
        self.assertEqual(str(path), '/foo/bar')
        
        # Test backslash conversion
        path = MemoryPath('\\foo\\bar')
        self.assertEqual(str(path), '/foo/bar')

    def test_str(self):
        """Test the string representation of a MemoryPath."""
        path = MemoryPath('/foo/bar')
        self.assertEqual(str(path), '/foo/bar')

    def test_truediv(self):
        """Test the division operator for path joining."""
        # Test joining with string
        path = MemoryPath('/foo')
        joined = path / 'bar'
        self.assertEqual(str(joined), '/foo/bar')
        self.assertIsInstance(joined, MemoryPath)
        
        # Test joining with another MemoryPath
        other = MemoryPath('baz')
        joined = path / other
        self.assertEqual(str(joined), '/foo/baz')
        self.assertIsInstance(joined, MemoryPath)

    def test_name(self):
        """Test the name method."""
        path = MemoryPath('/foo/bar.txt')
        self.assertEqual(path.name(), 'bar.txt')

    def test_stem(self):
        """Test the stem method."""
        path = MemoryPath('/foo/bar.txt')
        self.assertEqual(path.stem(), 'bar')

    def test_suffix(self):
        """Test the suffix method."""
        path = MemoryPath('/foo/bar.txt')
        self.assertEqual(path.suffix(), '.txt')

    def test_parent(self):
        """Test the parent method."""
        path = MemoryPath('/foo/bar')
        parent = path.parent()
        self.assertEqual(str(parent), '/foo')
        self.assertIsInstance(parent, MemoryPath)

    def test_exists(self):
        """Test the exists method."""
        # Test existing file
        self.assertTrue(self.test_file.exists())
        
        # Test existing directory
        self.assertTrue(self.test_dir.exists())
        
        # Test non-existent path
        path = MemoryPath('/non_existent')
        self.assertFalse(path.exists())

    def test_is_file(self):
        """Test the is_file method."""
        # Test file
        self.assertTrue(self.test_file.is_file())
        
        # Test directory
        self.assertFalse(self.test_dir.is_file())
        
        # Test non-existent path
        path = MemoryPath('/non_existent')
        self.assertFalse(path.is_file())

    def test_is_dir(self):
        """Test the is_dir method."""
        # Test directory
        self.assertTrue(self.test_dir.is_dir())
        
        # Test file
        self.assertFalse(self.test_file.is_dir())
        
        # Test non-existent path
        path = MemoryPath('/non_existent')
        self.assertFalse(path.is_dir())

    def test_glob(self):
        """Test the glob method."""
        # Create some files for globbing
        MemoryPath('/test_dir/file1.txt').write_text('File 1')
        MemoryPath('/test_dir/file2.txt').write_text('File 2')
        MemoryPath('/test_dir/file3.dat').write_text('File 3')
        
        # Test globbing with * pattern
        paths = list(self.test_dir.glob('*.txt'))
        self.assertEqual(len(paths), 3)  # test_file.txt, file1.txt, file2.txt
        self.assertTrue(all(isinstance(p, MemoryPath) for p in paths))
        
        # Test globbing with ? pattern
        paths = list(self.test_dir.glob('file?.txt'))
        self.assertEqual(len(paths), 2)  # file1.txt, file2.txt
        
        # Test globbing with specific pattern
        paths = list(self.test_dir.glob('file1.txt'))
        self.assertEqual(len(paths), 1)
        self.assertEqual(str(paths[0]), '/test_dir/file1.txt')

    def test_relative_to(self):
        """Test the relative_to method."""
        # Test relative to parent
        rel = self.test_file.relative_to(self.test_dir)
        self.assertEqual(str(rel), 'test_file.txt')
        
        # Test relative to root
        rel = self.test_file.relative_to('/')
        self.assertEqual(str(rel), 'test_dir/test_file.txt')
        
        # Test relative to self
        rel = self.test_file.relative_to(self.test_file)
        self.assertEqual(str(rel), '.')
        
        # Test error when not relative
        with self.assertRaises(ValueError):
            self.test_dir.relative_to(self.test_file)
            
        # Test error when different backend
        with self.assertRaises(ValueError):
            self.test_file.relative_to(Path('/test_dir'))

    def test_is_relative_to(self):
        """Test the is_relative_to method."""
        # Test relative to parent
        self.assertTrue(self.test_file.is_relative_to(self.test_dir))
        
        # Test relative to root
        self.assertTrue(self.test_file.is_relative_to('/'))
        
        # Test relative to self
        self.assertTrue(self.test_file.is_relative_to(self.test_file))
        
        # Test not relative
        self.assertFalse(self.test_dir.is_relative_to(self.test_file))
        
        # Test different backend
        self.assertFalse(self.test_file.is_relative_to(Path('/test_dir')))

    def test_resolve(self):
        """Test the resolve method."""
        # Memory paths are already absolute, so resolve should return a copy
        path = MemoryPath('/foo/bar')
        resolved = path.resolve()
        self.assertEqual(str(resolved), '/foo/bar')
        self.assertIsInstance(resolved, MemoryPath)
        self.assertIsNot(resolved, path)  # Should be a new object

    def test_iterdir(self):
        """Test the iterdir method."""
        # Test iterating over a directory
        paths = list(self.test_dir.iterdir())
        self.assertEqual(len(paths), 3)  # test_file.txt, nested_dir, and one more from glob test
        self.assertTrue(all(isinstance(p, MemoryPath) for p in paths))
        
        # Test error when not a directory
        with self.assertRaises(NotADirectoryError):
            list(self.test_file.iterdir())
            
        # Test error when path doesn't exist
        with self.assertRaises(NotADirectoryError):
            list(MemoryPath('/non_existent').iterdir())

    def test_open(self):
        """Test the open method."""
        # Test opening for reading in text mode
        with self.test_file.open('r') as f:
            content = f.read()
            self.assertEqual(content, 'Hello, world!')
            
        # Test opening for reading in binary mode
        with self.test_file.open('rb') as f:
            content = f.read()
            self.assertEqual(content, b'Hello, world!')
            
        # Test opening for writing in text mode
        with self.test_file.open('w') as f:
            f.write('New content')
        self.assertEqual(self.test_file.read_text(), 'New content')
        
        # Test opening for writing in binary mode
        with self.test_file.open('wb') as f:
            f.write(b'Binary content')
        self.assertEqual(self.test_file.read_bytes(), b'Binary content')
        
        # Test opening for appending in text mode
        with self.test_file.open('a') as f:
            f.write(' appended')
        self.assertEqual(self.test_file.read_text(), 'Binary content appended')
        
        # Test error when file doesn't exist
        with self.assertRaises(FileNotFoundError):
            MemoryPath('/non_existent').open('r')
            
        # Test error when opening a directory
        with self.assertRaises(IsADirectoryError):
            self.test_dir.open('r')

    def test_read_bytes(self):
        """Test the read_bytes method."""
        # Test reading bytes from a file
        content = self.test_file.read_bytes()
        self.assertEqual(content, b'Hello, world!')
        
        # Test error when file doesn't exist
        with self.assertRaises(FileNotFoundError):
            MemoryPath('/non_existent').read_bytes()
            
        # Test error when reading from a directory
        with self.assertRaises(IsADirectoryError):
            self.test_dir.read_bytes()

    def test_read_text(self):
        """Test the read_text method."""
        # Test reading text from a file
        content = self.test_file.read_text()
        self.assertEqual(content, 'Hello, world!')
        
        # Test reading with specific encoding
        self.test_file.write_bytes('Привет'.encode('utf-8'))
        content = self.test_file.read_text(encoding='utf-8')
        self.assertEqual(content, 'Привет')
        
        # Test error when file doesn't exist
        with self.assertRaises(FileNotFoundError):
            MemoryPath('/non_existent').read_text()
            
        # Test error when reading from a directory
        with self.assertRaises(IsADirectoryError):
            self.test_dir.read_text()

    def test_write_bytes(self):
        """Test the write_bytes method."""
        # Test writing bytes to a file
        path = MemoryPath('/new_file')
        path.write_bytes(b'Binary data')
        self.assertEqual(path.read_bytes(), b'Binary data')
        
        # Test overwriting existing file
        path.write_bytes(b'New data')
        self.assertEqual(path.read_bytes(), b'New data')
        
        # Test writing to a file in a non-existent directory
        path = MemoryPath('/new_dir/new_file')
        path.write_bytes(b'Data in new dir')
        self.assertEqual(path.read_bytes(), b'Data in new dir')
        self.assertTrue(MemoryPath('/new_dir').is_dir())

    def test_write_text(self):
        """Test the write_text method."""
        # Test writing text to a file
        path = MemoryPath('/new_text_file')
        path.write_text('Text data')
        self.assertEqual(path.read_text(), 'Text data')
        
        # Test writing with specific encoding
        path.write_text('Привет', encoding='utf-8')
        self.assertEqual(path.read_text(encoding='utf-8'), 'Привет')
        
        # Test overwriting existing file
        path.write_text('New text')
        self.assertEqual(path.read_text(), 'New text')

    def test_mkdir(self):
        """Test the mkdir method."""
        # Test creating a directory
        path = MemoryPath('/new_dir')
        path.mkdir()
        self.assertTrue(path.is_dir())
        
        # Test error when directory already exists
        with self.assertRaises(FileExistsError):
            path.mkdir()
            
        # Test creating with exist_ok=True
        path.mkdir(exist_ok=True)  # Should not raise
        
        # Test error when parent doesn't exist
        path = MemoryPath('/non_existent/new_dir')
        with self.assertRaises(FileNotFoundError):
            path.mkdir()
            
        # Test creating with parents=True
        path.mkdir(parents=True)
        self.assertTrue(path.is_dir())
        self.assertTrue(MemoryPath('/non_existent').is_dir())
        
        # Test error when path exists as a file
        self.test_file.parent().mkdir(exist_ok=True)  # Ensure parent exists
        with self.assertRaises(FileExistsError):
            MemoryPath(str(self.test_file)).mkdir()

    def test_rmdir(self):
        """Test the rmdir method."""
        # Test removing a directory
        path = MemoryPath('/empty_dir')
        path.mkdir()
        path.rmdir()
        self.assertFalse(path.exists())
        
        # Test error when directory doesn't exist
        with self.assertRaises(FileNotFoundError):
            MemoryPath('/non_existent').rmdir()
            
        # Test error when path is a file
        with self.assertRaises(NotADirectoryError):
            self.test_file.rmdir()
            
        # Test error when directory is not empty
        with self.assertRaises(OSError):
            self.test_dir.rmdir()

    def test_unlink(self):
        """Test the unlink method."""
        # Test removing a file
        path = MemoryPath('/test_file_to_remove')
        path.write_text('To be removed')
        path.unlink()
        self.assertFalse(path.exists())
        
        # Test error when file doesn't exist
        with self.assertRaises(FileNotFoundError):
            MemoryPath('/non_existent').unlink()
            
        # Test with missing_ok=True
        MemoryPath('/non_existent').unlink(missing_ok=True)  # Should not raise
        
        # Test error when path is a directory
        with self.assertRaises(IsADirectoryError):
            self.test_dir.unlink()

    def test_to_physical_path(self):
        """Test the to_physical_path method."""
        # Memory paths don't correspond to physical paths
        self.assertIsNone(self.test_file.to_physical_path())

    def test_to_storage_key(self):
        """Test the to_storage_key method."""
        # Test converting to storage key
        key = self.test_file.to_storage_key()
        self.assertEqual(key, 'memory:/test_dir/test_file.txt')

    def test_from_storage_key(self):
        """Test the from_storage_key method."""
        # Test creating from storage key with memory: prefix
        path = MemoryPath.from_storage_key('memory:/foo/bar')
        self.assertEqual(str(path), '/foo/bar')
        
        # Test creating from storage key without prefix
        path = MemoryPath.from_storage_key('/foo/bar')
        self.assertEqual(str(path), '/foo/bar')

    def test_reset_storage(self):
        """Test the reset_storage method."""
        # Create some files and directories
        MemoryPath('/test_reset').mkdir()
        MemoryPath('/test_reset/file.txt').write_text('Test')
        
        # Reset storage
        MemoryPath.reset_storage()
        
        # Check that storage is empty except for root
        self.assertTrue(MemoryPath('/').exists())
        self.assertFalse(MemoryPath('/test_reset').exists())
        self.assertFalse(MemoryPath('/test_reset/file.txt').exists())
        
        # Check that test_dir and test_file from setUp no longer exist
        self.assertFalse(self.test_dir.exists())
        self.assertFalse(self.test_file.exists())

    def test_get_storage_snapshot(self):
        """Test the get_storage_snapshot method."""
        # Get snapshot
        snapshot = MemoryPath.get_storage_snapshot()
        
        # Check that it's a copy, not the original
        self.assertIsNot(snapshot, MemoryPath._storage)
        
        # Check that it contains the expected entries
        self.assertIn('/', snapshot)
        self.assertIn('/test_dir', snapshot)
        self.assertIn('/test_dir/test_file.txt', snapshot)
        
        # Check that modifying the snapshot doesn't affect the original
        snapshot['/new_path'] = {'type': 'file', 'content': b'New'}
        self.assertNotIn('/new_path', MemoryPath._storage)

    def test_virtual_path_factory_integration(self):
        """Test integration with VirtualPathFactory."""
        # Test creating from storage key
        path = VirtualPathFactory.from_storage_key('memory:/foo/bar')
        self.assertIsInstance(path, MemoryPath)
        self.assertEqual(str(path), '/foo/bar')
        
        # Test creating from URI
        path = VirtualPathFactory.from_uri('memory:/foo/bar')
        self.assertIsInstance(path, MemoryPath)
        self.assertEqual(str(path), '/foo/bar')

    # --- Thread Safety Tests (Skipped) ---

    @pytest.mark.skip(reason="Thread safety tests for MemoryPath - skipped per plan")
    def test_thread_safety_concurrent_writes(self):
        """Test concurrent writes to the same MemoryPath file."""
        # Note: Imports moved to top level
        num_threads = 5
        write_count = 10
        path = MemoryPath('/concurrent_write_test.txt')
        results = [0] * num_threads # Simple way to track writes per thread

        def writer_task(thread_id):
            for i in range(write_count):
                try:
                    # Append unique data from each thread
                    data_to_write = f"Thread {thread_id}, Write {i}\n".encode('utf-8')
                    # Use open in append mode for simplicity, relies on internal locking
                    with path.open('ab') as f:
                         f.write(data_to_write)
                    results[thread_id] += 1
                except Exception as e:
                    print(f"Error in thread {thread_id}: {e}") # Basic error logging for debug

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=writer_task, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all writes occurred
        self.assertEqual(sum(results), num_threads * write_count, "Not all writes were successful")

        # Verify final content (order isn't guaranteed, but all lines should be present)
        final_content = path.read_text()
        lines = final_content.strip().split('\n')
        self.assertEqual(len(lines), num_threads * write_count, "Incorrect number of lines in final file")
        # Check if all expected lines are present (regardless of order)
        expected_lines = {f"Thread {i}, Write {j}" for i in range(num_threads) for j in range(write_count)}
        actual_lines = set(lines)
        self.assertSetEqual(actual_lines, expected_lines, "Final content mismatch")


    @pytest.mark.skip(reason="Thread safety tests for MemoryPath - skipped per plan")
    def test_thread_safety_concurrent_mkdir(self):
        """Test concurrent mkdir calls for the same and parent directories."""
        # Note: Imports moved to top level
        num_threads = 10
        path_str = '/concurrent_dir/subdir/nested'
        path = MemoryPath(path_str)
        parent_path = path.parent()
        results = {'success': 0, 'exists': 0, 'error': 0}
        lock = threading.Lock() # Lock for updating shared results dict

        def mkdir_task():
            try:
                # Attempt to create the directory, potentially with parents
                path.mkdir(parents=True, exist_ok=True)
                with lock:
                    results['success'] += 1
            except FileExistsError:
                 # This might happen if exist_ok=False, but we use True
                 with lock:
                     results['exists'] += 1
            except Exception:
                 with lock:
                     results['error'] += 1


        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=mkdir_task)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify the directory and its parents exist
        self.assertTrue(path.is_dir(), f"Final directory {path_str} should exist")
        self.assertTrue(parent_path.is_dir(), f"Parent directory {parent_path} should exist")
        self.assertTrue(parent_path.parent().is_dir(), "Grandparent directory should exist")

        # Verify no errors occurred (exist_ok=True handles the race condition)
        self.assertEqual(results['error'], 0, "Errors occurred during concurrent mkdir")
        # Verify that threads either succeeded or found it already existed
        self.assertEqual(results['success'] + results['exists'], num_threads, "Mismatch in mkdir results count")


if __name__ == '__main__':
    unittest.main()
