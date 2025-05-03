"""
Tests for the ZarrPath class.
"""

import io
import threading
import time
from pathlib import Path
import pytest
import numpy as np
import zarr

from ezstitcher.io.zarr_path import ZarrPath


def test_create_memory_zarr():
    """Test creating a memory-based Zarr store."""
    # Create a memory-based Zarr store
    root, store = ZarrPath.create_memory_zarr()
    assert isinstance(root, ZarrPath)
    assert isinstance(store, zarr.MemoryStore)
    assert str(root) == "/"


def test_create_file_zarr(tmp_path):
    """Test creating a file-based Zarr store."""
    # Create a file-based Zarr store
    zarr_path = tmp_path / "test.zarr"
    root, store = ZarrPath.create_file_zarr(zarr_path)
    assert isinstance(root, ZarrPath)
    assert isinstance(store, zarr.DirectoryStore)
    assert str(root) == "/"
    assert zarr_path.exists()


def test_write_read_bytes():
    """Test writing and reading bytes."""
    # Create a memory-based Zarr store
    root, store = ZarrPath.create_memory_zarr()

    # Create a file
    file_path = root / "test.bin"
    data = b"Hello, world!"
    file_path.write_bytes(data)

    # Read the file
    assert file_path.read_bytes() == data


def test_write_read_text():
    """Test writing and reading text."""
    # Create a memory-based Zarr store
    root, store = ZarrPath.create_memory_zarr()

    # Create a file
    file_path = root / "test.txt"
    data = "Hello, world!"
    file_path.write_text(data)

    # Read the file
    assert file_path.read_text() == data


def test_mkdir():
    """Test creating directories."""
    # Create a memory-based Zarr store
    root, store = ZarrPath.create_memory_zarr()

    # Create a directory
    dir_path = root / "test"
    dir_path.mkdir()
    assert dir_path.exists()
    assert dir_path.is_dir()

    # Create a nested directory
    nested_path = root / "test" / "nested"
    nested_path.mkdir(parents=True)
    assert nested_path.exists()
    assert nested_path.is_dir()


def test_iterdir():
    """Test iterating over directories."""
    # Create a memory-based Zarr store
    root, store = ZarrPath.create_memory_zarr()

    # Create some files and directories
    (root / "file1.txt").write_text("File 1")
    (root / "file2.txt").write_text("File 2")
    (root / "dir1").mkdir()
    (root / "dir2").mkdir()

    # Iterate over the root directory
    items = list(root.iterdir())
    assert len(items) == 4
    names = [item.name() for item in items]
    assert "file1.txt" in names
    assert "file2.txt" in names
    assert "dir1" in names
    assert "dir2" in names


def test_glob():
    """Test globbing."""
    # Create a memory-based Zarr store
    root, store = ZarrPath.create_memory_zarr()

    # Create some files and directories
    (root / "file1.txt").write_text("File 1")
    (root / "file2.txt").write_text("File 2")
    (root / "file3.bin").write_bytes(b"Binary data")
    (root / "dir1").mkdir()
    (root / "dir2").mkdir()

    # Glob for text files
    items = list(root.glob("*.txt"))
    assert len(items) == 2
    names = [item.name() for item in items]
    assert "file1.txt" in names
    assert "file2.txt" in names

    # Glob for directories
    items = list(root.glob("dir*"))
    assert len(items) == 2
    names = [item.name() for item in items]
    assert "dir1" in names
    assert "dir2" in names


def test_large_array_streaming():
    """Test streaming large arrays."""
    # Create a memory-based Zarr store
    root, store = ZarrPath.create_memory_zarr()

    # Create a large array (>10MB)
    file_path = root / "large.bin"
    data = b"x" * (12 * 1024 * 1024)  # 12MB
    file_path.write_bytes(data)

    # Read the array
    assert file_path.read_bytes() == data


@pytest.mark.skip(reason="Await full plan block approval")
def test_overwrite_protection():
    """Test overwrite protection."""
    # Create a memory-based Zarr store
    root, store = ZarrPath.create_memory_zarr()

    # Create a file
    file_path = root / "test.bin"
    file_path.write_bytes(b"first")

    # Try to overwrite without the overwrite flag
    with pytest.raises(FileExistsError):
        file_path.write_bytes(b"second")

    # Overwrite with the overwrite flag
    file_path.write_bytes(b"second", overwrite=True)
    assert file_path.read_bytes() == b"second"


def test_thread_safety():
    """Test thread safety of the cache."""
    # Create a memory-based Zarr store
    root, store = ZarrPath.create_memory_zarr()

    # Create some files
    for i in range(10):
        (root / f"file{i}.txt").write_text(f"File {i}")

    # Define a function to read and write files
    def read_write(index, delay=0):
        time.sleep(delay)  # Add a delay to increase chance of race condition
        file_path = root / f"file{index}.txt"
        content = file_path.read_text()
        file_path.write_text(content + " (updated)", overwrite=True)
        return file_path.read_text()

    # Create multiple threads that read and write files
    results = [None] * 10
    threads = []

    for i in range(10):
        def thread_func(idx=i):
            results[idx] = read_write(idx, 0.01 * idx)

        thread = threading.Thread(target=thread_func)
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Check that all files were updated correctly
    for i in range(10):
        assert results[i] == f"File {i} (updated)"
        assert (root / f"file{i}.txt").read_text() == f"File {i} (updated)"


if __name__ == '__main__':
    unittest.main()
