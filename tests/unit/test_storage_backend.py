# tests/unit/test_storage_backend.py

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil # For setting up disk tests if needed

# Import interfaces, implementations, types, constants
from ezstitcher.io.storage_backend import BasicStorageBackend, MicroscopyStorageBackend, DiskStorageBackend, FakeStorageBackend
from ezstitcher.io.types import ImageArray
from ezstitcher.io.constants import DEFAULT_IMAGE_EXTENSIONS
 # --- Fixtures ---
@pytest.fixture
def fake_backend() -> FakeStorageBackend:
    """Provides a clean FakeStorageBackend for each test."""
    return FakeStorageBackend()

@pytest.fixture
def disk_backend() -> DiskStorageBackend:
    """Provides a DiskStorageBackend instance."""
    # Note: Tests using this might need tmp_path fixture as well
    return DiskStorageBackend()

# --- Tests for FakeStorageBackend ---
class TestFakeStorageBackend:
    # Example: Test basic save/load/exists cycle
    def test_save_load_exists_cycle(self, fake_backend):
        img_data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        file_path = Path("test_dir/image.tif")

        assert not fake_backend.exists(file_path)
        assert not fake_backend.exists(file_path.parent)

        # Save the image
        assert fake_backend.save_image(img_data, file_path)

        # Check existence
        assert fake_backend.exists(file_path)
        assert fake_backend.exists(file_path.parent) # Directory should now exist

        # Load and verify
        loaded_img = fake_backend.load_image(file_path)
        assert loaded_img is not None
        np.testing.assert_array_equal(loaded_img, img_data)

    # Example: Test listing with patterns and recursion
    def test_list_files_complex(self, fake_backend):
        # Setup files
        fake_backend.save("content1", "root/file1.txt")
        fake_backend.save("content2", "root/file2.log")
        fake_backend.save("content3", "root/subdir/file3.txt")
        fake_backend.save("content4", "root/subdir/another.log")

        # List non-recursive, no pattern
        assert set(fake_backend.list_files("root")) == {Path("root/file1.txt"), Path("root/file2.log")}

        # List recursive, no pattern
        assert set(fake_backend.list_files("root", recursive=True)) == {
            Path("root/file1.txt"), Path("root/file2.log"),
            Path("root/subdir/file3.txt"), Path("root/subdir/another.log")
        }

        # List recursive, txt pattern
        assert set(fake_backend.list_files("root", pattern="*.txt", recursive=True)) == {
            Path("root/file1.txt"), Path("root/subdir/file3.txt")
        }

        # List non-recursive, log pattern
        assert set(fake_backend.list_files("root", pattern="*.log")) == {Path("root/file2.log")}

        # List from non-existent dir
        assert fake_backend.list_files("nonexistent") == []

    # Example: Test deletion
    def test_delete(self, fake_backend):
        file_path = Path("to_delete.dat")
        fake_backend.save("data", file_path)
        assert fake_backend.exists(file_path)
        assert fake_backend.delete_file(file_path)
        assert not fake_backend.exists(file_path)
        # Deleting non-existent file
        assert not fake_backend.delete_file(Path("nonexistent.dat"))

    # ... more tests for copy, ensure_directory, find_file_recursive etc. ...
    # ... tests for microscopy specific methods (find_image_dir, list_image_files etc.) ...


# --- Tests for DiskStorageBackend ---
class TestDiskStorageBackend:

    # Test NATIVELY implemented methods (require tmp_path fixture)
    def test_ensure_directory_native(self, disk_backend, tmp_path):
        new_dir = tmp_path / "new" / "subdir"
        assert not new_dir.exists() # Precondition
        created_path = disk_backend.ensure_directory(new_dir)
        assert created_path == new_dir # Should return the path
        assert new_dir.is_dir() # Postcondition: Directory exists

        # Test idempotency
        created_path_again = disk_backend.ensure_directory(new_dir)
        assert created_path_again == new_dir
        assert new_dir.is_dir()

    def test_copy_delete_native(self, disk_backend, tmp_path):
        src_dir = tmp_path / "src"
        dest_dir = tmp_path / "dest"
        src_file = src_dir / "source.txt"
        dest_file = dest_dir / "copied.txt"

        # Setup source file
        src_dir.mkdir()
        src_file.write_text("content")
        assert src_file.exists()
        assert not dest_dir.exists() # Precondition for copy creating dir

        # Test copy (should create dest_dir)
        assert disk_backend.copy_file(src_file, dest_file)
        assert dest_file.exists()
        assert dest_file.read_text() == "content"
        assert dest_dir.is_dir()

        # Test delete
        assert disk_backend.delete_file(dest_file)
        assert not dest_file.exists()

        # Test delete non-existent
        assert not disk_backend.delete_file(dest_file) # Should return False

    def test_list_files_native(self, disk_backend, tmp_path):
        # Setup
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.log").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").touch()

        # Test non-recursive, no pattern
        assert set(disk_backend.list_files(tmp_path)) == {tmp_path / "file1.txt", tmp_path / "file2.log"}

        # Test recursive, no pattern
        assert set(disk_backend.list_files(tmp_path, recursive=True)) == {
            tmp_path / "file1.txt", tmp_path / "file2.log", subdir / "file3.txt"
        }

        # Test recursive, txt pattern
        assert set(disk_backend.list_files(tmp_path, pattern="*.txt", recursive=True)) == {
            tmp_path / "file1.txt", subdir / "file3.txt"
        }

    # DEPRECATED tests for FileSystemManager delegation removed.
    # Native load/save logic should be tested separately if/when implemented.
 
    # We're now using our own implementation instead of delegating to the old FileSystemManager
    def test_list_image_files_native(self, disk_backend, tmp_path):
        # Setup
        (tmp_path / "file1.tif").touch()
        (tmp_path / "file2.png").touch()
        (tmp_path / "file3.txt").touch()  # Non-image file
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file4.tif").touch()

        # Test with default extensions (recursive by default)
        files = disk_backend.list_image_files(tmp_path)
        assert set(files) == {tmp_path / "file1.tif", tmp_path / "file2.png", subdir / "file4.tif"}

        # Test with specific extensions
        custom_ext = {".tif"}
        files = disk_backend.list_image_files(tmp_path, extensions=custom_ext)
        assert set(files) == {tmp_path / "file1.tif", subdir / "file4.tif"}

        # Test with non-recursive
        files = disk_backend.list_image_files(tmp_path, recursive=False)
        assert set(files) == {tmp_path / "file1.tif", tmp_path / "file2.png"}


    # ... more tests for other delegated methods (find_image_dir, rename, zstack etc.) ...