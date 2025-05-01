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
from ezstitcher.core.file_system_manager import FileSystemManager # For mocking DiskStorageBackend delegations

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

    # Test DELEGATED methods (mock FileSystemManager)
    @patch.object(FileSystemManager, 'load_image', return_value=np.ones((2,2), dtype=np.uint8))
    def test_load_image_delegated(self, mock_fsm_load, disk_backend):
        # No tmp_path needed as we mock the filesystem interaction
        img_path = "path/does/not/need/to/exist/image.tif"
        img = disk_backend.load_image(img_path)
        mock_fsm_load.assert_called_once_with(img_path)
        assert img is not None
        np.testing.assert_array_equal(img, np.ones((2,2), dtype=np.uint8))

    @patch.object(FileSystemManager, 'save_image', return_value=True)
    def test_save_image_delegated(self, mock_fsm_save, disk_backend):
        img_path = "path/output.tif"
        img_data = np.zeros((3,3))
        result = disk_backend.save_image(img_data, img_path)
        # Check that FSM was called correctly (might depend on FSM signature)
        mock_fsm_save.assert_called_once_with(img_path, img_data)
        assert result is True

    @patch.object(FileSystemManager, 'list_image_files', return_value=[Path("img1.tif"), Path("img2.png")])
    def test_list_image_files_delegated(self, mock_fsm_list, disk_backend):
        dir_path = "some/dir"
        # Test with default extensions
        files = disk_backend.list_image_files(dir_path)
        mock_fsm_list.assert_called_once_with(dir_path, list(DEFAULT_IMAGE_EXTENSIONS), True)
        assert files == [Path("img1.tif"), Path("img2.png")]

        # Test with specific extensions
        mock_fsm_list.reset_mock()
        custom_ext = {".tiff"}
        disk_backend.list_image_files(dir_path, extensions=custom_ext, recursive=False)
        mock_fsm_list.assert_called_once_with(dir_path, list(custom_ext), False)


    # ... more tests for other delegated methods (find_image_dir, rename, zstack etc.) ...