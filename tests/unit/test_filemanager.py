import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, create_autospec

# Assuming StorageBackend is the abstract base or interface
# Use the specific interface FileManager depends on
from ezstitcher.io.storage_backend import MicroscopyStorageBackend, DiskStorageBackend
from ezstitcher.io.filemanager import FileManager
# Recommended: Define a FakeStorageBackend for integration-style tests
# from ezstitcher.io.fake_storage_backend import FakeStorageBackend # Example path
# For now, we'll use mocks primarily for unit testing FileManager's delegation

# If constants are used for default extensions, import them for testing
from ezstitcher.io.constants import DEFAULT_IMAGE_EXTENSIONS
from ezstitcher.io.types import ImageArray # Import type alias

class TestFileManager:
    """Tests for the FileManager adapter class."""

    # --- Initialization Tests ---

    def test_init_default_backend_is_disk(self):
        """Verify FileManager defaults to DiskStorageBackend if none provided."""
        fm = FileManager()
        assert isinstance(fm.backend, DiskStorageBackend), \
            "Default backend should be DiskStorageBackend"

    def test_init_with_custom_backend(self):
        """Verify FileManager accepts and uses a provided backend instance."""
        # Use create_autospec to ensure mock matches the MicroscopyStorageBackend interface
        mock_backend = create_autospec(MicroscopyStorageBackend, instance=True)
        fm = FileManager(backend=mock_backend)
        assert fm.backend is mock_backend, \
            "FileManager should use the injected backend"

    def test_init_with_incorrect_backend_type_raises_error(self):
        """Verify FileManager raises TypeError if backend doesn't match interface."""
        class WrongBackend: # Doesn't implement MicroscopyStorageBackend
            pass
        with pytest.raises(TypeError, match="Backend must implement MicroscopyStorageBackend"):
             FileManager(backend=WrongBackend())


    # --- Delegation Tests (Using Mocks) ---

    # Subsection: Validating Dependency Injection and Correct Delegation
    # These tests use mock objects (MagicMock/create_autospec) to isolate FileManager.
    # They confirm that:
    # 1. FileManager correctly uses the injected backend (Dependency Injection).
    # 2. Method calls on FileManager are correctly delegated to the backend instance
    #    with the exact arguments passed.
    # This validates the adapter pattern implementation without needing real file ops.

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Provides a mock MicroscopyStorageBackend for testing delegation."""
        # Use autospec to ensure the mock adheres to the MicroscopyStorageBackend signature
        # Specify methods explicitly if autospec has issues with complex signatures or ABCs
        spec = MicroscopyStorageBackend
        # spec_set=True ensures mock only has attributes/methods defined in the spec
        return create_autospec(spec, instance=True, spec_set=True)


    @pytest.fixture
    def file_manager(self, mock_backend: MagicMock) -> FileManager:
        """Provides a FileManager instance initialized with the mock backend."""
        return FileManager(backend=mock_backend)

    def test_load_image_delegates(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test load_image calls backend.load_image with the correct path."""
        test_path = Path("images/test.tif")
        mock_image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        mock_backend.load_image.return_value = mock_image

        result = file_manager.load_image(test_path)

        mock_backend.load_image.assert_called_once_with(test_path)
        assert np.array_equal(result, mock_image)

    def test_save_image_delegates(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test save_image calls backend.save_image with correct args."""
        test_path = Path("output/saved.png")
        test_image: ImageArray = np.zeros((5, 5), dtype=np.uint16) # Use type alias
        test_metadata = {"key": "value"}
        mock_backend.save_image.return_value = True

        result = file_manager.save_image(test_image, test_path, test_metadata)

        # Use assert_called_once_with, checking positional and keyword args
        # Need to ensure the mock handles the np.ndarray comparison correctly if needed,
        # but usually checking the call signature is sufficient for delegation tests.
        mock_backend.save_image.assert_called_once_with(test_image, test_path, test_metadata)
        assert result is True

    def test_list_image_files_delegates_with_extensions(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test list_image_files delegates with explicit extensions."""
        test_dir = Path("data/")
        extensions = {".tif", ".png"} # Use Set as per type hint
        mock_files = [Path("data/img1.tif"), Path("data/subdir/img2.png")]
        mock_backend.list_image_files.return_value = mock_files

        result = file_manager.list_image_files(test_dir, extensions=extensions, recursive=True)

        mock_backend.list_image_files.assert_called_once_with(test_dir, extensions=extensions, recursive=True)
        assert result == mock_files

    def test_list_image_files_delegates_with_none_extensions(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test list_image_files delegates with None extensions (backend handles default)."""
        test_dir = Path("data/")
        mock_files = [Path("data/img1.tif")]
        mock_backend.list_image_files.return_value = mock_files

        result = file_manager.list_image_files(test_dir, extensions=None, recursive=False)

        # FileManager passes None; backend is responsible for interpreting this
        mock_backend.list_image_files.assert_called_once_with(test_dir, extensions=None, recursive=False)
        assert result == mock_files

    def test_ensure_directory_delegates(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test ensure_directory calls backend.ensure_directory."""
        test_dir = Path("new/dir")
        # Simulate backend returning the resolved path
        resolved_path = Path("/tmp/new/dir") # Example resolved path
        mock_backend.ensure_directory.return_value = resolved_path

        result = file_manager.ensure_directory(test_dir)

        mock_backend.ensure_directory.assert_called_once_with(test_dir)
        assert result == resolved_path

    def test_find_image_directory_delegates(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test find_image_directory delegates correctly."""
        test_plate_folder = Path("plate_data")
        extensions = {".tif"}
        expected_dir = Path("plate_data/Images")
        mock_backend.find_image_directory.return_value = expected_dir

        # Test with specific extensions
        result = file_manager.find_image_directory(test_plate_folder, extensions=extensions)
        mock_backend.find_image_directory.assert_called_once_with(test_plate_folder, extensions=extensions)
        assert result == expected_dir

        # Test with None extensions
        mock_backend.find_image_directory.reset_mock()
        mock_backend.find_image_directory.return_value = expected_dir # Reset return value if needed
        result_none = file_manager.find_image_directory(test_plate_folder, extensions=None)
        mock_backend.find_image_directory.assert_called_once_with(test_plate_folder, extensions=None)
        assert result_none == expected_dir

    def test_find_file_recursive_delegates(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test find_file_recursive delegates correctly."""
        test_dir = Path("search_root")
        filename = "target.txt"
        expected_path = Path("search_root/subdir/target.txt")
        mock_backend.find_file_recursive.return_value = expected_path

        result = file_manager.find_file_recursive(test_dir, filename)
        mock_backend.find_file_recursive.assert_called_once_with(test_dir, filename)
        assert result == expected_path

        # Test case where file is not found
        mock_backend.find_file_recursive.reset_mock()
        mock_backend.find_file_recursive.return_value = None
        result_none = file_manager.find_file_recursive(test_dir, "not_found.txt")
        mock_backend.find_file_recursive.assert_called_once_with(test_dir, "not_found.txt")
        assert result_none is None

    def test_delete_file_delegates(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test delete_file delegates correctly."""
        test_path = Path("file_to_delete.tmp")
        mock_backend.delete_file.return_value = True

        result = file_manager.delete_file(test_path)
        mock_backend.delete_file.assert_called_once_with(test_path)
        assert result is True

        # Test failure case
        mock_backend.delete_file.reset_mock()
        mock_backend.delete_file.return_value = False
        result_fail = file_manager.delete_file(test_path)
        mock_backend.delete_file.assert_called_once_with(test_path)
        assert result_fail is False

    def test_copy_file_delegates(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test copy_file delegates correctly."""
        source_path = Path("source/file.dat")
        dest_path = Path("dest/file_copy.dat")
        mock_backend.copy_file.return_value = True

        result = file_manager.copy_file(source_path, dest_path)
        mock_backend.copy_file.assert_called_once_with(source_path, dest_path)
        assert result is True

        # Test failure case
        mock_backend.copy_file.reset_mock()
        mock_backend.copy_file.return_value = False
        result_fail = file_manager.copy_file(source_path, dest_path)
        mock_backend.copy_file.assert_called_once_with(source_path, dest_path)
        assert result_fail is False

    def test_exists_delegates(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test exists delegates correctly."""
        test_path = Path("some/path/file.txt")
        mock_backend.exists.return_value = True

        result = file_manager.exists(test_path)
        mock_backend.exists.assert_called_once_with(test_path)
        assert result is True

        # Test False case
        mock_backend.exists.reset_mock()
        mock_backend.exists.return_value = False
        result_false = file_manager.exists(test_path)
        mock_backend.exists.assert_called_once_with(test_path)
        assert result_false is False

    def test_rename_files_delegates(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test rename_files_with_consistent_padding delegates correctly."""
        test_dir = Path("rename_dir")
        mock_parser = MagicMock() # Mock the parser object
        width = 3
        force = False
        expected_map = {Path("rename_dir/f1"): Path("rename_dir/f001")}
        mock_backend.rename_files_with_consistent_padding.return_value = expected_map

        result = file_manager.rename_files_with_consistent_padding(test_dir, mock_parser, width, force)
        mock_backend.rename_files_with_consistent_padding.assert_called_once_with(test_dir, mock_parser, width, force)
        assert result == expected_map

    def test_detect_zstack_delegates(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test detect_zstack_folders delegates correctly."""
        test_plate = Path("plate")
        pattern = "Z*"
        expected_result = (True, [Path("plate/Z01"), Path("plate/Z02")])
        mock_backend.detect_zstack_folders.return_value = expected_result

        result = file_manager.detect_zstack_folders(test_plate, pattern)
        mock_backend.detect_zstack_folders.assert_called_once_with(test_plate, pattern)
        assert result == expected_result

    def test_organize_zstack_delegates(self, file_manager: FileManager, mock_backend: MagicMock):
        """Test organize_zstack_folders delegates correctly."""
        test_plate = Path("plate_org")
        mock_parser = MagicMock()
        mock_backend.organize_zstack_folders.return_value = True

        result = file_manager.organize_zstack_folders(test_plate, mock_parser)
        mock_backend.organize_zstack_folders.assert_called_once_with(test_plate, mock_parser)
        assert result is True


    # --- Integration-Style Tests (Recommendation) ---

    # Subsection: Integration Testing with Fake Backend (Recommended)
    # While mock tests verify delegation, integration-style tests using a
    # FakeStorageBackend (in-memory implementation) are highly recommended.
    # These tests validate the interaction contract between FileManager and a
    # concrete backend implementation more thoroughly, catching potential
    # mismatches in assumptions or behavior without hitting the real filesystem.

    # @pytest.mark.skip(reason="FakeStorageBackend not yet implemented or available")
    # def test_save_then_load_integration_with_fake_backend(self):
    #     """Example integration-style test using a hypothetical FakeStorageBackend."""
    #     # Assumes FakeStorageBackend exists and mimics StorageBackend interface
    #     from ezstitcher.io.storage_backend import FakeStorageBackend # Use actual Fake
    #     fake_backend = FakeStorageBackend()
    #     file_manager = FileManager(backend=fake_backend)
    #     test_image = np.random.rand(10, 10).astype(np.uint8) # Example image
    #     test_path = Path("virtual/path/image.tif")
    #
    #     # Action 1: Save via FileManager -> FakeBackend
    #     save_success = file_manager.save_image(test_image, test_path)
    #     assert save_success is True
    #     assert fake_backend.exists(test_path), "File should exist in fake backend"
    #
    #     # Action 2: Load via FileManager -> FakeBackend
    #     loaded_image = file_manager.load_image(test_path)
    #     assert loaded_image is not None
    #     np.testing.assert_array_equal(loaded_image, test_image, "Loaded image should match saved image")
    #
    #     # Action 3: List files via FileManager -> FakeBackend
    #     files = file_manager.list_image_files(test_path.parent, extensions={".tif"})
    #     assert files == [test_path]
    #
    #     # Action 4: Delete via FileManager -> FakeBackend
    #     delete_success = file_manager.delete_file(test_path)
    #     assert delete_success is True
    #     assert not fake_backend.exists(test_path), "File should not exist after deletion"