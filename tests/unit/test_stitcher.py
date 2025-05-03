import pytest
from unittest.mock import patch, MagicMock, create_autospec
from pathlib import Path
import numpy as np
import pandas as pd

# Class under test
from ezstitcher.core.stitcher import Stitcher
# Mocks needed
from ezstitcher.io.filemanager import FileManager
from ezstitcher.core.config import StitcherConfig
from ezstitcher.core.microscope_interfaces import FilenameParser

@pytest.fixture
def mock_file_manager() -> MagicMock:
    """Provides a mock FileManager."""
    # Create a mock without spec_set to allow adding attributes
    mock = MagicMock(spec=FileManager)
    # Add backend attribute with a mock
    mock.backend = MagicMock()
    # Set up method behaviors
    mock.ensure_directory.side_effect = lambda p: Path(p) # Simple mock
    mock.load_image.return_value = np.zeros((10, 10), dtype=np.uint16) # Mock image load
    mock.save_image.return_value = True # Mock image save
    return mock

@pytest.fixture
def mock_config() -> MagicMock:
    """Provides a mock StitcherConfig."""
    mock = create_autospec(StitcherConfig, instance=True, spec_set=True)
    mock.margin_ratio = 0.1 # Example config value
    return mock

@pytest.fixture
def mock_filename_parser() -> MagicMock:
    """Provides a mock FilenameParser."""
    return create_autospec(FilenameParser, instance=True, spec_set=True)

@pytest.fixture
def mock_positions_df() -> pd.DataFrame:
    """Provides a mock positions DataFrame."""
    return pd.DataFrame({
        'file': ["file: tile1.tif", "file: tile2.tif"],
        'grid': [" grid: (0, 0)", " grid: (1, 0)"],
        'position': [" position: (0.0, 0.0)", " position: (9.0, 0.5)"]
    })

# --- Initialization Tests ---

def test_stitcher_init_with_injected_fm(mock_config, mock_filename_parser, mock_file_manager):
    """Test Stitcher uses the injected FileManager."""
    stitcher = Stitcher(config=mock_config, filename_parser=mock_filename_parser, file_manager=mock_file_manager)
    assert stitcher.file_manager is mock_file_manager

@patch('ezstitcher.core.stitcher.FileManager', autospec=True)
@patch('ezstitcher.io.storage_backend.DiskStorageBackend', autospec=True) # Patch DiskStorageBackend from correct module
def test_stitcher_init_creates_default_fm_with_warning(MockDiskBackend, MockFileManager, mock_config, mock_filename_parser, caplog):
    """Test Stitcher creates a default FileManager with a warning if none is provided."""
    mock_fm_instance = MockFileManager.return_value
    mock_disk_instance = MockDiskBackend.return_value
    mock_fm_instance.backend = mock_disk_instance # Link mock backend to mock FM

    stitcher = Stitcher(config=mock_config, filename_parser=mock_filename_parser, file_manager=None)

    MockDiskBackend.assert_called_once() # Check Disk backend was created
    MockFileManager.assert_called_once_with(mock_disk_instance) # Check FM was created with Disk backend
    assert stitcher.file_manager is mock_fm_instance
    assert "FileManager not injected into Stitcher" in caplog.text
    assert "Creating default (Disk backend)" in caplog.text

# --- Method Tests ---

@patch('ezstitcher.core.stitcher.pd.read_csv') # Mock pandas read_csv
@patch('ezstitcher.core.stitcher.subpixel_shift') # Mock scipy shift
def test_assemble_image_uses_file_manager(mock_shift, mock_read_csv, mock_config, mock_filename_parser, mock_file_manager, mock_positions_df):
    """Test assemble_image calls ensure_directory and load_image on the FileManager."""
    # Arrange
    stitcher = Stitcher(config=mock_config, filename_parser=mock_filename_parser, file_manager=mock_file_manager)
    positions_p = Path("path/to/positions.csv")
    images_p = Path("path/to/images")
    output_p = Path("path/to/output/stitched.tif")

    # Mock parse_positions_csv to return a list of (filename, x, y) tuples
    # This avoids the need to actually read the CSV file
    stitcher.parse_positions_csv = MagicMock(return_value=[
        ("tile1.tif", 0, 0),
        ("tile2.tif", 9, 0.5)
    ])

    # Mock Path.exists to return True for our test files
    with patch('pathlib.Path.exists', return_value=True):
        # Mock load_image to return consistent shape/dtype
        mock_file_manager.load_image.return_value = np.zeros((100, 100), dtype=np.uint16)
        # Mock shift to return input arrays (simplifies testing focus on FM)
        mock_shift.side_effect = lambda arr, **kwargs: arr

        # Act
        stitcher.assemble_image(positions_p, images_p, output_p)

    # Assert FileManager calls
    mock_file_manager.ensure_directory.assert_called_once_with(output_p.parent)
    # Check load_image calls based on mock_positions_df
    mock_file_manager.load_image.assert_any_call(images_p / "tile1.tif")
    mock_file_manager.load_image.assert_any_call(images_p / "tile2.tif")
    # The first tile is loaded twice - once to get dimensions and once for processing
    assert mock_file_manager.load_image.call_count == 3
    # Check save_image call
    mock_file_manager.save_image.assert_called_once()
    # Verify the output path passed to save_image
    args, kwargs = mock_file_manager.save_image.call_args
    saved_path, saved_img = args
    assert saved_path == output_p

# Add more tests for edge cases in assemble_image (e.g., missing files, shape mismatches)
# Add tests for generate_positions if it uses FileManager (currently seems to use parser directly)