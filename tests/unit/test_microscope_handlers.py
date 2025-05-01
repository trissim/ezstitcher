import pytest
from unittest.mock import patch, MagicMock, create_autospec
from pathlib import Path

# Interfaces and classes under test
from ezstitcher.core.microscope_interfaces import (
    MicroscopeHandler,
    MetadataHandler,
    FilenameParser,
    create_microscope_handler
)
# Specific implementations to test/mock
from ezstitcher.microscopes.opera_phenix import OperaPhenixMetadataHandler, OperaPhenixFilenameParser
from ezstitcher.microscopes.imagexpress import ImageXpressMetadataHandler, ImageXpressFilenameParser
# Mocked dependencies
from ezstitcher.io.filemanager import FileManager
from ezstitcher.io.storage_backend import DiskStorageBackend, BasicStorageBackend # Import backend for mocking

# --- Fixtures ---

@pytest.fixture
def mock_file_manager() -> MagicMock:
    """Provides a mock FileManager."""
    # Create a mock without spec_set to allow adding attributes
    mock = MagicMock(spec=FileManager)
    # Mock backend attribute
    mock.backend = MagicMock(spec=BasicStorageBackend)
    # Default behaviors
    mock.find_file_recursive.return_value = None
    mock.list_image_files.return_value = []  # Use list_image_files instead of list_files
    mock.exists.return_value = False
    return mock

@pytest.fixture
def mock_parser() -> MagicMock:
    """Provides a mock FilenameParser."""
    return create_autospec(FilenameParser, instance=True, spec_set=True)

@pytest.fixture
def mock_metadata_handler() -> MagicMock:
    """Provides a mock MetadataHandler."""
    return create_autospec(MetadataHandler, instance=True, spec_set=True)

# --- Tests for Specific Handlers ---

def test_opera_handler_init_uses_injected_fm(mock_file_manager):
    """Test OperaPhenixMetadataHandler uses injected FileManager."""
    handler = OperaPhenixMetadataHandler(file_manager=mock_file_manager)
    assert handler.file_manager is mock_file_manager

@patch('ezstitcher.microscopes.opera_phenix.FileManager', autospec=True)
def test_opera_handler_init_creates_default_fm(MockFileManager, caplog):
    """Test OperaPhenixMetadataHandler creates default FM with warning."""
    mock_fm_instance = MockFileManager.return_value
    handler = OperaPhenixMetadataHandler(file_manager=None)
    MockFileManager.assert_called_once()
    assert handler.file_manager is mock_fm_instance
    assert "FileManager not injected into OperaPhenixMetadataHandler" in caplog.text

def test_opera_handler_find_metadata_uses_fm(mock_file_manager):
    """Test OperaPhenixMetadataHandler.find_metadata_file uses FileManager."""
    handler = OperaPhenixMetadataHandler(file_manager=mock_file_manager)
    plate_p = Path("plate/path")
    expected_path = plate_p / "Index.xml"
    mock_file_manager.find_file_recursive.return_value = expected_path

    result = handler.find_metadata_file(plate_p)

    mock_file_manager.find_file_recursive.assert_called_once_with(plate_p, "Index.xml")
    assert result == expected_path

def test_imagexpress_handler_init_uses_injected_fm(mock_file_manager):
    """Test ImageXpressMetadataHandler uses injected FileManager."""
    handler = ImageXpressMetadataHandler(file_manager=mock_file_manager)
    assert handler.file_manager is mock_file_manager

@patch('ezstitcher.microscopes.imagexpress.FileManager', autospec=True)
def test_imagexpress_handler_init_creates_default_fm(MockFileManager, caplog):
    """Test ImageXpressMetadataHandler creates default FM with warning."""
    mock_fm_instance = MockFileManager.return_value
    handler = ImageXpressMetadataHandler(file_manager=None)
    MockFileManager.assert_called_once()
    assert handler.file_manager is mock_fm_instance
    assert "FileManager not injected into ImageXpressMetadataHandler" in caplog.text

@patch('ezstitcher.microscopes.imagexpress.tifffile.TiffFile') # Mock tifffile
def test_imagexpress_get_pixel_size_uses_fm(mock_tiff_file, mock_file_manager):
    """Test ImageXpressMetadataHandler.get_pixel_size uses FileManager."""
    handler = ImageXpressMetadataHandler(file_manager=mock_file_manager)
    plate_p = Path("plate/path")
    image_p = plate_p / "img.tif"
    mock_file_manager.list_image_files.return_value = [image_p]

    # Mock TiffFile context manager and tag reading
    mock_tiff_page = MagicMock()
    mock_tiff_page.tags = {'ImageDescription': MagicMock(value='id="spatial-calibration-x" value="0.5"')}
    mock_tiff_context = MagicMock()
    mock_tiff_context.__enter__.return_value.pages = [mock_tiff_page]
    mock_tiff_file.return_value = mock_tiff_context

    pixel_size = handler.get_pixel_size(plate_p)

    mock_file_manager.list_image_files.assert_called_once_with(plate_p, extensions={'.tif', '.tiff'}, recursive=True)
    mock_tiff_file.assert_called_once_with(image_p)
    assert pixel_size == 0.5

# --- Tests for MicroscopeHandler ---

def test_microscope_handler_init_uses_injected_fm(mock_parser, mock_metadata_handler, mock_file_manager):
    """Test MicroscopeHandler uses injected FileManager."""
    handler = MicroscopeHandler(parser=mock_parser, metadata_handler=mock_metadata_handler, file_manager=mock_file_manager)
    assert handler.file_manager is mock_file_manager

@patch('ezstitcher.core.microscope_interfaces.FileManager', autospec=True)
def test_microscope_handler_init_creates_default_fm(MockFileManager, mock_parser, mock_metadata_handler, caplog):
    """Test MicroscopeHandler creates default FM with warning."""
    mock_fm_instance = MockFileManager.return_value
    handler = MicroscopeHandler(parser=mock_parser, metadata_handler=mock_metadata_handler, file_manager=None)
    MockFileManager.assert_called_once()
    assert handler.file_manager is mock_fm_instance
    assert "FileManager not injected into MicroscopeHandler" in caplog.text

def test_microscope_handler_init_workspace_uses_mirror(mock_parser, mock_metadata_handler, mock_file_manager):
    """Test init_workspace calls mirror method on backend via FileManager."""
    # Mock the mirror method specifically on the backend mock
    mock_backend = mock_file_manager.backend
    mock_backend.mirror_directory_with_symlinks = MagicMock(return_value=50)

    handler = MicroscopeHandler(parser=mock_parser, metadata_handler=mock_metadata_handler, file_manager=mock_file_manager)
    plate_p = Path("plate")
    workspace_p = Path("workspace")

    count = handler.init_workspace(plate_p, workspace_p)

    mock_backend.mirror_directory_with_symlinks.assert_called_once_with(plate_p, workspace_p)
    assert count == 50
    # Ensure fallback ensure_directory was NOT called
    mock_file_manager.ensure_directory.assert_not_called()

def test_microscope_handler_init_workspace_fallback(mock_parser, mock_metadata_handler, mock_file_manager):
    """Test init_workspace falls back to ensure_directory if mirror method absent."""
    # Remove the mirror method from the backend mock to simulate absence
    mock_backend = mock_file_manager.backend
    del mock_backend.mirror_directory_with_symlinks

    handler = MicroscopeHandler(parser=mock_parser, metadata_handler=mock_metadata_handler, file_manager=mock_file_manager)
    plate_p = Path("plate")
    workspace_p = Path("workspace")

    count = handler.init_workspace(plate_p, workspace_p)

    # Verify fallback: ensure_directory was called, count is 0
    mock_file_manager.ensure_directory.assert_called_once_with(workspace_p)
    assert count == 0

# --- Tests for create_microscope_handler Factory ---

@patch('ezstitcher.core.microscope_interfaces.MicroscopeHandler._discover_handlers')
def test_create_handler_reuses_injected_fm(mock_discover, mock_file_manager):
    """Verify factory passes injected FileManager to handlers."""
    # Mock discovery to return specific classes
    MockOperaParser = create_autospec(OperaPhenixFilenameParser)
    MockOperaMeta = create_autospec(OperaPhenixMetadataHandler)
    # Include both naming conventions in the mock to match our implementation
    mock_discover.return_value = {
        'opera_phenix': (MockOperaParser, MockOperaMeta),
        'operaphenix': (MockOperaParser, MockOperaMeta)
    }

    # Mock FM behavior needed for auto-detection
    mock_file_manager.find_file_recursive.return_value = Path('fake_plate/Index.xml') # Simulate finding Opera file

    handler = create_microscope_handler(microscope_type='auto',
                                        plate_folder='fake_plate',
                                        file_manager=mock_file_manager) # Inject FM

    # Check auto-detection used FM
    mock_file_manager.find_file_recursive.assert_called_once_with(Path('fake_plate'), "Index.xml")
    # Check the correct handler type was chosen and FM was passed
    MockOperaMeta.assert_called_once()
    call_args, call_kwargs = MockOperaMeta.call_args
    assert call_kwargs.get('file_manager') is mock_file_manager
    # Check the final handler instance has the correct FM
    assert handler.file_manager is mock_file_manager
    # The metadata_handler is created with MockOperaMeta() which returns a mock with spec
    # We can't use isinstance with MagicMock directly, so check the spec instead
    assert handler.metadata_handler._spec_class == OperaPhenixMetadataHandler

@patch('ezstitcher.core.microscope_interfaces.MicroscopeHandler._discover_handlers')
@patch('ezstitcher.core.microscope_interfaces.FileManager', autospec=True)
def test_create_handler_creates_default_fm(MockFileManager, mock_discover, caplog):
    """Verify factory creates default FM if none provided and passes it."""
    # Mock discovery
    MockIXParser = create_autospec(ImageXpressFilenameParser)
    MockIXMeta = create_autospec(ImageXpressMetadataHandler)
    mock_discover.return_value = {'imagexpress': (MockIXParser, MockIXMeta)}

    # Mock FM behavior for auto-detection (simulate no Index.xml, but find HTD)
    mock_fm_instance = MockFileManager.return_value
    mock_fm_instance.find_file_recursive.return_value = None
    # Use list_image_files instead of list_files to match the actual method used
    mock_fm_instance.list_image_files.return_value = [Path("plate/file.HTD")] # Simulate finding HTD

    handler = create_microscope_handler(microscope_type='auto',
                                        plate_folder='plate',
                                        file_manager=None) # DO NOT inject FM

    # Check default FM was created
    MockFileManager.assert_called_once()
    assert "FileManager not provided to create_microscope_handler" in caplog.text
    # Check auto-detection used the default FM instance
    mock_fm_instance.find_file_recursive.assert_called_once_with(Path('plate'), "Index.xml")
    # Use list_image_files instead of list_files to match the actual method used
    mock_fm_instance.list_image_files.assert_called_once_with(Path('plate'), extensions={'.htd'}, recursive=False)
    # Check correct handler chosen and default FM passed
    MockIXMeta.assert_called_once()
    call_args, call_kwargs = MockIXMeta.call_args
    assert call_kwargs.get('file_manager') is mock_fm_instance
    # Check final handler instance has the default FM
    assert handler.file_manager is mock_fm_instance

# Test for microscope type normalization
def test_discover_handlers_normalizes_microscope_types():
    """Test that _discover_handlers normalizes microscope types with and without underscores."""
    # Clear the cache to ensure we get a fresh discovery
    MicroscopeHandler._handlers_cache = None

    # Get the handlers
    handlers = MicroscopeHandler._discover_handlers()

    # Check that both 'opera_phenix' and 'operaphenix' are in the handlers
    assert 'opera_phenix' in handlers
    assert 'operaphenix' in handlers

    # Check that they point to the same classes
    assert handlers['opera_phenix'] == handlers['operaphenix']

    # Check that the classes are the correct ones
    assert handlers['opera_phenix'][0] == OperaPhenixFilenameParser
    assert handlers['opera_phenix'][1] == OperaPhenixMetadataHandler

@patch('ezstitcher.core.microscope_interfaces.MicroscopeHandler._discover_handlers')
def test_create_handler_works_with_both_naming_conventions(mock_discover, mock_file_manager):
    """Test that create_microscope_handler works with both 'opera_phenix' and 'operaphenix'."""
    # Mock discovery to return specific classes
    MockOperaParser = create_autospec(OperaPhenixFilenameParser)
    MockOperaMeta = create_autospec(OperaPhenixMetadataHandler)
    # Include both naming conventions in the mock
    mock_discover.return_value = {
        'opera_phenix': (MockOperaParser, MockOperaMeta),
        'operaphenix': (MockOperaParser, MockOperaMeta)
    }

    # Test with underscore version
    handler1 = create_microscope_handler(
        microscope_type='opera_phenix',
        file_manager=mock_file_manager
    )

    # Test with no underscore version
    handler2 = create_microscope_handler(
        microscope_type='operaphenix',
        file_manager=mock_file_manager
    )

    # Both should work and create handlers
    assert handler1 is not None
    assert handler2 is not None

    # Both should have the same parser and metadata handler classes
    # We can't use isinstance with MagicMock directly, so check the spec instead
    assert handler1.parser._spec_class == OperaPhenixFilenameParser
    assert handler2.parser._spec_class == OperaPhenixFilenameParser

    # Also verify the metadata handlers have the correct spec
    assert handler1.metadata_handler._spec_class == OperaPhenixMetadataHandler
    assert handler2.metadata_handler._spec_class == OperaPhenixMetadataHandler