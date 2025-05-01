import pytest
from unittest.mock import patch, MagicMock, create_autospec
from pathlib import Path
import numpy as np # Import numpy if needed for mock return values

# Classes under test or needed for context
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.config import PipelineConfig
# Import interfaces/classes for mocking/type hinting
from ezstitcher.core.microscope_interfaces import MicroscopeHandler # Use the actual class
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.image_processor import ImageProcessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.pipeline import Pipeline, ProcessingContext, StepExecutionPlan # Added Pipeline imports

# Mocked dependencies
from ezstitcher.io.filemanager import FileManager

# Use autospec to create mocks that match the interface of the real classes
@pytest.fixture
def mock_file_manager() -> MagicMock:
    """Provides a mock FileManager adhering to its interface."""
    # Use MagicMock instead of create_autospec to avoid strict attribute checking
    mock = MagicMock(spec=FileManager)
    # Add backend attribute
    mock.backend = MagicMock(name="StorageBackend")
    # Add backend_type attribute for disk backend check
    mock.backend_type = "disk"
    # Add root_dir attribute needed for workspace creation
    mock.root_dir = Path("/mock/root/dir")
    # Simple mock returns input path converted to Path
    mock.ensure_directory.side_effect = lambda p: Path(p)
    # Mock find_image_directory to return a plausible path based on input
    # Make sure to include the extensions parameter in the lambda
    mock.find_image_directory.side_effect = lambda p, extensions=None: Path(p) / "Images" # Example logic
    # Mock list_image_files to return an empty list by default
    mock.list_image_files.return_value = []
    # Mock list_files method
    mock.list_files = MagicMock(return_value=[])
    # Mock other methods as needed for specific tests
    mock.rename_files_with_consistent_padding.return_value = {}
    # Mock detect_zstack_folders to return a tuple of (has_zstack, z_folders)
    mock.detect_zstack_folders.return_value = (False, [])
    mock.organize_zstack_folders.return_value = True
    # Mock create_symlink method
    mock.create_symlink = MagicMock(return_value=True)
    # Mock rename method
    mock.rename = MagicMock(return_value=True)
    # Mock find_file_recursive method
    mock.find_file_recursive = MagicMock(return_value=None)
    return mock

@pytest.fixture
def mock_config() -> MagicMock:
    """Provides a mock PipelineConfig."""
    # Use MagicMock instead of create_autospec to avoid strict attribute checking
    mock = MagicMock(spec=PipelineConfig)
    mock.well_filter = None # Default: no filter
    # Set other necessary config attributes
    mock.stitcher = MagicMock() # Mock stitcher config if needed
    mock.num_workers = 1 # Default workers for tests
    mock.positions_dir_suffix = "_positions"
    mock.stitched_dir_suffix = "_stitched"
    mock.out_dir_suffix = "_output"
    mock.grid_size = (3, 3) # Example grid size
    mock.pixel_size = 0.65 # Example pixel size
    return mock

@pytest.fixture
def mock_microscope_handler() -> MagicMock:
    """Provides a mock Microscope Handler."""
    # Use MagicMock instead of create_autospec to avoid strict attribute checking
    mock = MagicMock(spec=MicroscopeHandler)
    # Mock necessary methods like parse_filename
    def mock_parse(filename):
        name = Path(filename).stem # Extract base name
        parts = name.split('_')
        well = parts[1] if len(parts) > 1 else "Unknown"
        site = parts[3] if len(parts) > 3 else "S1"
        # Return a dictionary that includes expected keys
        return {'well': well, 'site': site, 'filename': filename, 'extension': Path(filename).suffix, 'channel': 'ch1', 'z_index': 1} # Example parsing
    mock.parse_filename.side_effect = mock_parse
    # Mock methods called during init or prepare_images if necessary
    mock.get_grid_dimensions.return_value = (3, 3)
    mock.get_pixel_size.return_value = 0.65
    # Mock parser attribute if Stitcher needs it
    mock.parser = mock # Often the handler itself acts as the parser
    # Mock init_workspace if it's called during orchestrator init
    mock.init_workspace = MagicMock()
    # Mock auto_detect_patterns if needed by tests
    mock.auto_detect_patterns.return_value = {} # Default empty patterns
    return mock

@pytest.fixture
def mock_stitcher() -> MagicMock:
    """Provides a mock Stitcher."""
    return create_autospec(Stitcher, instance=True, spec_set=True)

@pytest.fixture
def mock_image_processor() -> MagicMock:
    """Provides a mock ImageProcessor."""
    return create_autospec(ImageProcessor, instance=True, spec_set=True)

@pytest.fixture
def mock_focus_analyzer() -> MagicMock:
    """Provides a mock FocusAnalyzer."""
    return create_autospec(FocusAnalyzer, instance=True, spec_set=True)


# Patch dependencies instantiated within PipelineOrchestrator if not injected.
# Use context manager within tests where needed, or fixture if applied globally.
@pytest.fixture
def patch_internal_deps():
    """ Context manager/fixture to patch dependencies created inside Orchestrator """
    # Patch the specific modules where these classes/functions are looked up
    with patch('ezstitcher.core.pipeline_orchestrator.FileManager', autospec=True) as MockFileManager, \
         patch('ezstitcher.core.pipeline_orchestrator.create_microscope_handler', autospec=True) as MockCreateHandler, \
         patch('ezstitcher.core.pipeline_orchestrator.Stitcher', autospec=True) as MockStitcher, \
         patch('ezstitcher.core.pipeline_orchestrator.ImageProcessor', autospec=True) as MockImageProcessor, \
         patch('ezstitcher.core.pipeline_orchestrator.FocusAnalyzer', autospec=True) as MockFocusAnalyzer, \
         patch('ezstitcher.core.pipeline_orchestrator.PipelineConfig', autospec=True) as MockPipelineConfig: # Patch config too if default is created
        # Yield a dictionary of the mocks for potential use in tests
        yield {
            "FileManager": MockFileManager,
            "CreateHandler": MockCreateHandler,
            "Stitcher": MockStitcher,
            "ImageProcessor": MockImageProcessor,
            "FocusAnalyzer": MockFocusAnalyzer,
            "PipelineConfig": MockPipelineConfig,
        }


def test_orchestrator_init_with_injected_fm(mock_file_manager: MagicMock, mock_config: MagicMock, patch_internal_deps):
    """Test initialization correctly uses the injected FileManager."""
    # Mock create_microscope_handler to return a mock instance
    mock_handler_instance = MagicMock(spec=MicroscopeHandler)
    mock_handler_instance.parser = MagicMock() # Mock parser attribute
    patch_internal_deps["CreateHandler"].return_value = mock_handler_instance

    # Prevent default FM creation by providing one
    orchestrator = PipelineOrchestrator(
        plate_path="dummy/plate",
        config=mock_config,
        file_manager=mock_file_manager # Inject the mock
    )
    assert orchestrator.file_manager is mock_file_manager
    # Verify ensure_directory was called on workspace path during init
    expected_workspace = Path("dummy/plate_workspace")
    mock_file_manager.ensure_directory.assert_called_once_with(expected_workspace)
    # Verify create_microscope_handler was called
    patch_internal_deps["CreateHandler"].assert_called_once_with('auto', plate_folder=Path("dummy/plate"))
    # Verify prepare_images was called (which calls find_image_directory etc. on fm)
    # The actual call might not include the extensions parameter explicitly
    mock_file_manager.find_image_directory.assert_called_once_with(Path("dummy/plate"))


def test_orchestrator_init_creates_default_fm_if_none_provided(mock_config: MagicMock, patch_internal_deps):
    """Test initialization creates a default FileManager if not injected."""
    MockFileManager = patch_internal_deps["FileManager"]
    # Create a mock instance with the necessary attributes
    mock_fm_instance = MagicMock(spec=FileManager)
    mock_fm_instance.backend = MagicMock(name="StorageBackend")
    # Add backend_type attribute for disk backend check
    mock_fm_instance.backend_type = "disk"
    # Add root_dir attribute needed for workspace creation
    mock_fm_instance.root_dir = Path("/mock/root/dir")
    # Set up find_image_directory to handle extensions parameter
    mock_fm_instance.find_image_directory.side_effect = lambda p, extensions=None: Path(p) / "Images"
    # Mock create_symlink method
    mock_fm_instance.create_symlink = MagicMock(return_value=True)
    # Mock rename method
    mock_fm_instance.rename = MagicMock(return_value=True)
    # Mock find_file_recursive method
    mock_fm_instance.find_file_recursive = MagicMock(return_value=None)
    # Mock list_files method
    mock_fm_instance.list_files = MagicMock(return_value=[])
    MockFileManager.return_value = mock_fm_instance # Set as the return value

    # Mock create_microscope_handler
    mock_handler_instance = MagicMock(spec=MicroscopeHandler)
    mock_handler_instance.parser = MagicMock()
    patch_internal_deps["CreateHandler"].return_value = mock_handler_instance

    orchestrator = PipelineOrchestrator(plate_path="dummy/plate", config=mock_config, file_manager=None)

    MockFileManager.assert_called_once_with() # Check default constructor called
    assert orchestrator.file_manager is mock_fm_instance
    # Verify ensure_directory was called on the default instance
    expected_workspace = Path("dummy/plate_workspace")
    mock_fm_instance.ensure_directory.assert_called_once_with(expected_workspace)
    # Verify prepare_images calls on the default fm instance
    # The actual call might not include the extensions parameter explicitly
    mock_fm_instance.find_image_directory.assert_called_once_with(Path("dummy/plate"))


def test_get_wells_to_process_calls_fm_list_files(mock_file_manager: MagicMock, mock_config: MagicMock, mock_microscope_handler: MagicMock, patch_internal_deps):
    """Verify _get_wells_to_process uses file_manager.list_image_files."""
    # Setup mock return value for list_image_files
    mock_file_manager.list_image_files.return_value = [
        Path("dummy/input/Well_A1_Site_1.tif"),
        Path("dummy/input/Well_A1_Site_2.tif"),
        Path("dummy/input/Well_B2_Site_1.tif"),
    ]
    # Mock find_image_directory needed by prepare_images called during init
    input_dir = Path("dummy/input")
    mock_file_manager.find_image_directory.return_value = input_dir

    # Mock create_microscope_handler to return our specific mock handler
    patch_internal_deps["CreateHandler"].return_value = mock_microscope_handler

    orchestrator = PipelineOrchestrator(
        plate_path="dummy/plate",
        file_manager=mock_file_manager,
        config=mock_config
    )
    # Orchestrator init calls prepare_images which calls find_image_directory etc.
    mock_file_manager.find_image_directory.assert_called()
    # The orchestrator.input_dir will be set to the result of find_image_directory
    # which is mocked to return Path(p) / "Images", so we need to update our assertion
    assert orchestrator.input_dir == Path("dummy/plate") / "Images"
    assert orchestrator.microscope_handler is mock_microscope_handler

    wells = orchestrator._get_wells_to_process()

    # Assert file_manager method was called correctly
    # The actual call will use orchestrator.input_dir which is set to Path("dummy/plate") / "Images"
    mock_file_manager.list_image_files.assert_called_once_with(Path("dummy/plate/Images"), recursive=True)
    # Assert the result based on orchestrator logic and mocked file listing/parsing
    assert wells == ['A1', 'B2'] # Should be sorted
    # Check that parse_filename was called for each file
    assert mock_microscope_handler.parse_filename.call_count == 3


def test_prepare_images_calls_fm_methods(mock_file_manager: MagicMock, mock_microscope_handler: MagicMock, patch_internal_deps):
    """Verify prepare_images calls the correct FileManager methods."""
    plate_p = Path("dummy/plate")
    image_dir = plate_p / "Images"
    mock_file_manager.find_image_directory.return_value = image_dir
    mock_file_manager.detect_zstack_folders.return_value = (True, [image_dir / "Z01"]) # Simulate Z-stacks

    patch_internal_deps["CreateHandler"].return_value = mock_microscope_handler

    orchestrator = PipelineOrchestrator(plate_path=plate_p, file_manager=mock_file_manager)

    # Reset mocks called during init before calling prepare_images directly if needed,
    # or verify calls made during init first.
    # For this test, let's assume init completed and we call prepare_images again (though usually called internally)
    mock_file_manager.reset_mock() # Reset mocks for clarity
    mock_file_manager.find_image_directory.return_value = image_dir
    mock_file_manager.detect_zstack_folders.return_value = (True, [image_dir / "Z01"])

    result_dir = orchestrator.prepare_images(plate_p)

    assert result_dir == image_dir
    # The actual call might not include the extensions parameter explicitly
    mock_file_manager.find_image_directory.assert_called_once_with(plate_p)
    mock_file_manager.rename_files_with_consistent_padding.assert_called_once_with(
        directory=image_dir,
        parser=mock_microscope_handler,
        width=pytest.approx(3), # Use approx for default value check
        force_suffixes=True
    )
    # The actual call might not include the pattern parameter explicitly
    mock_file_manager.detect_zstack_folders.assert_called_once_with(image_dir)
    mock_file_manager.organize_zstack_folders.assert_called_once_with(
        plate_folder=image_dir,
        filename_parser=mock_microscope_handler
    )


def test_stitch_images_calls_fm_ensure_directory(mock_file_manager: MagicMock, mock_stitcher: MagicMock, mock_microscope_handler: MagicMock, mock_config: MagicMock, patch_internal_deps):
    """Verify stitch_images uses FileManager to ensure output directory."""
    plate_p = Path("dummy/plate")
    input_dir = plate_p / "Images"
    output_dir = plate_p / "Stitched"
    well = "A1"
    positions_file = plate_p / "Positions" / f"{well}.csv"

    # Mock methods called during init
    mock_file_manager.find_image_directory.return_value = input_dir
    patch_internal_deps["CreateHandler"].return_value = mock_microscope_handler
    patch_internal_deps["Stitcher"].return_value = mock_stitcher # Ensure stitcher is mocked

    orchestrator = PipelineOrchestrator(
        plate_path=plate_p,
        file_manager=mock_file_manager,
        config=mock_config
    )
    # Ensure stitcher instance is set during init
    assert orchestrator.stitcher is mock_stitcher

    # Reset mocks called during init
    mock_file_manager.reset_mock()
    # Re-configure mocks needed for stitch_images
    mock_file_manager.find_image_directory.return_value = input_dir

    # Call the method under test
    # Note: stitch_images has complex internal logic (getting patterns etc.)
    # We focus on verifying the FileManager interaction here.
    # We need to mock get_patterns_for_well where it's imported and used
    # Mock where it's used, not where it's defined
    with patch('ezstitcher.core.pipeline_orchestrator.get_patterns_for_well',
               return_value=["pattern1.tif"]):
        orchestrator.stitch_images(well, input_dir, output_dir, positions_file)

    # Verify ensure_directory was called
    mock_file_manager.ensure_directory.assert_called_once_with(output_dir)
    # Verify find_image_directory was called again inside stitch_images
    # The actual call might not include the extensions parameter explicitly
    mock_file_manager.find_image_directory.assert_called_once_with(input_dir)
    # Verify the stitcher's assemble_image was called (if mocking that deep)
    # mock_stitcher.assemble_image.assert_called_once() # Add more specific args check if needed