import pytest
from unittest.mock import patch, MagicMock, create_autospec
import numpy as np
from pathlib import Path

# Classes to test
from ezstitcher.core.steps import Step, ImageStitchingStep
# Mocks needed
from ezstitcher.core.pipeline import ProcessingContext
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.io.filemanager import FileManager
from ezstitcher.core.microscope_interfaces import MicroscopeHandler # Use the actual class

@pytest.fixture
def mock_context() -> MagicMock:
    """Provides a mock ProcessingContext with nested mocks."""
    # Create a MagicMock instead of autospec to avoid strict attribute checking
    context = MagicMock(spec=ProcessingContext)
    # Setup nested mocks directly
    context.orchestrator = MagicMock(spec=PipelineOrchestrator)
    context.orchestrator.file_manager = MagicMock(spec=FileManager)
    context.orchestrator.file_manager.list_files = MagicMock(return_value=[])
    context.orchestrator.file_manager.backend = MagicMock(name="StorageBackend")
    context.orchestrator.microscope_handler = MagicMock(spec=MicroscopeHandler)
    # Mock the parser attribute, assuming it's often the handler itself or a sub-object
    context.orchestrator.microscope_handler.parser = MagicMock(spec=MicroscopeHandler) # Mock parser separately if needed

    # Default configurations
    context.get_step_input_dir.return_value = Path("mock/input")
    context.get_step_output_dir.return_value = Path("mock/output")
    context.well_filter = ["A01"]
    context.results = {} # Initialize results dict

    # Default mock behaviors
    fm_mock = context.orchestrator.file_manager
    fm_mock.find_image_directory.return_value = Path("mock/input/actual")
    fm_mock.load_image.return_value = np.ones((10, 10), dtype=np.uint8) # Default successful load
    fm_mock.save_image.return_value = True # Default successful save
    fm_mock.ensure_directory.return_value = Path("mock/output") # Return path
    fm_mock.exists.return_value = True # Assume files exist by default for simplicity

    parser_mock = context.orchestrator.microscope_handler.parser
    parser_mock.path_list_from_pattern.return_value = ["image1.tif"] # Default pattern match

    # Mock plate_path on orchestrator if needed by steps
    context.orchestrator.plate_path = Path("mock/plate")
    # Mock config on orchestrator if needed
    context.orchestrator.config = MagicMock()
    context.orchestrator.config.positions_dir_suffix = "_positions" # Example config value

    return context

def test_base_step_process_calls_filemanager(mock_context: MagicMock):
    """Test Step.process uses FileManager from context for load/save."""
    # Arrange
    step = Step(func=lambda x, **kwargs: [img * 2 for img in x], name="Test Multiply") # Ensure func matches expected signature
    fm_mock = mock_context.orchestrator.file_manager
    parser_mock = mock_context.orchestrator.microscope_handler.parser
    microscope_handler_mock = mock_context.orchestrator.microscope_handler

    # Mock auto_detect_patterns to return patterns for the well
    microscope_handler_mock.auto_detect_patterns.return_value = {'A01': ['pattern1']}

    # Use patch.object for methods on the class being tested
    with patch.object(Step, '_apply_processing', side_effect=lambda images, func: [func(images)[0]]): # Mock apply processing to match expected call
        # Act
        result_context = step.process(mock_context)

    # Assert
    # Check FileManager calls
    fm_mock.find_image_directory.assert_called_once_with(Path("mock/input"))
    parser_mock.path_list_from_pattern.assert_called_once_with(Path("mock/input/actual"), 'pattern1')
    fm_mock.load_image.assert_called_once_with(Path("mock/input/actual/image1.tif"))
    fm_mock.ensure_directory.assert_called_once_with(Path("mock/output"))
    # Check that save_image was called with the processed image (lambda x: x*2)
    fm_mock.save_image.assert_called_once() # Ensure it was called
    args, kwargs = fm_mock.save_image.call_args
    saved_img, saved_path = args
    assert np.array_equal(saved_img, np.ones((10, 10), dtype=np.uint8) * 2)
    assert saved_path == Path("mock/output/image1.tif")

    # Check context results
    assert 'A01' in result_context.results
    expected_path = str(Path("mock/output/image1.tif").resolve())
    assert result_context.results['A01'] == {'pattern1': [expected_path]}


def test_image_stitching_step_calls_orchestrator(mock_context: MagicMock):
    """Test ImageStitchingStep calls orchestrator.stitch_images."""
    # Arrange
    stitching_step = ImageStitchingStep(name="Test Stitch")
    orchestrator_mock = mock_context.orchestrator
    fm_mock = orchestrator_mock.file_manager

    # Setup plate path and positions file path
    plate_p = Path("mock/plate")
    orchestrator_mock.plate_path = plate_p
    positions_dir = plate_p.parent / f"{plate_p.name}_positions"
    positions_file = positions_dir / "A01.csv"

    # Mock FileManager.exists to return True only for the parent directory check
    fm_mock.exists.return_value = True # Or side_effect = [True]

    # Act
    stitching_step.process(mock_context)

    # Assert
    # Verify FileManager.exists was called for the parent positions directory
    fm_mock.exists.assert_called_once_with(positions_dir)
    # The code does NOT call exists(positions_file) if the directory exists, so we don't assert that.

    # Verify orchestrator.stitch_images was called
    # images_to_stitch_dir defaults to default_input_dir ('mock/input') in this case
    orchestrator_mock.stitch_images.assert_called_once_with(
        well="A01",
        input_dir=Path("mock/input"),
        output_dir=Path("mock/output"),
        positions_file=positions_file
    )


def test_image_stitching_step_finds_positions_via_recursive_search(mock_context: MagicMock):
    """Test ImageStitchingStep finds positions dir via recursive search fallback."""
    # Arrange
    stitching_step = ImageStitchingStep(name="Test Stitch Recursive Fallback")
    orchestrator_mock = mock_context.orchestrator
    fm_mock = orchestrator_mock.file_manager

    plate_p = Path("mock/plate")
    orchestrator_mock.plate_path = plate_p
    input_dir = Path("mock/input")
    parent_positions_dir = plate_p.parent / f"{plate_p.name}_positions"
    # Define the path that the recursive search should find
    found_positions_dir = Path("mock/some/other/location/my_positions_dir")
    expected_positions_file = found_positions_dir / "A01.csv"

    # Mock FileManager.exists: False for the primary check
    fm_mock.exists.return_value = False

    # Create a mock Path object that behaves like a directory with "positions" in the name
    mock_dir_path = MagicMock(spec=Path)
    mock_dir_path.is_dir.return_value = True
    mock_dir_path.name = "my_positions_dir"  # Contains "positions"

    # Create a list of files/directories for the recursive search
    fm_mock.list_files.return_value = [
        Path("mock/some/file.txt"),
        mock_dir_path,  # Use the mock Path that will pass the is_dir check
        Path("mock/some/other/another_file.csv")
    ]

    # Make the mock directory return the expected path when used with Path constructor
    # This simulates what happens when the code does Path(positions_dir) / f"{well}.csv"
    mock_dir_path.__truediv__.return_value = expected_positions_file

    # Set up the mock to return the expected path when converted to a Path
    with patch.object(Path, '__new__', return_value=found_positions_dir):
        mock_context.get_step_input_dir.return_value = input_dir

        # Act
        stitching_step.process(mock_context)

        # Assert
        # Verify FileManager.exists was called for the primary location
        fm_mock.exists.assert_called_once_with(parent_positions_dir)
        # Verify FileManager.list_files was called for the recursive search
        fm_mock.list_files.assert_called_once_with(Path(input_dir).parent, recursive=True)

        # Verify orchestrator.stitch_images was called with the path from the recursively found directory
        orchestrator_mock.stitch_images.assert_called_once_with(
            well="A01",
            input_dir=input_dir,
            output_dir=mock_context.get_step_output_dir.return_value,
            positions_file=expected_positions_file # Check it used the fallback path
        )


def test_image_stitching_step_raises_error_if_positions_not_found(mock_context: MagicMock):
    """Test ImageStitchingStep raises error if positions directory is not found."""
    # Arrange
    stitching_step = ImageStitchingStep(name="Test Stitch No Pos")
    orchestrator_mock = mock_context.orchestrator
    fm_mock = orchestrator_mock.file_manager

    plate_p = Path("mock/plate")
    orchestrator_mock.plate_path = plate_p
    input_dir = Path("mock/input")
    parent_positions_dir = plate_p.parent / f"{plate_p.name}_positions"

    # Mock FileManager.exists: False for primary check
    fm_mock.exists.return_value = False
    # Mock FileManager.list_files: Return empty list for recursive search
    fm_mock.list_files.return_value = []
    mock_context.get_step_input_dir.return_value = input_dir

    # Act & Assert - Check for the correct error message
    with pytest.raises(ValueError, match="No positions directory found for well A01"):
        stitching_step.process(mock_context)

    # Verify FileManager.exists call was made
    fm_mock.exists.assert_called_once_with(parent_positions_dir)
    # Verify FileManager.list_files call was made
    fm_mock.list_files.assert_called_once_with(Path(input_dir).parent, recursive=True)
    # Ensure stitch_images was NOT called
    orchestrator_mock.stitch_images.assert_not_called()


# Add tests for:
# - Error handling in base Step (e.g., find_image_directory fails, load_image returns None)
# - Multiple files per pattern in base Step
# - In-place processing (_save_images logic if input_dir == output_dir)
# - Different group_by / variable_components scenarios in base Step