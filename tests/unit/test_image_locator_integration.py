import os
import shutil
import pytest
from pathlib import Path
import tempfile

from ezstitcher.core.image_locator import ImageLocator
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.plate_processor import PlateProcessor
from ezstitcher.core.config import PlateProcessorConfig
from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator

@pytest.fixture(scope="module")
def base_test_dir():
    base_dir = Path(tempfile.mkdtemp())
    yield base_dir
    # Clean up after tests
    shutil.rmtree(base_dir)

@pytest.fixture
def flat_plate_dir(base_test_dir):
    plate_dir = base_test_dir / "flat_plate"
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=(2, 2),
        image_size=(256, 256),
        tile_size=(128, 128),
        overlap_percent=10,
        wavelengths=2,
        z_stack_levels=1,
        format="ImageXpress"
    )
    generator.generate_dataset()
    return plate_dir

def test_file_system_manager_list_image_files(flat_plate_dir):
    """Test that FileSystemManager.list_image_files() uses ImageLocator."""
    fs_manager = FileSystemManager()
    
    # Get images using FileSystemManager
    timepoint_dir = flat_plate_dir / "TimePoint_1"
    fs_images = fs_manager.list_image_files(timepoint_dir)
    
    # Get images directly using ImageLocator
    locator_images = ImageLocator.find_images_in_directory(timepoint_dir)
    
    # Verify that both methods return the same results
    assert len(fs_images) == len(locator_images)
    assert set(fs_images) == set(locator_images)

def test_plate_processor_initialize_filename_parser(flat_plate_dir):
    """Test that PlateProcessor._initialize_filename_parser_and_convert() uses ImageLocator."""
    # Create a PlateProcessor with auto microscope type
    config = PlateProcessorConfig(microscope_type='auto')
    processor = PlateProcessor(config)
    
    # Initialize the filename parser
    processor._initialize_filename_parser_and_convert(flat_plate_dir)
    
    # Verify that the filename parser was initialized correctly
    assert processor.filename_parser is not None
    assert processor.filename_parser.__class__.__name__ == 'ImageXpressFilenameParser'

def test_image_locator_find_timepoint_dir(flat_plate_dir):
    """Test that ImageLocator.find_timepoint_dir() works correctly."""
    # Find the timepoint directory
    timepoint_dir = ImageLocator.find_timepoint_dir(flat_plate_dir)
    
    # Verify that the timepoint directory was found
    assert timepoint_dir is not None
    assert timepoint_dir.name == "TimePoint_1"
    assert timepoint_dir.parent == flat_plate_dir

def test_image_locator_find_image_locations(flat_plate_dir):
    """Test that ImageLocator.find_image_locations() works correctly."""
    # Find all image locations
    image_locations = ImageLocator.find_image_locations(flat_plate_dir)
    
    # Verify that the timepoint location was found
    assert 'timepoint' in image_locations
    assert len(image_locations['timepoint']) > 0
    
    # Verify that all images are Path objects
    for location_type, images in image_locations.items():
        if location_type != 'z_stack':  # z_stack is a nested dictionary
            for img_path in images:
                assert isinstance(img_path, Path)
                assert img_path.exists()
