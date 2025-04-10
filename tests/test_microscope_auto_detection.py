import os
import shutil
import pytest
from pathlib import Path
import tempfile

from ezstitcher.core.filename_parser import create_parser, ImageXpressFilenameParser, OperaPhenixFilenameParser
from ezstitcher.core.plate_processor import PlateProcessor
from ezstitcher.core.config import PlateProcessorConfig
from ezstitcher.core.main import process_plate_auto
from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator

@pytest.fixture(scope="module")
def base_test_dir():
    base_dir = Path(tempfile.mkdtemp())
    yield base_dir
    # Clean up after tests
    shutil.rmtree(base_dir)

@pytest.fixture
def imagexpress_plate_dir(base_test_dir):
    plate_dir = base_test_dir / "imagexpress_plate"
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

@pytest.fixture
def opera_phenix_plate_dir(base_test_dir):
    plate_dir = base_test_dir / "opera_phenix_plate"
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=(2, 2),
        image_size=(256, 256),
        tile_size=(128, 128),
        overlap_percent=10,
        wavelengths=2,
        z_stack_levels=1,
        format="OperaPhenix"
    )
    generator.generate_dataset()
    return plate_dir

def test_create_parser_with_auto_and_sample_files():
    """Test create_parser with 'auto' and sample files."""
    # ImageXpress sample files
    imagexpress_files = [
        "A01_s001_w1.tif",
        "A01_s001_w2.tif",
        "A01_s002_w1.tif",
        "A01_s002_w2.tif"
    ]
    parser = create_parser('auto', sample_files=imagexpress_files)
    assert isinstance(parser, ImageXpressFilenameParser)
    
    # Opera Phenix sample files
    opera_phenix_files = [
        "r01c01f001p01-ch1sk1fk1fl1.tiff",
        "r01c01f001p01-ch2sk1fk1fl1.tiff",
        "r01c01f002p01-ch1sk1fk1fl1.tiff",
        "r01c01f002p01-ch2sk1fk1fl1.tiff"
    ]
    parser = create_parser('auto', sample_files=opera_phenix_files)
    assert isinstance(parser, OperaPhenixFilenameParser)

def test_create_parser_with_auto_and_plate_folder(imagexpress_plate_dir, opera_phenix_plate_dir):
    """Test create_parser with 'auto' and a plate folder."""
    # ImageXpress plate folder
    parser = create_parser('auto', plate_folder=imagexpress_plate_dir)
    assert isinstance(parser, ImageXpressFilenameParser)
    
    # Opera Phenix plate folder
    parser = create_parser('auto', plate_folder=opera_phenix_plate_dir)
    assert isinstance(parser, OperaPhenixFilenameParser)

def test_plate_processor_initialize_filename_parser(imagexpress_plate_dir, opera_phenix_plate_dir):
    """Test PlateProcessor._initialize_filename_parser_and_convert with 'auto'."""
    # ImageXpress plate folder
    config = PlateProcessorConfig(microscope_type='auto')
    processor = PlateProcessor(config)
    processor._initialize_filename_parser_and_convert(imagexpress_plate_dir)
    assert isinstance(processor.filename_parser, ImageXpressFilenameParser)
    
    # Opera Phenix plate folder
    config = PlateProcessorConfig(microscope_type='auto')
    processor = PlateProcessor(config)
    processor._initialize_filename_parser_and_convert(opera_phenix_plate_dir)
    assert isinstance(processor.filename_parser, OperaPhenixFilenameParser)

def test_process_plate_auto_with_no_config(imagexpress_plate_dir):
    """Test process_plate_auto with no config (should default to 'auto')."""
    success = process_plate_auto(imagexpress_plate_dir)
    assert success, "process_plate_auto with no config failed"
