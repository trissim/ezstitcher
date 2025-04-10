import os
import shutil
import pytest
from pathlib import Path
from ezstitcher.core.main import process_plate_auto
from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator

@pytest.fixture(scope="module")
def base_test_dir(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("auto_config_tests")
    yield base_dir
    # Uncomment to clean up after tests
    # shutil.rmtree(base_dir)

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

@pytest.fixture
def zstack_plate_dir(base_test_dir):
    plate_dir = base_test_dir / "zstack_plate"
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=(2, 2),
        image_size=(256, 256),
        tile_size=(128, 128),
        overlap_percent=10,
        wavelengths=2,
        z_stack_levels=5,
        format="ImageXpress"
    )
    generator.generate_dataset()
    return plate_dir

def test_auto_config_flat_plate(flat_plate_dir):
    """Test process_plate_auto with no config on a flat plate."""
    success = process_plate_auto(flat_plate_dir)
    assert success, "Flat plate processing with auto config failed"

def test_auto_config_zstack_plate(zstack_plate_dir):
    """Test process_plate_auto with no config on a Z-stack plate."""
    success = process_plate_auto(zstack_plate_dir)
    assert success, "Z-stack plate processing with auto config failed"

def test_auto_config_with_overrides(flat_plate_dir):
    """Test process_plate_auto with no config but with overrides."""
    success = process_plate_auto(
        flat_plate_dir,
        **{"stitcher.tile_overlap": 15}
    )
    assert success, "Flat plate processing with auto config and overrides failed"
