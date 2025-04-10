import os
import shutil
import pytest
from pathlib import Path
from ezstitcher.core.main import process_plate_auto
from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator

@pytest.fixture(scope="module")
def base_test_dir():
    """Create base test directory for ImageXpress tests."""
    base_dir = Path(__file__).parent / "tests_data" / "imagexpress_refactored_auto"

    # Delete the directory if it exists
    if base_dir.exists():
        shutil.rmtree(base_dir)

    # Create the directory
    base_dir.mkdir(parents=True, exist_ok=True)

    yield base_dir

    # Uncomment to clean up after tests
    # shutil.rmtree(base_dir)

@pytest.fixture
def test_dir(base_test_dir, request):
    """Create test-specific directory."""
    test_name = request.node.name
    test_dir = base_test_dir / test_name

    # Create the directory if it doesn't exist
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSetting up test directory for {test_name}: {test_dir}")

    return test_dir

@pytest.fixture
def flat_plate_dir(test_dir):
    """Create synthetic flat plate data."""
    plate_dir = test_dir / "flat_plate"
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

    # Create a copy of the original data for inspection
    original_dir = test_dir / "flat_plate_original"
    if not original_dir.exists():
        shutil.copytree(plate_dir, original_dir)

    return plate_dir

@pytest.fixture
def zstack_plate_dir(test_dir):
    """Create synthetic Z-stack plate data."""
    plate_dir = test_dir / "zstack_plate"
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

    # Create a copy of the original data for inspection
    original_dir = test_dir / "zstack_plate_original"
    if not original_dir.exists():
        shutil.copytree(plate_dir, original_dir)

    return plate_dir

def test_flat_plate_minimal(flat_plate_dir):
    """Test processing a flat plate with minimal configuration."""
    success = process_plate_auto(
        flat_plate_dir,
        microscope_type="auto"  # Use auto-detection instead of "ImageXpress"
    )
    assert success, "Flat plate processing failed"

def test_zstack_projection_minimal(zstack_plate_dir):
    """Test processing a Z-stack plate with projection."""
    success = process_plate_auto(
        zstack_plate_dir,
        microscope_type="auto",  # Use auto-detection instead of "ImageXpress"
        **{"z_stack_processor.create_projections": True}
    )
    assert success, "Z-stack projection processing failed"

def test_zstack_per_plane_minimal(zstack_plate_dir):
    """Test processing a Z-stack plate with per-plane stitching."""
    success = process_plate_auto(
        zstack_plate_dir,
        microscope_type="auto",  # Use auto-detection instead of "ImageXpress"
        **{"z_stack_processor.stitch_all_z_planes": True}
    )
    assert success, "Z-stack per-plane processing failed"

def test_multi_channel_minimal(flat_plate_dir):
    """Test processing a flat plate with multiple reference channels."""
    success = process_plate_auto(
        flat_plate_dir,
        microscope_type="auto",  # Use auto-detection instead of "ImageXpress"
        **{"reference_channels": ["1", "2"]}
    )
    assert success, "Multi-channel reference processing failed"
