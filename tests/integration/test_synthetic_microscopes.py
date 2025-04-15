import os
import shutil
import pytest
from pathlib import Path
from ezstitcher.core.main import process_plate_auto
from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator

# Define microscope configurations
MICROSCOPE_CONFIGS = {
    "ImageXpress": {
        "format": "ImageXpress",
        "test_dir_name": "imagexpress_refactored",
        "microscope_type": "auto",  # Use auto-detection
        "auto_image_size": True,
    },
    "OperaPhenix": {
        "format": "OperaPhenix",
        "test_dir_name": "opera_phenix_refactored",
        "microscope_type": "OperaPhenix",  # Explicitly specify type
        "auto_image_size": True,
    }
}
syn_data_params = {
                      "grid_size": (4, 4),
                      "tile_size": (128, 128),
                      "overlap_percent": 10,
                      "wavelengths": 2,
                      "cell_size_range": (5, 10),
                      "wells": ['A01', 'B02'],
                  }
# Test-specific parameters that can be customized per microscope
TEST_PARAMS = {
    "ImageXpress": {
        "default": syn_data_params
        # Add test-specific overrides here if needed
    },
    "OperaPhenix": {
        "default": syn_data_params
        # Add test-specific overrides here if needed
    }
}

@pytest.fixture(scope="module", params=list(MICROSCOPE_CONFIGS.keys()))
def microscope_config(request):
    """Provide microscope configuration based on the parameter."""
    return MICROSCOPE_CONFIGS[request.param]

@pytest.fixture(scope="module")
def base_test_dir(microscope_config):
    """Create base test directory for the specific microscope type."""
    base_dir = Path(__file__).parent / "tests_data" / microscope_config["test_dir_name"]

    # Delete the directory if it exists
    if base_dir.exists():
        print(f"Cleaning up existing test data directory: {base_dir}")
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
def test_params(request, microscope_config):
    """Get test-specific parameters for the current microscope type."""
    test_name = request.node.name
    microscope_type = microscope_config["format"]

    # Get default parameters for this microscope type
    default_params = TEST_PARAMS.get(microscope_type, {}).get("default", {})

    # Get test-specific parameters (if any)
    test_specific_params = TEST_PARAMS.get(microscope_type, {}).get(test_name, {})

    # Merge default and test-specific parameters, with test-specific taking precedence
    params = {**default_params, **test_specific_params}

    return params

@pytest.fixture
def flat_plate_dir(test_dir, microscope_config, test_params):
    """Create synthetic flat plate data for the specified microscope type."""
    plate_dir = test_dir / "flat_plate"

    # Get parameters from test_params with defaults if not specified
    grid_size = test_params.get("grid_size", (3, 3))
    tile_size = test_params.get("tile_size", (128, 128))
    overlap_percent = test_params.get("overlap_percent", 10)
    wavelengths = test_params.get("wavelengths", 2)
    z_stack_levels = test_params.get("z_stack_levels", 1)
    cell_size_range = test_params.get("cell_size_range", (5, 10))
    wells = test_params.get("wells", ['A01'])  # Extract wells parameter

    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=grid_size,
        tile_size=tile_size,
        overlap_percent=overlap_percent,
        wavelengths=wavelengths,
        z_stack_levels=z_stack_levels,
        cell_size_range=cell_size_range,
        wells=wells,  # Pass wells parameter
        format=microscope_config["format"],
        auto_image_size=microscope_config["auto_image_size"]
    )
    generator.generate_dataset()

    # Create a copy of the original data for inspection
    original_dir = test_dir / "flat_plate_original"
    if not original_dir.exists():
        shutil.copytree(plate_dir, original_dir)

    # Always return the plate directory - let the core library handle the directory structure
    return plate_dir

@pytest.fixture
def zstack_plate_dir(test_dir, microscope_config, test_params):
    """Create synthetic Z-stack plate data for the specified microscope type."""
    plate_dir = test_dir / "zstack_plate"

    # Get parameters from test_params with defaults if not specified
    grid_size = test_params.get("grid_size", (3, 3))
    tile_size = test_params.get("tile_size", (128, 128))
    overlap_percent = test_params.get("overlap_percent", 10)
    wavelengths = test_params.get("wavelengths", 2)
    cell_size_range = test_params.get("cell_size_range", (5, 10))
    wells = test_params.get("wells", ['A01'])  # Extract wells parameter

    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=grid_size,
        tile_size=tile_size,
        overlap_percent=overlap_percent,
        wavelengths=wavelengths,
        z_stack_levels=5,  # Always use 5 z-stack levels for this fixture
        cell_size_range=cell_size_range,
        wells=wells,  # Pass wells parameter
        format=microscope_config["format"],
        auto_image_size=microscope_config["auto_image_size"]
    )
    generator.generate_dataset()

    # Create a copy of the original data for inspection
    original_dir = test_dir / "zstack_plate_original"
    if not original_dir.exists():
        shutil.copytree(plate_dir, original_dir)

    # Always return the plate directory - let the core library handle the directory structure
    return plate_dir

def test_flat_plate_minimal(flat_plate_dir, microscope_config):
    """Test processing a flat plate with minimal configuration."""
    success = process_plate_auto(
        flat_plate_dir,
        microscope_type=microscope_config["microscope_type"]
    )
    assert success, "Flat plate processing failed"

def test_zstack_projection_minimal(zstack_plate_dir, microscope_config):
    """Test processing a Z-stack plate with projection."""
    success = process_plate_auto(
        zstack_plate_dir,
        microscope_type=microscope_config["microscope_type"],
        **{"z_stack_processor.create_projections": True}
    )
    assert success, "Z-stack projection processing failed"

def test_zstack_per_plane_minimal(zstack_plate_dir, microscope_config):
    from pathlib import Path
    """Test processing a Z-stack plate with per-plane stitching."""
    success = process_plate_auto(
        zstack_plate_dir,
        microscope_type="auto",
        **{"z_stack_processor.stitch_all_z_planes": True}
    )
    assert success, "Z-stack per-plane processing failed"

    # Check if stitched directory was created
    stitched_dir = Path(zstack_plate_dir).parent / f"{Path(zstack_plate_dir).name}_stitched"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check what files are in the stitched directory
    all_files = list(stitched_dir.glob("*.tif"))
    print(f"All files in stitched directory: {[f.name for f in all_files]}")

    # Check if stitched files have Z-plane suffixes
    stitched_files = list(stitched_dir.glob("*_z*.tif"))
    print(f"Files with z suffixes: {[f.name for f in stitched_files]}")
    assert len(stitched_files) > 0, "No Z-plane files found"

def test_multi_channel_minimal(flat_plate_dir, microscope_config):
    """Test processing a flat plate with multiple reference channels."""
    success = process_plate_auto(
        flat_plate_dir,
        microscope_type=microscope_config["microscope_type"],
        **{"reference_channels": ["1", "2"]}
    )
    assert success, "Multi-channel reference processing failed"


def test_best_focus_reference(zstack_plate_dir, microscope_config):
    """Test processing a Z-stack plate using best focus as reference for stitching.

    This test finds the best focus z-plane for each tile and uses those for stitching,
    without stitching all z-planes.
    """
    success = process_plate_auto(
        zstack_plate_dir,
        microscope_type=microscope_config["microscope_type"],
        **{
            "z_stack_processor.create_projections": False,
            "z_stack_processor.focus_detect": True,
            "z_stack_processor.use_best_focus_for_reference": True,
            "z_stack_processor.stitch_all_z_planes": False  # Only stitch the best focus planes
        }
    )
    assert success, "Z-stack best focus reference processing failed"