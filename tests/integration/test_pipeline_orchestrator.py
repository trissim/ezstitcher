import os
import shutil
import pytest
from pathlib import Path
import numpy as np

from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.config import ZStackProcessorConfig, StitcherConfig, PipelineConfig
from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator

# Define microscope configurations
MICROSCOPE_CONFIGS = {
    "ImageXpress": {
        "format": "ImageXpress",
        "test_dir_name": "imagexpress_pipeline",
        "microscope_type": "auto",  # Use auto-detection
        "auto_image_size": True,
    },
    "OperaPhenix": {
        "format": "OperaPhenix",
        "test_dir_name": "opera_phenix_pipeline",
        "microscope_type": "OperaPhenix",  # Explicitly specify type
        "auto_image_size": True,
    }
}

# Test parameters
syn_data_params = {
    "grid_size": (4, 4),
    "tile_size": (64, 64),
    "overlap_percent": 10,
    "wavelengths": 2,
    "cell_size_range": (3, 6),
    "wells": ['A01', 'B02'],
}

# Test-specific parameters that can be customized per microscope format
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
def test_function_dir(base_test_dir, microscope_config, request):
    """Create test directory for a specific test function."""
    # Get the test function name without the parameter
    test_name = request.node.originalname or request.node.name.split('[')[0]
    # Create a directory for this specific test function
    test_dir = base_test_dir / f"{test_name}[{microscope_config['format']}]"
    test_dir.mkdir(exist_ok=True)
    yield test_dir

@pytest.fixture(scope="module")
def test_params(microscope_config):
    """Get test parameters for the specific microscope type."""
    # Use the format key instead of microscope_type
    return TEST_PARAMS[microscope_config["format"]]["default"]

@pytest.fixture
def flat_plate_dir(test_function_dir, microscope_config, test_params):
    """Create synthetic flat plate data for the specified microscope type."""
    plate_dir = test_function_dir / "flat_plate"

    # Get parameters from test_params with defaults if not specified
    grid_size = test_params.get("grid_size", (3, 3))
    tile_size = test_params.get("tile_size", (128, 128))
    overlap_percent = test_params.get("overlap_percent", 10)
    wavelengths = test_params.get("wavelengths", 2)
    z_stack_levels = test_params.get("z_stack_levels", 1)
    cell_size_range = test_params.get("cell_size_range", (5, 10))
    wells = test_params.get("wells", ['A01'])

    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=grid_size,
        tile_size=tile_size,
        overlap_percent=overlap_percent,
        wavelengths=wavelengths,
        z_stack_levels=z_stack_levels,
        cell_size_range=cell_size_range,
        wells=wells,
        format=microscope_config["format"],
        auto_image_size=microscope_config["auto_image_size"]
    )
    generator.generate_dataset()

    # Create a copy of the original data for inspection
    original_dir = test_function_dir / "flat_plate_original"
    if not original_dir.exists():
        shutil.copytree(plate_dir, original_dir)

    # Always return the plate directory - let the core library handle the directory structure
    return plate_dir

@pytest.fixture
def zstack_plate_dir(test_function_dir, microscope_config, test_params):
    """Create synthetic Z-stack plate data for the specified microscope type."""
    plate_dir = test_function_dir / "zstack_plate"

    # Get parameters from test_params with defaults if not specified
    grid_size = test_params.get("grid_size", (3, 3))
    tile_size = test_params.get("tile_size", (128, 128))
    overlap_percent = test_params.get("overlap_percent", 10)
    wavelengths = test_params.get("wavelengths", 2)
    cell_size_range = test_params.get("cell_size_range", (5, 10))
    wells = test_params.get("wells", ['A01'])

    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=grid_size,
        tile_size=tile_size,
        overlap_percent=overlap_percent,
        wavelengths=wavelengths,
        z_stack_levels=5,  # Always use 5 z-stack levels for this fixture
        cell_size_range=cell_size_range,
        wells=wells,
        format=microscope_config["format"],
        auto_image_size=microscope_config["auto_image_size"]
    )
    generator.generate_dataset()

    # Create a copy of the original data for inspection
    original_dir = test_function_dir / "zstack_plate_original"
    if not original_dir.exists():
        shutil.copytree(plate_dir, original_dir)

    # Always return the plate directory - let the core library handle the directory structure
    return plate_dir

# Define preprocessing functions for testing
def enhance_contrast(img):
    """Simple contrast enhancement for testing."""
    p_low, p_high = np.percentile(img, (2, 98))
    return np.clip((img - p_low) * (65535 / (p_high - p_low)), 0, 65535).astype(np.uint16)

def background_subtract(img):
    """Simple background subtraction for testing."""
    background = np.percentile(img, 10)
    return np.clip(img - background, 0, 65535).astype(np.uint16)

def test_flat_plate_minimal(flat_plate_dir):
    """Test processing a flat plate with minimal configuration."""
    # Create pipeline configuration
    config = PipelineConfig(
        reference_channels=["1"],
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(flat_plate_dir)

    assert success, "Flat plate processing failed"

    # Check if output directories were created
    processed_dir = Path(flat_plate_dir).parent / f"{Path(flat_plate_dir).name}_processed"
    stitched_dir = Path(flat_plate_dir).parent / f"{Path(flat_plate_dir).name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created
    stitched_files = list(stitched_dir.glob("*.tif"))
    assert len(stitched_files) > 0, "No stitched files created"

def test_zstack_projection_minimal(zstack_plate_dir):
    """Test processing a Z-stack plate with projection."""
    # Create pipeline configuration
    config = PipelineConfig(
        reference_channels=["1"],
        zstack_config=ZStackProcessorConfig(
            reference_flatten="max"
        ),
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(zstack_plate_dir)

    assert success, "Z-stack projection processing failed"

    # Check if output directories were created
    processed_dir = Path(zstack_plate_dir).parent / f"{Path(zstack_plate_dir).name}_processed"
    stitched_dir = Path(zstack_plate_dir).parent / f"{Path(zstack_plate_dir).name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created
    stitched_files = list(stitched_dir.glob("*.tif"))
    assert len(stitched_files) > 0, "No stitched files created"

    # Check that no Z-plane files were created (since we're using max projection)
    z_files = list(stitched_dir.glob("*_z*.tif"))
    assert len(z_files) == 0, "Z-plane files were created when using max projection"

def test_zstack_per_plane_minimal(zstack_plate_dir):
    """Test processing a Z-stack plate with per-plane stitching."""
    # Create pipeline configuration
    config = PipelineConfig(
        reference_channels=["1","2"],
        zstack_config=ZStackProcessorConfig(
            reference_flatten="max",  # No projection, keep all planes
            stitch_flatten=None
        ),
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(zstack_plate_dir)

    assert success, "Z-stack per-plane processing failed"

    # Check if output directories were created
    processed_dir = Path(zstack_plate_dir).parent / f"{Path(zstack_plate_dir).name}_processed"
    stitched_dir = Path(zstack_plate_dir).parent / f"{Path(zstack_plate_dir).name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created
    all_files = list(stitched_dir.glob("*.tif"))
    print(f"All files in stitched directory: {[f.name for f in all_files]}")
    assert len(all_files) > 0, "No stitched files created"

    # Check if stitched files have Z-plane suffixes
    stitched_files = list(stitched_dir.glob("*_z*.tif"))
    print(f"Files with z suffixes: {[f.name for f in stitched_files]}")
    assert len(stitched_files) > 0, "No Z-plane files found"

def test_multi_channel_minimal(flat_plate_dir):
    """Test processing a flat plate with multiple reference channels."""
    # Create pipeline configuration
    config = PipelineConfig(
        reference_channels=["1", "2"],
        reference_composite_weights={
            "1": 0.7,
            "2": 0.3
        },
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(flat_plate_dir)

    assert success, "Multi-channel reference processing failed"

    # Check if output directories were created
    processed_dir = Path(flat_plate_dir).parent / f"{Path(flat_plate_dir).name}_processed"
    stitched_dir = Path(flat_plate_dir).parent / f"{Path(flat_plate_dir).name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created for both channels
    stitched_files = list(stitched_dir.glob("*.tif"))
    assert len(stitched_files) > 0, "No stitched files created"

def test_best_focus_reference(zstack_plate_dir):
    """Test processing a Z-stack plate using best focus planes to be assembled for stitching."""
    # Create pipeline configuration
    config = PipelineConfig(
        #reference_channels=["1"],
        zstack_config=ZStackProcessorConfig(
            stitch_flatten='best_focus',
            reference_flatten="max",
            focus_method="combined"
        ),
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(zstack_plate_dir)

    assert success, "Z-stack best focus reference processing failed"

    # Check if output directories were created
    processed_dir = Path(zstack_plate_dir).parent / f"{Path(zstack_plate_dir).name}_processed"
    stitched_dir = Path(zstack_plate_dir).parent / f"{Path(zstack_plate_dir).name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created
    stitched_files = list(stitched_dir.glob("*.tif"))
    assert len(stitched_files) > 0, "No stitched files created"

def test_preprocessing_functions(flat_plate_dir):
    """Test processing a flat plate with preprocessing functions."""
    # Create pipeline configuration
    config = PipelineConfig(
        reference_channels=["1"],
        reference_preprocessing={
            "1": enhance_contrast
        },
        final_preprocessing={
            "1": background_subtract
        },
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(flat_plate_dir)

    assert success, "Processing with preprocessing functions failed"

    # Check if output directories were created
    processed_dir = Path(flat_plate_dir).parent / f"{Path(flat_plate_dir).name}_processed"
    post_processed_dir = Path(flat_plate_dir).parent / f"{Path(flat_plate_dir).name}_post_processed"
    stitched_dir = Path(flat_plate_dir).parent / f"{Path(flat_plate_dir).name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert post_processed_dir.exists(), "Post-processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if processed files were created
    processed_files = list(processed_dir.glob("*.tif"))
    post_processed_files = list(post_processed_dir.glob("*.tif"))

    assert len(processed_files) > 0, "No processed files created"
    assert len(post_processed_files) > 0, "No post-processed files created"

def test_all_channels_stitched(flat_plate_dir):
    """Test that all available channels are stitched by default."""
    # Create pipeline configuration with only one reference channel
    config = PipelineConfig(
        reference_channels=["1"],  # Only specify channel 1 as reference
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(flat_plate_dir)

    assert success, "Processing with all channels failed"

    # Check if output directories were created
    stitched_dir = Path(flat_plate_dir).parent / f"{Path(flat_plate_dir).name}_stitched"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created for all channels, even though only channel 1 was specified as reference
    channel1_files = list(stitched_dir.glob("*_w1*.tif"))
    channel2_files = list(stitched_dir.glob("*_w2*.tif"))

    assert len(channel1_files) > 0, "No channel 1 stitched files created"
    assert len(channel2_files) > 0, "No channel 2 stitched files created (should be stitched by default)"
