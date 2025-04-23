import os
import shutil
import pytest
from pathlib import Path
import numpy as np
from typing import List, Union

from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.config import StitcherConfig, PipelineConfig
from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator
from ezstitcher.core.image_locator import ImageLocator


def find_image_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = True) -> List[Path]:
    """
    Find all image files in a directory matching a pattern, using all supported extensions.
    Recursively searches in subdirectories by default to handle nested well folders.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match (default: "*" for all files)
        recursive: Whether to search recursively in subdirectories (default: True)

    Returns:
        List of Path objects for image files
    """
    directory = Path(directory)
    image_files = []

    # Use rglob for recursive search or glob for non-recursive
    glob_func = directory.rglob if recursive else directory.glob

    for ext in ImageLocator.DEFAULT_EXTENSIONS:
        image_files.extend(list(glob_func(f"**/{pattern}{ext}" if recursive else f"{pattern}{ext}")))

    return sorted(image_files)

# Define microscope configurations
MICROSCOPE_CONFIGS = {
    "ImageXpress": {
        "format": "ImageXpress",
        "test_dir_name": "imagexpress_pipeline",
        "microscope_type": "auto",  # Use auto-detection
        "auto_image_size": True
    },
    "OperaPhenix": {
        "format": "OperaPhenix",
        "test_dir_name": "opera_phenix_pipeline",
        "microscope_type": "auto",  # Explicitly specify type
        "auto_image_size": True
    }
}

# Test parameters
syn_data_params = {
    "grid_size": (4, 4),
    "tile_size": (64, 64),
    "overlap_percent": 10,
    "wavelengths": 2,
    "cell_size_range": (3, 6),
    "wells": ['A01','H08'],
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

    ##### FIX THIS######
    # uncomment to clean up after tests
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


# Import the ImagePreprocessor for stack functions
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.utils import track_thread_activity, clear_thread_activity, print_thread_activity_report

# Create an instance of ImagePreprocessor for testing
_image_preprocessor = ImagePreprocessor()

# Define a wrapper function for stack_equalize_histogram
def normalize(stack):
    """Apply true histogram equalization to an entire stack."""
    return _image_preprocessor.stack_percentile_normalize(stack,low_percentile=0.1, high_percentile=99.99)

@pytest.fixture
def thread_tracker():
    """Fixture to track thread activity for tests."""
    # Store the original method
    original_process_well = PipelineOrchestrator.process_well

    # Apply the decorator to the process_well method
    PipelineOrchestrator.process_well = track_thread_activity(original_process_well)

    # Clear any previous thread activity data
    clear_thread_activity()

    # Provide the fixture
    yield

    # Restore the original method
    PipelineOrchestrator.process_well = original_process_well


@pytest.fixture
def base_pipeline_config(microscope_config):
    """Create a base pipeline configuration with default values."""
    config = PipelineConfig(
        reference_channels=["1"],
        cleanup_processed=False,
        cleanup_post_processed=False,
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        )
    )
    # We don't need to set workspace_path as it's handled in the PipelineOrchestrator.run method
    return config

def create_config(base_config, **kwargs):
    """Create a new configuration by overriding base config values.

    Args:
        base_config: Base configuration object
        **kwargs: Values to override in the base configuration

    Returns:
        New configuration with overridden values
    """
    # Create a copy of the base config dict
    config_dict = base_config.__dict__.copy()

    # Handle special case for reference_composite_weights
    if 'reference_composite_weights' in kwargs and isinstance(kwargs['reference_composite_weights'], dict):
        # Convert dictionary weights to a list
        weights_dict = kwargs['reference_composite_weights']
        channels = kwargs.get('reference_channels', config_dict.get('reference_channels', []))

        # Create a list of weights in the same order as channels
        weights_list = [weights_dict.get(channel, 0.0) for channel in channels]
        kwargs['reference_composite_weights'] = weights_list

    # Override with new values
    for key, value in kwargs.items():
        config_dict[key] = value

    # Create a new config object
    return PipelineConfig(**config_dict)

def test_flat_plate_minimal(flat_plate_dir, base_pipeline_config, thread_tracker):
    """Test processing a flat plate with minimal configuration."""
    # Use the base configuration
    config = base_pipeline_config

    # Ensure num_workers is set to a value greater than 1
    config.num_workers = 2

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(flat_plate_dir)

    assert success, "Flat plate processing failed"

    # Check if output directories were created
    # Use the plate path to check for output directories
    plate_path = Path(flat_plate_dir)
    workspace_path = plate_path.parent / f"{plate_path.name}_workspace"
    processed_dir = workspace_path.parent / f"{workspace_path.name}_processed"
    stitched_dir = workspace_path.parent / f"{workspace_path.name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created
    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched files created"

    # Print and analyze thread activity
    analysis = print_thread_activity_report()

    # Assert that multiple threads were used if num_workers > 1
    if config.num_workers > 1:
        assert analysis['max_concurrent'] > 1, f"Expected multiple concurrent threads, but only {analysis['max_concurrent']} was used"
        assert len(analysis['overlaps']) > 0, "Expected thread overlaps, but none were found"
    else:
        print("Skipping multithreading check since num_workers=1")

def test_zstack_projection_minimal(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test processing a Z-stack plate with projection."""
    # Create pipeline configuration based on the base config
    config = create_config(base_pipeline_config, reference_flatten="max_projection")

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(zstack_plate_dir)

    assert success, "Z-stack projection processing failed"

    # Check if output directories were created
    # Use the plate path to check for output directories
    plate_path = Path(zstack_plate_dir)
    workspace_path = plate_path.parent / f"{plate_path.name}_workspace"
    processed_dir = workspace_path.parent / f"{workspace_path.name}_processed"
    stitched_dir = workspace_path.parent / f"{workspace_path.name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created
    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched files created"

    # Print thread activity report
    print_thread_activity_report()

def test_zstack_per_plane_minimal(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test processing a Z-stack plate with per-plane stitching."""
    # Create pipeline configuration based on the base config
    config = create_config(
        base_pipeline_config,
        reference_channels=["1","2"],
        reference_flatten="max",  # No projection, keep all planes
        stitch_flatten=None
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(zstack_plate_dir)

    assert success, "Z-stack per-plane processing failed"

    # Check if output directories were created
    # Use the plate path to check for output directories
    plate_path = Path(zstack_plate_dir)
    workspace_path = plate_path.parent / f"{plate_path.name}_workspace"
    processed_dir = workspace_path.parent / f"{workspace_path.name}_processed"
    stitched_dir = workspace_path.parent / f"{workspace_path.name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created
    all_files = find_image_files(stitched_dir)
    print(f"All files in stitched directory: {[f.name for f in all_files]}")
    assert len(all_files) > 0, "No stitched files created"

    # Print thread activity report
    print_thread_activity_report()

def test_multi_channel_minimal(flat_plate_dir, base_pipeline_config, thread_tracker):
    """Test processing a flat plate with multiple reference channels."""
    # Create pipeline configuration based on the base config
    config = create_config(
        base_pipeline_config,
        reference_channels=["1", "2"],
        reference_composite_weights={"1": 0.7, "2": 0.3}  # Use dictionary format for weights
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(flat_plate_dir)

    assert success, "Multi-channel reference processing failed"

    # Check if output directories were created
    # Use the plate path to check for output directories
    plate_path = Path(flat_plate_dir)
    workspace_path = plate_path.parent / f"{plate_path.name}_workspace"
    processed_dir = workspace_path.parent / f"{workspace_path.name}_processed"
    stitched_dir = workspace_path.parent / f"{workspace_path.name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created for both channels
    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched files created"

    # Print thread activity report
    print_thread_activity_report()

def test_best_focus_reference(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test processing a Z-stack plate using best focus planes to be assembled for stitching."""
    # Create pipeline configuration based on the base config
    config = create_config(
        base_pipeline_config,
        reference_flatten="max_projection",
        stitch_flatten='best_focus',
        focus_method="combined"
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(zstack_plate_dir)

    assert success, "Z-stack best focus reference processing failed"

    # Check if output directories were created
    # Use the plate path to check for output directories
    plate_path = Path(zstack_plate_dir)
    workspace_path = plate_path.parent / f"{plate_path.name}_workspace"
    processed_dir = workspace_path.parent / f"{workspace_path.name}_processed"
    stitched_dir = workspace_path.parent / f"{workspace_path.name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created
    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched files created"

    # Print thread activity report
    print_thread_activity_report()

def test_preprocessing_functions(flat_plate_dir, base_pipeline_config, thread_tracker):
    """Test processing a flat plate with preprocessing functions."""
    # Create pipeline configuration based on the base config

    funcs = [normalize]
    config = create_config(
        base_pipeline_config,
        reference_processing={
            "1": funcs
        },
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(flat_plate_dir)

    assert success, "Processing with preprocessing functions failed"

    # Check if output directories were created
    # Use the plate path to check for output directories
    plate_path = Path(flat_plate_dir)
    workspace_path = plate_path.parent / f"{plate_path.name}_workspace"
    processed_dir = workspace_path.parent / f"{workspace_path.name}_processed"
    post_processed_dir = workspace_path.parent / f"{workspace_path.name}_post_processed"
    stitched_dir = workspace_path.parent / f"{workspace_path.name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert post_processed_dir.exists(), "Post-processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if processed files were created
    processed_files = find_image_files(processed_dir)
    post_processed_files = find_image_files(post_processed_dir)

    assert len(processed_files) > 0, "No processed files created"
    assert len(post_processed_files) > 0, "No post-processed files created"

    # Print thread activity report
    print_thread_activity_report()

def test_all_channels_stitched(flat_plate_dir, base_pipeline_config, thread_tracker):
    """Test that all available channels are stitched by default."""
    # Use the base configuration which already has reference_channels=["1"]
    config = base_pipeline_config

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(flat_plate_dir)

    assert success, "Processing with all channels failed"

    # Check if output directories were created
    # Use the plate path to check for output directories
    plate_path = Path(flat_plate_dir)
    workspace_path = plate_path.parent / f"{plate_path.name}_workspace"
    stitched_dir = workspace_path.parent / f"{workspace_path.name}_stitched"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if stitched files were created
    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched files created"

    # Print thread activity report
    print_thread_activity_report()

def calcein_process(stack):
    """Apply tophat filter to Calcein images."""
    return [ImagePreprocessor.tophat(img) for img in stack]

def dapi_process(stack):
    """Apply tophat filter to DAPI images."""
    stack = ImagePreprocessor.stack_percentile_normalize(stack,low_percentile=0.1,high_percentile=99.9)
    return [ImagePreprocessor.tophat(img) for img in stack]

def test_mixed_preprocessing_functions(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test that both single-image and stack-processing functions can be used."""
    # Create pipeline configuration based on the base config
    config = create_config(
        base_pipeline_config,
        reference_channels=["1", "2"],
        # Channel 1 uses a single-image function
        # Channel 2 uses a stack-processing function
        reference_processing={
            "1": calcein_process,
            "2": dapi_process,
        },
        reference_flatten="max_projection"
    )
    # Commented out: config.stitch_flatten = "max_projection"

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    success = pipeline.run(zstack_plate_dir)

    assert success, "Processing with mixed preprocessing functions failed"

    # Check if output directories were created
    # Use the plate path to check for output directories
    plate_path = Path(zstack_plate_dir)
    workspace_path = plate_path.parent / f"{plate_path.name}_workspace"
    processed_dir = workspace_path.parent / f"{workspace_path.name}_processed"
    post_processed_dir = workspace_path.parent / f"{workspace_path.name}_post_processed"
    stitched_dir = workspace_path.parent / f"{workspace_path.name}_stitched"

    assert processed_dir.exists(), "Processed directory not created"
    assert post_processed_dir.exists(), "Post-processed directory not created"
    assert stitched_dir.exists(), "Stitched directory not created"

    # Check if processed files were created for both channels
    processed_files = find_image_files(processed_dir)
    post_processed_files = find_image_files(post_processed_dir)
    stitched_files = find_image_files(stitched_dir)

    assert len(processed_files) > 0, "No processed files created"
    assert len(post_processed_files) > 0, "No post-processed files created"
    assert len(stitched_files) > 0, "No stitched files created"

    # Print thread activity report
    print_thread_activity_report()