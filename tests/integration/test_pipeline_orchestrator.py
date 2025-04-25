import shutil
import pytest
from pathlib import Path
import numpy as np
from typing import List, Union

from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.config import StitcherConfig, PipelineConfig
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.image_processor import ImageProcessor as IP
from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator
from ezstitcher.core.image_locator import ImageLocator
from ezstitcher.core.file_system_manager import FileSystemManager as fs_manager
from ezstitcher.core.utils import stack



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
    #"wells": ['A01', 'A02', 'B01', 'B02', 'C01', 'C02', 'D01', 'D02']
    "wells": ['A01', 'D02']
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

def create_synthetic_plate_data(test_function_dir, microscope_config, test_params, plate_name, z_stack_levels):
    """Create synthetic plate data for the specified microscope type.

    Args:
        test_function_dir: Directory for test function
        microscope_config: Microscope configuration
        test_params: Test parameters
        plate_name: Name of the plate directory
        z_stack_levels: Number of Z-stack levels

    Returns:
        Path to the plate directory
    """
    plate_dir = test_function_dir / plate_name

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
        z_stack_levels=z_stack_levels,
        cell_size_range=cell_size_range,
        wells=wells,
        format=microscope_config["format"],
        auto_image_size=microscope_config["auto_image_size"]
    )
    generator.generate_dataset()

    # No longer creating a copy of the original data
    # This helps keep the test directories cleaner

    # Always return the plate directory - let the core library handle the directory structure
    return plate_dir


@pytest.fixture
def flat_plate_dir(test_function_dir, microscope_config, test_params):
    """Create synthetic flat plate data for the specified microscope type."""
    return create_synthetic_plate_data(
        test_function_dir=test_function_dir,
        microscope_config=microscope_config,
        test_params=test_params,
        plate_name="flat_plate",
        z_stack_levels=1  # Flat plate has only 1 Z-level
    )


@pytest.fixture
def zstack_plate_dir(test_function_dir, microscope_config, test_params):
    """Create synthetic Z-stack plate data for the specified microscope type."""
    return create_synthetic_plate_data(
        test_function_dir=test_function_dir,
        microscope_config=microscope_config,
        test_params=test_params,
        plate_name="zstack_plate",
        z_stack_levels=5  # Z-stack plate has 5 Z-levels
    )


# Import thread tracking utilities
from ezstitcher.core.utils import track_thread_activity, clear_thread_activity, print_thread_activity_report

# Create an instance of ImageProcessor for testing

# Define a wrapper function for stack_equalize_histogram
def normalize(stack):
    """Apply true histogram equalization to an entire stack."""
    return IP.stack_percentile_normalize(stack,low_percentile=0.1, high_percentile=99.99)

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
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        ),
        num_workers=1,
    )
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

    # Override with new values
    for key, value in kwargs.items():
        config_dict[key] = value

    # Create a new config object
    return PipelineConfig(**config_dict)


def calcein_process(stack):
    """Apply tophat filter to Calcein images."""
    return [IP.tophat(img) for img in stack]

def dapi_process(stack):
    """Apply tophat filter to DAPI images."""
    stack = IP.stack_percentile_normalize(stack, low_percentile=0.1, high_percentile=99.9)
    return [IP.tophat(img) for img in stack]



def test_pipeline_architecture(flat_plate_dir, base_pipeline_config, thread_tracker):
    """
    Test the pipeline architecture with the orchestrator's built-in multithreaded run method.

    This test demonstrates how to:
    1. Create pipelines for the orchestrator
    2. Use the orchestrator's built-in multithreaded run method
    3. Process multiple wells in parallel
    """
    # The orchestrator will set up the directories and wells when run is called

    config = base_pipeline_config

    orchestrator = PipelineOrchestrator(config=base_pipeline_config,plate_path=flat_plate_dir)

    # Create position generation pipeline with reference steps
    position_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks
            Step(name="Z-Stack Flattening",
                 func=(IP.create_projection, {'method': 'max_projection'}),
                 variable_components=['z_index'],
                 input_dir=orchestrator.workspace_path),

            # Step 2: Process channels with a sequence of functions and their parameters
            Step(name="Image Enhancement Processing",
                 func=[
                     (stack(IP.sharpen), {'amount': 1.5}),
                     (IP.stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5}),
                     IP.stack_equalize_histogram  # No parameters needed
                 ],
            ),

            Step(func=(IP.create_composite, {'weights': [0.7, 0.3]}),
                 variable_components=['channel']),

            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Create image assembly pipeline
    assembly_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks with best focus
            Step(name="Z-Stack Flattening",
                 func=(IP.create_projection, {'method': 'max_projection'}),
                 variable_components=['z_index'],
                 input_dir=orchestrator.workspace_path
                 ),

            # Step 2: Process channels
            Step(name="Channel Processing",
                 func=IP.stack_percentile_normalize,
            ),

            ImageStitchingStep()
        ],
        name="Image Assembly Pipeline"
    )

    # Create a list of pipelines to run
    pipelines = [position_pipeline, assembly_pipeline]
    # Run the orchestrator with the pipelines
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"
    print_thread_activity_report()

def test_zstack_pipeline_architecture_focus(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """
    Test the pipeline architecture with the orchestrator's built-in multithreaded run method.

    This test demonstrates how to:
    1. Create pipelines for the orchestrator
    2. Use the orchestrator's built-in multithreaded run method
    3. Process multiple wells in parallel
    """
    # The orchestrator will set up the directories and wells when run is called
    config = base_pipeline_config

    orchestrator = PipelineOrchestrator(config=base_pipeline_config,plate_path=zstack_plate_dir)

    # Create focus directory
    focus_dir = orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_focus"

    # Create position generation pipeline with reference steps
    position_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks
            Step(name="Z-Stack Flattening",
                 func=(IP.create_projection, {'method': 'max_projection'}),
                 variable_components=['z_index'],
                 input_dir=orchestrator.workspace_path),

            # Step 2: Process channels
            Step(name="Feature Enhancement",
                 func=stack(IP.sharpen)),

            Step(func=IP.create_composite,
                 variable_components=['channel']),

            # Step 3: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    #Get best focus
    assembly_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks with best focus
            Step(name="cleaning",
                 func=[stack(IP.tophat)],  # Use stack() for single-image functions
                 input_dir=orchestrator.workspace_path,
                 output_dir=focus_dir),

            # Step 2: Stitch images
            Step(name="Focus",
                 func=(IP.create_projection, {'method': 'best_focus'}),
                 variable_components=['z_index']),

            ImageStitchingStep()
        ],
        name="Focused Image Assembly Pipeline"
    )

    pipelines = [position_pipeline, assembly_pipeline]
    # Run the orchestrator with the pipelines
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"
    print_thread_activity_report()

def test_zstack_pipeline_architecture(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """
    Test the pipeline architecture with the orchestrator's built-in multithreaded run method.

    This test demonstrates how to:
    1. Create pipelines for the orchestrator
    2. Use the orchestrator's built-in multithreaded run method
    3. Process multiple wells in parallel
    """
    # The orchestrator will set up the directories and wells when run is called
    config = base_pipeline_config

    orchestrator = PipelineOrchestrator(config=base_pipeline_config,plate_path=zstack_plate_dir)




    # Create position generation pipeline with reference steps
    position_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks
            Step(name="Z-Stack Flattening",
                 func=(IP.create_projection, {'method': 'max_projection'}),
                 variable_components=['z_index'],
                 input_dir=orchestrator.workspace_path),

            # Step 2: Process channels
            Step(name="Channel Processing",
                 func=IP.stack_percentile_normalize,
                 variable_components=['channel']),

            # Step 3: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Create image assembly pipeline
    assembly_pipeline = Pipeline(
        steps=[
            # Step 1: Clean final images with channel-specific processing
            Step(name="Channel-specific cleaning",
                 func={
                     # DAPI channel with larger footprint
                     "1": (stack(IP.tophat), {'footprint_size': 5}),
                     # GFP channel with smaller footprint
                     "2": (stack(IP.tophat), {'footprint_size': 3})
                 },
                 group_by='channel',
                 input_dir=orchestrator.workspace_path),

            # Step 2: Stitch images
            ImageStitchingStep()
        ],
        name="Image Assembly Pipeline"
    )

    # Create a list of pipelines to run
    pipelines = [position_pipeline, assembly_pipeline]
    # Run the orchestrator with the pipelines
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"
    print_thread_activity_report()

def test_minimal_pipeline_with_defaults(flat_plate_dir, base_pipeline_config, thread_tracker):
    """
    Test a minimal pipeline that only defines input directory and handles processing,
    position generation, and stitching in one go using defaults.

    This test verifies that:
    1. A pipeline can be created with minimal configuration
    2. ImageStitchingStep correctly uses the pipeline's input directory by default
    3. The entire workflow (processing, position generation, stitching) works with defaults
    """
    # Set up the orchestrator
    config = base_pipeline_config
    orchestrator = PipelineOrchestrator(config=config, plate_path=flat_plate_dir)

    # Set up directories
    #dirs = setup_directories(orchestratorn.workspace_path, orchestrator.input_dir)

    # Create a single all-in-one pipeline that does everything with absolute minimal configuration
    # Only defining the input directory - everything else should be handled automatically
    all_in_one_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,
        # No output_dir defined - should be handled automatically
        steps=[
            # Step 1: Basic image processing
            Step(
                name="Basic Processing",
                func=IP.stack_percentile_normalize
            ),

            PositionGenerationStep(),

            ImageStitchingStep(
                input_dir=orchestrator.workspace_path
            )
        ],
        name="Absolute Minimal Pipeline"
    )

    # Run the pipeline
    success = orchestrator.run(pipelines=[all_in_one_pipeline])
    assert success, "Minimal pipeline execution failed"

    # Since we didn't specify an output directory, we need to find where the images were saved
    # They should be in a directory with 'stitched' in the name
    workspace_parent = orchestrator.workspace_path.parent
    stitched_dir = None

    # Look for directories with 'stitched' in the name
    for path in workspace_parent.glob("*stitched*"):
        if path.is_dir():
            stitched_dir = path
            break

    assert stitched_dir is not None, "Could not find stitched images directory"
    print(f"Found stitched images directory: {stitched_dir}")

    # Verify that stitched images were created
    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched images were created"

    print(f"Successfully created {len(stitched_files)} stitched images")
    print("Using absolute minimal pipeline configuration with defaults")
    print_thread_activity_report()