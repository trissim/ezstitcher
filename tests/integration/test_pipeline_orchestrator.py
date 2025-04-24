import shutil
import pytest
from pathlib import Path
import numpy as np
from typing import List, Union

from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.config import StitcherConfig, PipelineConfig
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
from ezstitcher.core.image_preprocessor import ImagePreprocessor as IP
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

    # Create a copy of the original data for inspection
    original_dir = test_function_dir / f"{plate_name}_original"
    if not original_dir.exists():
        shutil.copytree(plate_dir, original_dir)

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


# Import the ImagePreprocessor for stack functions
from ezstitcher.core.utils import track_thread_activity, clear_thread_activity, print_thread_activity_report

# Create an instance of ImagePreprocessor for testing

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
        reference_channels=["1"],
        cleanup_processed=False,
        cleanup_post_processed=False,
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        ),
        num_workers=1,
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

def setup_directories(workspace_dir, input_dir):
    """
    Set up directory structure for processing.

    Args:
        plate_path: Path to the plate folder
        input_dir: Path to the input directory

    Returns:
        dict: Dictionary of directories
    """
    # Create main directories

    workspace = workspace_dir
    processed = workspace.parent / f"{workspace.name}_processed"
    post_processed = workspace.parent / f"{workspace.name}_post_processed"
    positions = workspace.parent / f"{workspace.name}_positions"
    stitched = workspace.parent / f"{workspace.name}_stitched"
    focus = workspace.parent / f"{workspace.name}_best_focus"

    dirs = {
        'input': input_dir,
        'workspace': workspace,
        'processed': processed,
        'post_processed': post_processed,
        'positions': positions,
        'stitched': stitched,
        'focus': focus
    }

    # Ensure main directories exist
#    for dir_path in dirs.values():
#        fs_manager.ensure_directory(dir_path)

    return dirs


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


    dirs = setup_directories(orchestrator.workspace_path, orchestrator.input_dir)


    # Create position generation pipeline with reference steps
    position_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks
            Step(name="Z-Stack Flattening",
                 func=IP.create_projection,
                 variable_components=['z_index'],
                 processing_args={'method': 'max_projection'},
                 input_dir=dirs['input'],  
                 output_dir=dirs['processed']),  

            # Step 2: Process channels
            Step(name="Image Enhancement Processing",
                 #func=IP.stack_percentile_normalize,
                 func=[stack(IP.sharpen),
                      IP.stack_percentile_normalize,
                       IP.stack_equalize_histogram],
            ),

            # Step 3: Create composites
            Step(name="Composite Creation",
                 func=IP.create_composite,
                 variable_components=['channel']),

            # Step 4: Generate positions
            PositionGenerationStep(
                name="Generate Positions",
            )
        ],
        name="Position Generation Pipeline"
    )

    # Create image assembly pipeline
    assembly_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks with best focus
            Step(name="Z-Stack Flattening",
                 func=IP.create_projection,
                 variable_components=['z_index'],
                 processing_args={'method': 'max_projection'},
                 input_dir=dirs['input'],
                 output_dir=dirs['post_processed']
                 ),

            # Step 2: Process channels
            Step(name="Channel Processing",
                 func=IP.stack_percentile_normalize,
            ),

            # Step 3: Stitch images
            ImageStitchingStep(
                name="Stitch Images",
            )
        ],
        name="Image Assembly Pipeline"
    )

    # Create a list of pipelines to run
    pipelines = [position_pipeline, assembly_pipeline]
    # Run the orchestrator with the pipelines
    success = orchestrator.run(pipelines=pipelines)
    assert success, f"{test_name} failed"
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


    dirs = setup_directories(orchestrator.workspace_path, orchestrator.input_dir)


    # Create position generation pipeline with reference steps
    position_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks
            Step(name="Z-Stack Flattening",
                 func=IP.create_projection,
                 variable_components=['z_index'],
                 processing_args={'method': 'max_projection'},
                 input_dir=dirs['input'],  
                 output_dir=dirs['processed']),  

            # Step 2: Process channels
            Step(name="Feature Enhancement",
                 func=stack(IP.sharpen),
                 variable_components=['site']),

            Step(name="Composite Creation",
                 func=IP.create_composite,
                 variable_components=['site']),

            # Step 3: Generate positions
            PositionGenerationStep(
                name="Generate Positions",
                output_dir=dirs['positions'])
        ],
        name="Position Generation Pipeline"
    )

    #Get best focus
    assembly_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks with best focus
            Step(name="cleaning",
                 func=[IP.tophat],
                 input_dir=dirs['input'],
                 output_dir=dirs['focus']),

            # Step 2: Stitch images
            Step(name="Focus",
                 func=IP.create_projection,
                 variable_components=['z_index'],
                 processing_args={'method': 'best_focus'}),

            ImageStitchingStep(
                name="Stitch Focused Images",
                positions_dir=dirs['positions'],
                output_dir=dirs['stitched']),
        ],
        name="Focused Image Assembly Pipeline"
    )

    pipelines = [position_pipeline, assembly_pipeline]
    # Run the orchestrator with the pipelines
    success = orchestrator.run(pipelines=pipelines)
    assert success, f"{test_name} failed"
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


    dirs = setup_directories(orchestrator.workspace_path, orchestrator.input_dir)


    # Create position generation pipeline with reference steps
    position_pipeline = Pipeline(
        steps=[
            # Step 1: Flatten Z-stacks
            Step(name="Z-Stack Flattening",
                 func=IP.create_projection,
                 variable_components=['z_index'],
                 processing_args={'method': 'max_projection'},
                 input_dir=dirs['input'],  
                 output_dir=dirs['processed']),  

            # Step 2: Process channels
            Step(name="Channel Processing",
                 func=IP.stack_percentile_normalize,
                 variable_components=['channel']),

            # Step 3: Generate positions
            PositionGenerationStep(
                name="Generate Positions",
                output_dir=dirs['positions'])
        ],
        name="Position Generation Pipeline"
    )

    # Create image assembly pipeline
    assembly_pipeline = Pipeline(
        steps=[
            # Step 1: Clean final images
            Step(name="cleaning",
                 func=stack(IP.tophat),
                 input_dir=dirs['input'],
                 output_dir=dirs['post_processed']),

            # Step 2: Stitch images
            ImageStitchingStep(
                name="Stitch Images",
                positions_dir=dirs['positions'],
                output_dir=dirs['stitched']),
        ],
        name="Image Assembly Pipeline"
    )

    # Create a list of pipelines to run
    pipelines = [position_pipeline, assembly_pipeline]
    # Run the orchestrator with the pipelines
    success = orchestrator.run(pipelines=pipelines)
    assert success, f"{test_name} failed"
    print_thread_activity_report()