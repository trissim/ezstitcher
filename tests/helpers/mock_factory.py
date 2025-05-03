"""
Mock factory utilities for EZStitcher tests.

This module provides standardized functions for creating mock objects
that properly mimic the behavior of real EZStitcher components.
"""

from unittest.mock import MagicMock, create_autospec
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from ezstitcher.core.config import PipelineConfig, StitcherConfig
from ezstitcher.core.pipeline import Pipeline, ProcessingContext
from ezstitcher.core.pipeline_factories import AutoPipelineFactory
from ezstitcher.io.filemanager import FileManager
from ezstitcher.io.storage_backend import MicroscopyStorageBackend, DiskStorageBackend


def create_mock_stitcher_config(
    tile_overlap: float = 10.0,
    max_shift: int = 50,
    margin_ratio: float = 0.1,
    pixel_size: float = 1.0
) -> MagicMock:
    """
    Create a mock StitcherConfig with all required attributes.

    Args:
        tile_overlap: Tile overlap percentage
        max_shift: Maximum shift in pixels
        margin_ratio: Margin ratio
        pixel_size: Pixel size

    Returns:
        MagicMock: Configured with StitcherConfig attributes
    """
    mock = MagicMock(spec=StitcherConfig)
    mock.tile_overlap = tile_overlap
    mock.max_shift = max_shift
    mock.margin_ratio = margin_ratio
    mock.pixel_size = pixel_size
    return mock


def create_mock_pipeline_config(
    out_dir_suffix: str = "_out",
    positions_dir_suffix: str = "_positions",
    stitched_dir_suffix: str = "_stitched",
    num_workers: int = 1,
    well_filter: Optional[List[str]] = None,
    stitcher: Optional[MagicMock] = None,
    force_parser: Optional[str] = None,
    storage_mode: str = "legacy",
    storage_root: Optional[Path] = None,
    output_dir: Optional[Path] = Path("output")
) -> MagicMock:
    """
    Create a mock PipelineConfig with all required attributes.

    Args:
        out_dir_suffix: Suffix for output directories
        positions_dir_suffix: Suffix for positions directories
        stitched_dir_suffix: Suffix for stitched directories
        num_workers: Number of worker threads
        well_filter: List of wells to process
        stitcher: Mock StitcherConfig (created if None)
        force_parser: Force a specific parser type
        storage_mode: Storage mode ('legacy', 'memory', 'zarr')
        storage_root: Root directory for storage
        output_dir: Output directory (commonly used in tests)

    Returns:
        MagicMock: Configured with PipelineConfig attributes
    """
    mock = MagicMock(spec=PipelineConfig)
    mock.out_dir_suffix = out_dir_suffix
    mock.positions_dir_suffix = positions_dir_suffix
    mock.stitched_dir_suffix = stitched_dir_suffix
    mock.num_workers = num_workers
    mock.well_filter = well_filter
    mock.stitcher = stitcher or create_mock_stitcher_config()
    mock.force_parser = force_parser
    mock.storage_mode = storage_mode
    mock.storage_root = storage_root
    mock.output_dir = output_dir
    return mock


def create_mock_storage_backend(
    backend_name: str = "DiskStorageBackend"
) -> MagicMock:
    """
    Create a mock storage backend with proper type name.

    Args:
        backend_name: Name to use for the backend's type.__name__

    Returns:
        MagicMock: Configured with storage backend attributes
    """
    mock_backend = MagicMock(spec=MicroscopyStorageBackend)
    type(mock_backend).__name__ = backend_name
    return mock_backend


def create_mock_file_manager(
    root_dir: Union[str, Path] = "/mock/root",
    backend: Optional[MagicMock] = None,
    backend_name: str = "DiskStorageBackend"
) -> MagicMock:
    """
    Create a mock FileManager with all required attributes and methods.

    Args:
        root_dir: Root directory for file operations
        backend: Mock backend (created if None)
        backend_name: Name to use for the backend's type.__name__ if backend is None

    Returns:
        MagicMock: Configured with FileManager attributes and methods
    """
    mock_fm = MagicMock(spec=FileManager)
    mock_fm.root_dir = Path(root_dir)
    mock_fm.backend = backend or create_mock_storage_backend(backend_name)
    
    # Configure common method behaviors
    mock_fm.ensure_directory.side_effect = lambda p: Path(p)
    mock_fm.find_image_directory.side_effect = lambda p, extensions=None: Path(p) / "Images"
    mock_fm.list_image_files.return_value = [Path("dummy/plate/Images/Well_A1_Site_1.tif")]
    
    return mock_fm


def create_mock_pipeline(
    name: str = "MockPipeline",
    steps: Optional[List[MagicMock]] = None,
    path_overrides: Optional[Dict[str, Union[str, Path]]] = None
) -> MagicMock:
    """
    Create a mock Pipeline with all required attributes and methods.

    Args:
        name: Pipeline name
        steps: List of mock steps
        path_overrides: Dictionary of path overrides

    Returns:
        MagicMock: Configured with Pipeline attributes and methods
    """
    mock = MagicMock(spec=Pipeline)
    mock.name = name
    mock.steps = steps or []
    mock.path_overrides = path_overrides or {}
    return mock


def create_mock_pipeline_factory(
    pipelines: Optional[List[MagicMock]] = None
) -> MagicMock:
    """
    Create a mock AutoPipelineFactory that returns the specified pipelines.

    Args:
        pipelines: List of mock pipelines to return from create_pipelines()

    Returns:
        MagicMock: Mock factory class with configured instance
    """
    mock_factory = MagicMock(spec=AutoPipelineFactory)
    
    # Create default pipelines if none provided
    if pipelines is None:
        mock_pos_pipeline = create_mock_pipeline(name="Position Generation Pipeline")
        mock_assembly_pipeline = create_mock_pipeline(name="Image Assembly Pipeline")
        pipelines = [mock_pos_pipeline, mock_assembly_pipeline]
    
    # Configure the factory instance
    mock_factory_instance = mock_factory.return_value
    mock_factory_instance.create_pipelines.return_value = pipelines
    
    return mock_factory


def create_mock_processing_context(
    well_id: str = "A1",
    config: Optional[MagicMock] = None,
    orchestrator: Optional[MagicMock] = None,
    file_manager: Optional[MagicMock] = None,
    storage_adapter: Optional[MagicMock] = None,
    handler: Optional[MagicMock] = None,
    results: Optional[Dict[Any, Any]] = None
) -> MagicMock:
    """
    Create a mock ProcessingContext with all required attributes.

    Args:
        well_id: Well ID
        config: Mock PipelineConfig
        orchestrator: Mock PipelineOrchestrator
        file_manager: Mock FileManager
        storage_adapter: Mock StorageAdapter
        handler: Mock MicroscopeHandler
        results: Dictionary of results

    Returns:
        MagicMock: Configured with ProcessingContext attributes
    """
    mock = MagicMock(spec=ProcessingContext)
    mock.well_id = well_id
    mock.config = config or create_mock_pipeline_config()
    mock.orchestrator = orchestrator
    mock.file_manager = file_manager
    mock.storage_adapter = storage_adapter
    mock.handler = handler
    mock.results = results or {}
    
    # Configure get_step_input_dir and get_step_output_dir methods
    mock.get_step_input_dir.side_effect = lambda step: Path(f"input_dir_for_{step.name}")
    mock.get_step_output_dir.side_effect = lambda step: Path(f"output_dir_for_{step.name}")
    
    return mock


def configure_mock_for_well_detection(
    mock_file_manager: MagicMock,
    mock_microscope_handler: MagicMock,
    wells: List[str] = ["A1", "B2", "C3"]
) -> None:
    """
    Configure mocks for well detection.

    Args:
        mock_file_manager: Mock FileManager
        mock_microscope_handler: Mock MicroscopeHandler
        wells: List of wells to detect
    """
    # Configure file_manager to return image files
    mock_file_manager.list_image_files.return_value = [
        Path(f"dummy/plate/Images/Well_{well}_Site_1.tif") for well in wells
    ]
    
    # Configure microscope_handler to parse filenames
    mock_microscope_handler.parse_filename.side_effect = lambda f: {'well': Path(f).stem.split('_')[1]}


def configure_mock_for_path_resolution(
    mock_context: MagicMock,
    input_dir_pattern: str = "input_dir_for_{step_name}",
    output_dir_pattern: str = "output_dir_for_{step_name}"
) -> None:
    """
    Configure a mock ProcessingContext for path resolution.

    Args:
        mock_context: Mock ProcessingContext
        input_dir_pattern: Pattern for input directory paths
        output_dir_pattern: Pattern for output directory paths
    """
    mock_context.get_step_input_dir.side_effect = lambda step: Path(input_dir_pattern.format(step_name=step.name))
    mock_context.get_step_output_dir.side_effect = lambda step: Path(output_dir_pattern.format(step_name=step.name))
