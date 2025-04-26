"""
Integration tests for pipeline factories.

This module tests the pipeline factory functions with synthetic data.
"""

import pytest
from pathlib import Path
import numpy as np

from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.config import StitcherConfig, PipelineConfig
from ezstitcher.core.pipeline_factories import (
    create_basic_pipeline,
    create_multichannel_pipeline,
    create_zstack_pipeline,
    create_focus_pipeline
)

# Import fixtures from test_pipeline_orchestrator.py if available
# If not available, you'll need to create them or modify these tests
try:
    from tests.integration.test_pipeline_orchestrator import (
        microscope_config, base_test_dir, test_function_dir, test_params,
        flat_plate_dir, zstack_plate_dir, thread_tracker, base_pipeline_config,
        create_synthetic_plate_data, find_image_files
    )
except ImportError:
    # Define minimal fixtures for testing
    import tempfile
    import shutil
    import os

    @pytest.fixture
    def base_test_dir():
        """Create a temporary directory for testing."""
        test_dir = tempfile.mkdtemp()
        yield test_dir
        shutil.rmtree(test_dir)

    @pytest.fixture
    def flat_plate_dir(base_test_dir):
        """Create a synthetic flat plate directory."""
        plate_dir = Path(base_test_dir) / "flat_plate"
        plate_dir.mkdir(parents=True, exist_ok=True)
        # Create minimal structure for testing
        (plate_dir / "A1").mkdir(parents=True, exist_ok=True)
        # Create a dummy image
        with open(plate_dir / "A1" / "image.tif", "wb") as f:
            f.write(b"dummy image")
        yield plate_dir

    @pytest.fixture
    def zstack_plate_dir(base_test_dir):
        """Create a synthetic Z-stack plate directory."""
        plate_dir = Path(base_test_dir) / "zstack_plate"
        plate_dir.mkdir(parents=True, exist_ok=True)
        # Create minimal structure for testing
        (plate_dir / "A1").mkdir(parents=True, exist_ok=True)
        # Create dummy images for Z-stack
        for z in range(3):
            with open(plate_dir / "A1" / f"image_z{z}.tif", "wb") as f:
                f.write(b"dummy image")
        yield plate_dir

    @pytest.fixture
    def base_pipeline_config():
        """Create a basic pipeline configuration."""
        return PipelineConfig()

    @pytest.fixture
    def thread_tracker():
        """Dummy thread tracker."""
        yield None

    def find_image_files(directory):
        """Find image files in a directory."""
        return list(Path(directory).glob("**/*.tif"))


def test_basic_stitching_pipeline(flat_plate_dir, base_pipeline_config, thread_tracker):
    """
    Test basic stitching pipeline with synthetic data.
    """
    # Set up the orchestrator
    config = base_pipeline_config
    orchestrator = PipelineOrchestrator(config=config, plate_path=flat_plate_dir)

    # Create basic stitching pipeline
    pipelines = create_basic_pipeline(
        input_dir=orchestrator.workspace_path
    )

    # Run the orchestrator with the pipelines
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"

    # Verify that stitched images were created
    stitched_dir = orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_stitched"
    assert stitched_dir.exists(), "Stitched directory not found"

    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched images were created"


def test_multichannel_stitching_pipeline(flat_plate_dir, base_pipeline_config, thread_tracker):
    """
    Test multi-channel stitching pipeline with synthetic data.
    """
    # Set up the orchestrator
    config = base_pipeline_config
    orchestrator = PipelineOrchestrator(config=config, plate_path=flat_plate_dir)

    # Create multi-channel stitching pipeline
    pipelines = create_multichannel_pipeline(
        input_dir=orchestrator.workspace_path,
        weights=[0.7, 0.3]
    )

    # Run the orchestrator with the pipelines
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"

    # Verify that stitched images were created
    stitched_dir = orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_stitched"
    assert stitched_dir.exists(), "Stitched directory not found"

    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched images were created"


def test_zstack_stitching_pipeline(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """
    Test Z-stack stitching pipeline with synthetic data.
    """
    # Set up the orchestrator
    config = base_pipeline_config
    orchestrator = PipelineOrchestrator(config=config, plate_path=zstack_plate_dir)

    # Create Z-stack stitching pipeline
    pipelines = create_zstack_pipeline(
        input_dir=orchestrator.workspace_path,
        method="max"
    )

    # Run the orchestrator with the pipelines
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"

    # Verify that stitched images were created
    stitched_dir = orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_stitched"
    assert stitched_dir.exists(), "Stitched directory not found"

    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched images were created"


def test_focus_stitching_pipeline(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """
    Test focus-based stitching pipeline with synthetic data.
    """
    # Set up the orchestrator
    config = base_pipeline_config
    orchestrator = PipelineOrchestrator(config=config, plate_path=zstack_plate_dir)

    # Create Z-stack stitching pipeline with focus method
    pipelines = create_focus_pipeline(
        input_dir=orchestrator.workspace_path,
        metric="laplacian"
    )

    # Run the orchestrator with the pipelines
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"

    # Verify that stitched images were created
    stitched_dir = orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_stitched"
    assert stitched_dir.exists(), "Stitched directory not found"

    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched images were created"
