"""
Integration tests for AutoPipelineFactory using default behaviors and minimal configuration.
"""

import pytest
from pathlib import Path

from ezstitcher.core.pipeline_factories import AutoPipelineFactory
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

# Import fixtures from test_pipeline_orchestrator.py
from tests.integration.test_pipeline_orchestrator import (
    microscope_config,
    base_test_dir,
    test_function_dir,
    test_params,
    flat_plate_dir,
    zstack_plate_dir,
    thread_tracker,
    base_pipeline_config,
    create_synthetic_plate_data,
    find_image_files
)

def test_2d_plate_stitch(flat_plate_dir, base_pipeline_config, thread_tracker):
    """Test basic 2D plate stitching with minimal configuration."""
    orchestrator = PipelineOrchestrator(config=base_pipeline_config, plate_path=flat_plate_dir)

    # Create output directory path for the factory
    output_dir = orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_stitched"

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        output_dir=output_dir,
        normalize=True
    )
    pipelines = factory.create_pipelines()

    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"

    # Check that some output directory ending with _stitched exists
    stitched_dirs = list(orchestrator.workspace_path.parent.glob(f"*_stitched"))
    assert len(stitched_dirs) > 0, f"No stitched output directory found in {orchestrator.workspace_path.parent}"

def test_3d_plate_per_plane_stitch(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test 3D plate stitching with per-plane stitching (no z-flattening)."""
    orchestrator = PipelineOrchestrator(config=base_pipeline_config, plate_path=zstack_plate_dir)

    # Create output directory path for the factory
    output_dir = orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_stitched"

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        output_dir=output_dir,
        normalize=True,
        flatten_z=False  # Explicitly set to false to ensure per-plane stitching
    )
    pipelines = factory.create_pipelines()

    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"

    # Check that some output directory ending with _stitched exists
    stitched_dirs = list(orchestrator.workspace_path.parent.glob(f"*_stitched"))
    assert len(stitched_dirs) > 0, f"No stitched output directory found in {orchestrator.workspace_path.parent}"

def test_3d_plate_max_projection_stitch(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test 3D plate stitching with max projection."""
    orchestrator = PipelineOrchestrator(config=base_pipeline_config, plate_path=zstack_plate_dir)

    # Create output directory path for the factory
    output_dir = orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_stitched"

    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        output_dir=output_dir,
        normalize=True,
        flatten_z=True,
        z_method="max"
    )
    pipelines = factory.create_pipelines()

    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"

    # Check that some output directory ending with _stitched exists
    stitched_dirs = list(orchestrator.workspace_path.parent.glob(f"*_stitched"))
    assert len(stitched_dirs) > 0, f"No stitched output directory found in {orchestrator.workspace_path.parent}"
