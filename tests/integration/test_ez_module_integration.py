"""
Integration tests for EZ module using default behaviors and minimal configuration.

These tests verify that the EZ module correctly wraps the AutoPipelineFactory
and PipelineOrchestrator functionality.
"""

import pytest
from pathlib import Path

from ezstitcher.ez import stitch_plate, EZStitcher

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


def test_2d_plate_stitch_one_liner(flat_plate_dir, base_pipeline_config, thread_tracker):
    """Test basic 2D plate stitching with the one-liner function."""
    # Create output directory path
    output_dir = flat_plate_dir.parent / f"{flat_plate_dir.name}_stitched"

    # Use the one-liner function
    result_path = stitch_plate(
        flat_plate_dir,
        output_path=output_dir,
        normalize=True
    )

    # Check that the output directory exists and matches the returned path
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"
    assert result_path == output_dir, f"Returned path {result_path} does not match expected {output_dir}"

    # Check that output directory exists
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Check that the workspace directory was created
    workspace_dir = flat_plate_dir.parent / f"{flat_plate_dir.name}_workspace"
    assert workspace_dir.exists(), f"Workspace directory {workspace_dir} does not exist"


def test_2d_plate_stitch_class(flat_plate_dir, base_pipeline_config, thread_tracker):
    """Test basic 2D plate stitching with the EZStitcher class."""
    # Create output directory path
    output_dir = flat_plate_dir.parent / f"{flat_plate_dir.name}_ez_stitched"

    # Use the EZStitcher class
    stitcher = EZStitcher(
        flat_plate_dir,
        output_path=output_dir,
        normalize=True
    )

    # Run stitching
    result_path = stitcher.stitch()

    # Check that the output directory exists and matches the returned path
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"
    assert result_path == output_dir, f"Returned path {result_path} does not match expected {output_dir}"

    # Check that output directory exists
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Check that the workspace directory was created
    workspace_dir = flat_plate_dir.parent / f"{flat_plate_dir.name}_workspace"
    assert workspace_dir.exists(), f"Workspace directory {workspace_dir} does not exist"


def test_3d_plate_per_plane_stitch(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test 3D plate stitching with per-plane stitching (no z-flattening)."""
    # Create output directory path
    output_dir = zstack_plate_dir.parent / f"{zstack_plate_dir.name}_per_plane_stitched"

    # Use the EZStitcher class with explicit flatten_z=False
    stitcher = EZStitcher(
        zstack_plate_dir,
        output_path=output_dir,
        normalize=True,
        flatten_z=False  # Explicitly set to false to ensure per-plane stitching
    )

    # Run stitching
    result_path = stitcher.stitch()

    # Check that the output directory exists and matches the returned path
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"
    assert result_path == output_dir, f"Returned path {result_path} does not match expected {output_dir}"

    # Check that output directory exists
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Check that the workspace directory was created
    workspace_dir = zstack_plate_dir.parent / f"{zstack_plate_dir.name}_workspace"
    assert workspace_dir.exists(), f"Workspace directory {workspace_dir} does not exist"


def test_3d_plate_max_projection_stitch(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test 3D plate stitching with max projection."""
    # Create output directory path
    output_dir = zstack_plate_dir.parent / f"{zstack_plate_dir.name}_max_stitched"

    # Use the one-liner function with z-flattening parameters
    result_path = stitch_plate(
        zstack_plate_dir,
        output_path=output_dir,
        normalize=True,
        flatten_z=True,
        z_method="max"
    )

    # Check that the output directory exists and matches the returned path
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"
    assert result_path == output_dir, f"Returned path {result_path} does not match expected {output_dir}"

    # Check that output directory exists
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Check that the workspace directory was created
    workspace_dir = zstack_plate_dir.parent / f"{zstack_plate_dir.name}_workspace"
    assert workspace_dir.exists(), f"Workspace directory {workspace_dir} does not exist"


def test_3d_plate_focus_detection_stitch(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test 3D plate stitching with focus detection using combined metric."""
    # Create output directory path
    output_dir = zstack_plate_dir.parent / f"{zstack_plate_dir.name}_focus_stitched"

    # Use the EZStitcher class with focus detection
    stitcher = EZStitcher(
        zstack_plate_dir,
        output_path=output_dir,
        normalize=True,
        flatten_z=True,
        z_method="combined"  # Use the focus metric directly as z_method
    )

    # Run stitching
    result_path = stitcher.stitch()

    # Check that the output directory exists and matches the returned path
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"
    assert result_path == output_dir, f"Returned path {result_path} does not match expected {output_dir}"

    # Check that output directory exists
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Check that the workspace directory was created
    workspace_dir = zstack_plate_dir.parent / f"{zstack_plate_dir.name}_workspace"
    assert workspace_dir.exists(), f"Workspace directory {workspace_dir} does not exist"


def test_well_filter(flat_plate_dir, base_pipeline_config, thread_tracker):
    """Test stitching with well filter."""
    # Create output directory path
    output_dir = flat_plate_dir.parent / f"{flat_plate_dir.name}_filtered_stitched"

    # Get available wells
    from ezstitcher.ez.utils import detect_wells
    all_wells = detect_wells(flat_plate_dir)

    # Use only the first well if multiple wells are available
    well_filter = [all_wells[0]] if all_wells else None

    # Use the EZStitcher class with well filter
    stitcher = EZStitcher(
        flat_plate_dir,
        output_path=output_dir,
        normalize=True,
        well_filter=well_filter
    )

    # Run stitching
    result_path = stitcher.stitch()

    # Check that the output directory exists and matches the returned path
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"
    assert result_path == output_dir, f"Returned path {result_path} does not match expected {output_dir}"

    # Check that output directory exists
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Check that the workspace directory was created
    workspace_dir = flat_plate_dir.parent / f"{flat_plate_dir.name}_workspace"
    assert workspace_dir.exists(), f"Workspace directory {workspace_dir} does not exist"


def test_auto_detection(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test auto-detection of Z-stacks and channels."""
    # Create output directory path
    output_dir = zstack_plate_dir.parent / f"{zstack_plate_dir.name}_auto_stitched"

    # Use the EZStitcher class with auto-detection (no explicit flatten_z or channel_weights)
    stitcher = EZStitcher(
        zstack_plate_dir,
        output_path=output_dir,
        normalize=True
    )

    # Check that Z-stacks were detected
    assert stitcher.flatten_z is True, "Z-stacks were not auto-detected"

    # Run stitching
    result_path = stitcher.stitch()

    # Check that the output directory exists and matches the returned path
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"
    assert result_path == output_dir, f"Returned path {result_path} does not match expected {output_dir}"

    # Check that output directory exists
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Check that the workspace directory was created
    workspace_dir = zstack_plate_dir.parent / f"{zstack_plate_dir.name}_workspace"
    assert workspace_dir.exists(), f"Workspace directory {workspace_dir} does not exist"


def test_set_options(flat_plate_dir, base_pipeline_config, thread_tracker):
    """Test setting options after initialization."""
    # Create output directory path
    output_dir = flat_plate_dir.parent / f"{flat_plate_dir.name}_options_stitched"

    # Use the EZStitcher class with minimal initialization
    stitcher = EZStitcher(
        flat_plate_dir
    )

    # Set options after initialization
    stitcher.set_options(
        output_path=output_dir,
        normalize=True,
        z_method="max"
    )

    # Run stitching
    result_path = stitcher.stitch()

    # Check that the output directory exists and matches the returned path
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"
    assert result_path == output_dir, f"Returned path {result_path} does not match expected {output_dir}"

    # Check that output directory exists
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Check that the workspace directory was created
    workspace_dir = flat_plate_dir.parent / f"{flat_plate_dir.name}_workspace"
    assert workspace_dir.exists(), f"Workspace directory {workspace_dir} does not exist"
