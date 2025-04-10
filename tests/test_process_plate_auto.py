import os
from pathlib import Path
import numpy as np
import tifffile
import pytest
from ezstitcher.core.main import process_plate_auto, apply_nested_overrides
from ezstitcher.core.config import PlateProcessorConfig

def create_synthetic_plate(output_dir: Path, z_stack_levels=1, z_step_size=2.0):
    """
    Create synthetic ImageXpress-style plate data with optional Z-stack levels.
    """
    wells = ["A01"]
    sites = [1, 2]
    channels = [1, 2]

    timepoint_dir = output_dir / "TimePoint_1"
    timepoint_dir.mkdir(parents=True, exist_ok=True)

    for well in wells:
        for site in sites:
            for channel in channels:
                for z in range(z_stack_levels):
                    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
                    filename = f"{well}_s{site:03d}_w{channel}_z{z:03d}.tif"
                    tifffile.imwrite(timepoint_dir / filename, img)

def test_process_plate_auto_flat_imx(tmp_path):
    """
    Test process_plate_auto() on synthetic flat (non-Z-stack) plate.
    """
    plate_dir = tmp_path / "flat_plate"
    create_synthetic_plate(plate_dir, z_stack_levels=1)
    
    success = process_plate_auto(plate_dir)
    assert success, "Flat plate processing with process_plate_auto() failed"

    # Check output directory exists
    stitched_dir = plate_dir.parent / f"{plate_dir.name}_stitched"
    assert stitched_dir.exists(), "Stitched output directory not found"

def test_process_plate_auto_zstack_imx(tmp_path):
    """
    Test process_plate_auto() on synthetic Z-stack plate.
    """
    plate_dir = tmp_path / "zstack_plate"
    create_synthetic_plate(plate_dir, z_stack_levels=5)

    success = process_plate_auto(plate_dir)
    assert success, "Z-stack plate processing with process_plate_auto() failed"

    # Check output directory exists
    stitched_dir = plate_dir.parent / f"{plate_dir.name}_stitched"
    assert stitched_dir.exists(), "Stitched output directory not found"

def test_process_plate_auto_flat_opera(tmp_path):
    """
    Test process_plate_auto() on synthetic flat (non-Z-stack) plate.
    """
    plate_dir = tmp_path / "flat_plate"
    create_synthetic_plate(plate_dir, z_stack_levels=1)

    success = process_plate_auto(plate_dir)
    assert success, "Flat plate processing with process_plate_auto() failed"

    # Check output directory exists
    stitched_dir = plate_dir.parent / f"{plate_dir.name}_stitched"
    assert stitched_dir.exists(), "Stitched output directory not found"

def test_process_plate_auto_zstack_opera(tmp_path):
    """
    Test process_plate_auto() on synthetic Z-stack plate.
    """
    plate_dir = tmp_path / "zstack_plate"
    create_synthetic_plate(plate_dir, z_stack_levels=5)

    success = process_plate_auto(plate_dir)
    assert success, "Z-stack plate processing with process_plate_auto() failed"

    # Check output directory exists
def test_apply_nested_overrides():
    config = PlateProcessorConfig()
    overrides = {
        "stitcher.tile_overlap": 42,
        "focus_analyzer.method": "max_intensity"
    }
    apply_nested_overrides(config, overrides)
    assert config.stitcher.tile_overlap == 42, "Nested override for stitcher.tile_overlap failed"
    assert config.focus_analyzer.method == "max_intensity", "Nested override for focus_analyzer.method failed"
    stitched_dir = plate_dir.parent / f"{plate_dir.name}_stitched"
    assert stitched_dir.exists(), "Stitched output directory not found"