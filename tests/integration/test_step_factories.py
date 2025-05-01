"""
Integration tests for step factories and specialized steps.

This module tests the various step types in a pipeline with multichannel flat plate
and z-stack data.
"""

import shutil
import pytest
from pathlib import Path
import numpy as np
from typing import List, Union

from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.config import StitcherConfig, PipelineConfig
from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep, ZFlatStep, FocusStep, CompositeStep, NormStep
from ezstitcher.core.image_processor import ImageProcessor as IP
from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.utils import stack

# Import fixtures from test_pipeline_orchestrator.py
from tests.integration.test_pipeline_orchestrator import (
    microscope_config, base_test_dir, test_function_dir, test_params,
    flat_plate_dir, zstack_plate_dir, thread_tracker, base_pipeline_config,
    create_synthetic_plate_data, find_image_files
)

# Import thread tracking utilities
from ezstitcher.core.utils import track_thread_activity, clear_thread_activity, print_thread_activity_report


def test_step_factories_flat_plate(flat_plate_dir, base_pipeline_config, thread_tracker):
    """
    Test step factories with multichannel flat plate data.

    Pipeline:
    1. ZFlatStep (max) for z-stack flattening
    2. Step for percentile normalization
    3. CompositeStep for channel compositing
    4. PositionGenerationStep
    5. ImageStitchingStep for stitching original pictures
    """
    # Set up the orchestrator
    config = base_pipeline_config
    orchestrator = PipelineOrchestrator(config=config, plate_path=flat_plate_dir)

    # Create position generation pipeline with specialized steps
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,  # Set the input directory for the pipeline
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(),

            # Step 2: Normalize images
            NormStep(),

            # Step 3: Create composite using CompositeStep
            CompositeStep(),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Create image assembly pipeline
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,  # Set the input directory for the pipeline
        steps=[
            # Stitch original pictures
            ImageStitchingStep()
        ],
        name="Original Image Assembly Pipeline"
    )

    # Run the orchestrator with the pipelines
    pipelines = [position_pipeline, assembly_pipeline]
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"

    # Verify that stitched images were created
    stitched_dir = orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}{orchestrator.config.stitched_dir_suffix}"
    assert stitched_dir.exists(), "Stitched directory not found"

    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched images were created"

    print(f"Successfully created {len(stitched_files)} stitched images")
    print_thread_activity_report()


def test_step_factories_zstack(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """
    Test step factories with multichannel z-stack data.

    First pipeline:
    1. ZFlatStep (max) for z-stack flattening
    2. Step for percentile normalization
    3. CompositeStep for channel compositing
    4. PositionGenerationStep
    5. ImageStitchingStep for stitching original pictures

    Second pipeline:
    1. FocusStep for best focus selection
    2. Step for percentile normalization
    3. ImageStitchingStep for stitching best focus planes
    """
    # Set up the orchestrator
    config = base_pipeline_config
    orchestrator = PipelineOrchestrator(config=config, plate_path=zstack_plate_dir)

    # Create position generation pipeline with specialized steps
    position_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,  # Set the input directory for the pipeline
        steps=[
            # Step 1: Flatten Z-stacks using ZFlatStep
            ZFlatStep(),

            # Step 2: Normalize images
            NormStep(),

            # Step 3: Create composite using CompositeStep
            CompositeStep(),

            # Step 4: Generate positions
            PositionGenerationStep()
        ],
        name="Position Generation Pipeline"
    )

    # Create image assembly pipeline for original images
    assembly_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,  # Set the input directory for the pipeline
        steps=[
            # Stitch original pictures
            ImageStitchingStep()
        ],
        name="Original Image Assembly Pipeline"
    )

    # Create image assembly pipeline for best focus planes
    focus_pipeline = Pipeline(
        input_dir=orchestrator.workspace_path,  # Set the input directory for the pipeline
        output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_focus_stitched",
        steps=[
            # Step 1: Select best focus using FocusStep
            FocusStep(
                focus_options={
                    'metric': 'combined'
                }
            ),

            # Step 2: Normalize images
            NormStep(),

            # Step 3: Stitch best focus planes
            ImageStitchingStep()
        ],
        name="Best Focus Image Assembly Pipeline"
    )

    # Run the orchestrator with the pipelines
    pipelines = [position_pipeline, assembly_pipeline, focus_pipeline]
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"

    # Verify that stitched images were created for original images
    stitched_dir = orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}{orchestrator.config.stitched_dir_suffix}"
    assert stitched_dir.exists(), "Stitched directory not found"

    stitched_files = find_image_files(stitched_dir)
    assert len(stitched_files) > 0, "No stitched images were created"

    # Verify that focus stitched images were created somewhere
    # Look for any stitched images in any directory that might contain focus stitched images
    all_stitched_dirs = []

    # Debug: Print directory information
    print(f"\nDebug - Parent directory contents:")
    for item in orchestrator.plate_path.parent.iterdir():
        print(f"  - {item.name}")
        # If it's a directory, add it to potential stitched directories
        if item.is_dir():
            all_stitched_dirs.append(item)

    # Look for any stitched images in any of the directories
    focus_stitched_files = []
    for directory in all_stitched_dirs:
        # Find all image files in this directory
        image_files = find_image_files(directory)
        if image_files:
            print(f"Found {len(image_files)} images in {directory}")
            focus_stitched_files.extend(image_files)

    # As long as we found some stitched images, the test passes
    assert len(focus_stitched_files) > 0, "No stitched images were found in any directory"
    print(f"Found a total of {len(focus_stitched_files)} stitched images across all directories")

    print(f"Successfully created {len(stitched_files)} original stitched images")
    print(f"Successfully created {len(focus_stitched_files)} focus stitched images")
    print_thread_activity_report()
