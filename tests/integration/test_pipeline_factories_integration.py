"""
Integration tests for pipeline factory using default behaviors and minimal configuration.
"""

import pytest
from pathlib import Path

from ezstitcher.core.pipeline_factories import (
    PipelineFactory,
    BasicPipelineFactory,
    MultichannelPipelineFactory,
    ZStackPipelineFactory,
    FocusPipelineFactory
)
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

def test_basic_pipeline(flat_plate_dir, base_pipeline_config, thread_tracker):
    """Test basic pipeline with minimal configuration."""
    orchestrator = PipelineOrchestrator(config=base_pipeline_config, plate_path=flat_plate_dir)
    
    factory = BasicPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True
    )
    pipelines = factory.create_pipelines()
    
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"
    assert (orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_stitched").exists()

def test_multichannel_pipeline(flat_plate_dir, base_pipeline_config, thread_tracker):
    """Test multichannel pipeline with weights."""
    orchestrator = PipelineOrchestrator(config=base_pipeline_config, plate_path=flat_plate_dir)
    
    factory = MultichannelPipelineFactory(
        input_dir=orchestrator.workspace_path,
        weights=[0.7, 0.3]  # Using weights parameter instead of channel_weights
    )
    pipelines = factory.create_pipelines()
    
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"
    assert (orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_stitched").exists()

def test_zstack_pipeline(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test z-stack pipeline with max projection."""
    orchestrator = PipelineOrchestrator(config=base_pipeline_config, plate_path=zstack_plate_dir)
    
    factory = ZStackPipelineFactory(
        input_dir=orchestrator.workspace_path,
        method="projection",  # Using method instead of z_method
        method_options={'method': 'max'}
    )
    pipelines = factory.create_pipelines()
    
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"
    assert (orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_stitched").exists()

def test_focus_pipeline(zstack_plate_dir, base_pipeline_config, thread_tracker):
    """Test focus pipeline with variance metric."""
    orchestrator = PipelineOrchestrator(config=base_pipeline_config, plate_path=zstack_plate_dir)
    
    factory = FocusPipelineFactory(
        input_dir=orchestrator.workspace_path,
        focus_options={'metric': 'variance_of_laplacian'}
    )
    pipelines = factory.create_pipelines()
    
    success = orchestrator.run(pipelines=pipelines)
    assert success, "Pipeline execution failed"
    assert (orchestrator.workspace_path.parent / f"{orchestrator.workspace_path.name}_stitched").exists()
