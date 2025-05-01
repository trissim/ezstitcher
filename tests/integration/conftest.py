"""
Pytest configuration file for integration tests.
"""
import pytest
import shutil
from pathlib import Path
import numpy as np

from ezstitcher.core.config import StitcherConfig, PipelineConfig
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator

# Import fixtures from test_pipeline_orchestrator.py
from tests.integration.test_pipeline_orchestrator import (
    microscope_config,
    base_test_dir,
    test_function_dir,
    test_params,
    flat_plate_dir,
    zstack_plate_dir,
    thread_tracker,
    base_pipeline_config
)

# These fixtures are now available to all integration tests
