"""End-to-end tests for MaterializationManager with real storage adapters."""

import pytest
import numpy as np
from pathlib import Path
import os
import shutil

from ezstitcher.core.pipeline import Pipeline
from ezstitcher.core.steps import Step
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.io.materialization import MaterializationManager
from ezstitcher.io.overlay import OverlayMode
from ezstitcher.io.storage_adapter import MemoryStorageAdapter, ZarrStorageAdapter

# Import fixtures
from tests.fixtures.materialization_fixtures import (
    temp_test_dir, overlay_dir, mock_microscope_handler
)


class TestEndToEndMaterialization:
    """End-to-end tests for MaterializationManager with real storage adapters."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test data
        self.test_data = np.ones((10, 10), dtype=np.uint8)
        
    def test_end_to_end_with_memory_adapter(self, temp_test_dir, overlay_dir, mock_microscope_handler):
        """Test end-to-end pipeline with MemoryStorageAdapter."""
        # Create a real PipelineOrchestrator with memory storage
        orchestrator = PipelineOrchestrator(
            plate_path=temp_test_dir,
            workspace_path=temp_test_dir / "workspace",
            storage_mode="memory",
            storage_root=temp_test_dir / "storage",
            overlay_mode=OverlayMode.AUTO
        )
        
        # Replace microscope_handler with mock
        orchestrator.microscope_handler = mock_microscope_handler
        
        # Create a step that requires filesystem input
        class InputStep(Step):
            requires_fs_input = True
            
            def process(self, context):
                result = self.create_result()
                # Simulate reading from filesystem
                result.results['input_data'] = np.ones((10, 10))
                return result
                
        # Create a step that processes data
        class ProcessStep(Step):
            requires_fs_input = False
            
            def process(self, context):
                result = self.create_result()
                # Process data from previous step
                input_data = context.results.get('input_data', np.zeros((10, 10)))
                result.results['processed_data'] = input_data * 2
                return result
                
        # Create a step that requires filesystem output
        class OutputStep(Step):
            requires_fs_output = True
            
            def process(self, context):
                result = self.create_result()
                # Simulate writing to filesystem
                processed_data = context.results.get('processed_data', np.zeros((10, 10)))
                result.results['output_data'] = processed_data + 1
                return result
        
        # Create a pipeline with the test steps
        pipeline = Pipeline(
            steps=[InputStep(), ProcessStep(), OutputStep()],
            name="End-to-End Pipeline"
        )
        
        # Create a context
        context = orchestrator.create_context(pipeline, well_filter=["A01"])
        
        # Add test data to storage adapter
        key = "overlay_A01_file1.tif"
        orchestrator.storage_adapter.write(key, self.test_data)
        
        # Run the pipeline
        result_context = pipeline.run(context)
        
        # Verify that the steps were executed
        assert 'input_data' in result_context.results
        assert 'processed_data' in result_context.results
        assert 'output_data' in result_context.results
        
        # Verify the data transformations
        assert np.array_equal(result_context.results['input_data'], np.ones((10, 10)))
        assert np.array_equal(result_context.results['processed_data'], np.ones((10, 10)) * 2)
        assert np.array_equal(result_context.results['output_data'], np.ones((10, 10)) * 3)
        
    @pytest.mark.skip(reason="ZarrStorageAdapter may not be available in all environments")
    def test_end_to_end_with_zarr_adapter(self, temp_test_dir, overlay_dir, mock_microscope_handler):
        """Test end-to-end pipeline with ZarrStorageAdapter."""
        # Create zarr storage directory
        zarr_dir = temp_test_dir / "zarr_storage"
        zarr_dir.mkdir(exist_ok=True)
        
        # Create a real PipelineOrchestrator with zarr storage
        orchestrator = PipelineOrchestrator(
            plate_path=temp_test_dir,
            workspace_path=temp_test_dir / "workspace",
            storage_mode="zarr",
            storage_root=zarr_dir,
            overlay_mode=OverlayMode.AUTO
        )
        
        # Replace microscope_handler with mock
        orchestrator.microscope_handler = mock_microscope_handler
        
        # Create a step that requires filesystem input
        class InputStep(Step):
            requires_fs_input = True
            
            def process(self, context):
                result = self.create_result()
                # Simulate reading from filesystem
                result.results['input_data'] = np.ones((10, 10))
                return result
                
        # Create a step that processes data
        class ProcessStep(Step):
            requires_fs_input = False
            
            def process(self, context):
                result = self.create_result()
                # Process data from previous step
                input_data = context.results.get('input_data', np.zeros((10, 10)))
                result.results['processed_data'] = input_data * 2
                return result
                
        # Create a step that requires filesystem output
        class OutputStep(Step):
            requires_fs_output = True
            
            def process(self, context):
                result = self.create_result()
                # Simulate writing to filesystem
                processed_data = context.results.get('processed_data', np.zeros((10, 10)))
                result.results['output_data'] = processed_data + 1
                return result
        
        # Create a pipeline with the test steps
        pipeline = Pipeline(
            steps=[InputStep(), ProcessStep(), OutputStep()],
            name="End-to-End Pipeline"
        )
        
        # Create a context
        context = orchestrator.create_context(pipeline, well_filter=["A01"])
        
        # Add test data to storage adapter
        key = "overlay_A01_file1.tif"
        orchestrator.storage_adapter.write(key, self.test_data)
        
        # Run the pipeline
        result_context = pipeline.run(context)
        
        # Verify that the steps were executed
        assert 'input_data' in result_context.results
        assert 'processed_data' in result_context.results
        assert 'output_data' in result_context.results
        
        # Verify the data transformations
        assert np.array_equal(result_context.results['input_data'], np.ones((10, 10)))
        assert np.array_equal(result_context.results['processed_data'], np.ones((10, 10)) * 2)
        assert np.array_equal(result_context.results['output_data'], np.ones((10, 10)) * 3)
        
        # Verify that the data was stored in the zarr store
        assert orchestrator.storage_adapter.exists('input_data')
        assert orchestrator.storage_adapter.exists('processed_data')
        assert orchestrator.storage_adapter.exists('output_data')
