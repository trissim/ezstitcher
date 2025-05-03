"""Tests for storage adapter path resolution."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.config import PipelineConfig
from ezstitcher.io.filemanager import FileManager


class TestStorageAdapterPathResolution:
    """Test the storage adapter path resolution logic."""

    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.plate_path = self.temp_dir / "plate"
        self.plate_path.mkdir(exist_ok=True)
        
        # Create a file manager for testing
        self.file_manager = FileManager(backend="disk")
        
    def teardown_method(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_storage_adapter_path_resolution_zarr(self):
        """Test that the storage adapter's root path is correctly suffixed for zarr mode."""
        # Create a config with custom suffixes
        config = PipelineConfig(
            out_dir_suffix="_out",
            positions_dir_suffix="_positions",
            stitched_dir_suffix="_stitched"
        )
        
        # Create a mock file manager
        mock_file_manager = MagicMock(spec=FileManager)
        
        # Create an orchestrator with zarr storage mode
        orchestrator = PipelineOrchestrator(
            plate_path=self.plate_path,
            config=config,
            storage_mode="zarr"
        )
        
        # Replace the file manager with our mock
        orchestrator.file_manager = mock_file_manager
        
        # Initialize the storage adapter
        orchestrator.initialize_storage_adapter()
        
        # Verify that the storage adapter's root path is correctly suffixed
        expected_storage_root = self.plate_path.parent / f"{self.plate_path.name}{config.out_dir_suffix}" / "zarr_storage"
        assert orchestrator.storage_adapter is not None
        assert orchestrator.storage_adapter.storage_root == expected_storage_root
        
        # Verify that the storage adapter's root path is not the same as the input directory
        assert orchestrator.storage_adapter.storage_root != orchestrator.input_dir
        
        # Verify that the directory was created
        mock_file_manager.ensure_directory.assert_any_call(expected_storage_root)
    
    def test_storage_adapter_path_resolution_memory(self):
        """Test that the storage adapter's root path is correctly suffixed for memory mode."""
        # Create a config with custom suffixes
        config = PipelineConfig(
            out_dir_suffix="_out",
            positions_dir_suffix="_positions",
            stitched_dir_suffix="_stitched"
        )
        
        # Create a mock file manager
        mock_file_manager = MagicMock(spec=FileManager)
        
        # Create an orchestrator with memory storage mode
        orchestrator = PipelineOrchestrator(
            plate_path=self.plate_path,
            config=config,
            storage_mode="memory"
        )
        
        # Replace the file manager with our mock
        orchestrator.file_manager = mock_file_manager
        
        # Initialize the storage adapter
        orchestrator.initialize_storage_adapter()
        
        # Verify that the overlay root is correctly set
        expected_base_dir = self.plate_path.parent / f"{self.plate_path.name}{config.out_dir_suffix}"
        expected_storage_root = expected_base_dir / "memory_storage"
        expected_overlay_root = expected_storage_root / "overlay"
        
        # Verify that the directories were created
        mock_file_manager.ensure_directory.assert_any_call(expected_storage_root)
        mock_file_manager.ensure_directory.assert_any_call(expected_overlay_root)
    
    def test_storage_adapter_path_resolution_with_explicit_storage_root(self):
        """Test that the storage adapter uses the explicit storage_root if provided."""
        # Create a config with custom suffixes
        config = PipelineConfig(
            out_dir_suffix="_out",
            positions_dir_suffix="_positions",
            stitched_dir_suffix="_stitched"
        )
        
        # Create a mock file manager
        mock_file_manager = MagicMock(spec=FileManager)
        
        # Create an explicit storage root
        explicit_storage_root = self.temp_dir / "explicit_storage_root"
        
        # Create an orchestrator with zarr storage mode and explicit storage root
        orchestrator = PipelineOrchestrator(
            plate_path=self.plate_path,
            config=config,
            storage_mode="zarr",
            storage_root=explicit_storage_root
        )
        
        # Replace the file manager with our mock
        orchestrator.file_manager = mock_file_manager
        
        # Initialize the storage adapter
        orchestrator.initialize_storage_adapter()
        
        # Verify that the storage adapter uses the explicit storage root
        assert orchestrator.storage_adapter is not None
        assert orchestrator.storage_adapter.storage_root == explicit_storage_root
        
        # Verify that the directory was created
        mock_file_manager.ensure_directory.assert_any_call(explicit_storage_root)
    
    def test_integration_with_real_file_manager(self):
        """Test the storage adapter path resolution with a real file manager."""
        # Create a config with custom suffixes
        config = PipelineConfig(
            out_dir_suffix="_out",
            positions_dir_suffix="_positions",
            stitched_dir_suffix="_stitched"
        )
        
        # Create an orchestrator with zarr storage mode
        orchestrator = PipelineOrchestrator(
            plate_path=self.plate_path,
            config=config,
            storage_mode="zarr"
        )
        
        # Initialize the file manager
        orchestrator.file_manager = FileManager(backend="disk")
        
        # Initialize the storage adapter
        orchestrator.initialize_storage_adapter()
        
        # Verify that the storage adapter's root path is correctly suffixed
        expected_storage_root = self.plate_path.parent / f"{self.plate_path.name}{config.out_dir_suffix}" / "zarr_storage"
        assert orchestrator.storage_adapter is not None
        assert orchestrator.storage_adapter.storage_root == expected_storage_root
        
        # Verify that the storage adapter's root path is not the same as the input directory
        assert orchestrator.storage_adapter.storage_root != orchestrator.input_dir
        
        # Verify that the directory was created
        assert expected_storage_root.exists()
        assert (expected_storage_root / "overlay").exists()
