"""
Fixtures for materialization tests.
"""

import pytest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for testing."""
    test_dir = Path(tempfile.mkdtemp())
    
    # Create subdirectories
    (test_dir / "workspace").mkdir(exist_ok=True)
    (test_dir / "positions").mkdir(exist_ok=True)
    (test_dir / "stitched").mkdir(exist_ok=True)
    (test_dir / "storage").mkdir(exist_ok=True)
    
    # Create test files
    for well in ["A01", "A02", "B01", "B02"]:
        well_dir = test_dir / well
        well_dir.mkdir(exist_ok=True)
        
        # Create test images
        for i in range(3):
            for ch in range(2):
                img_path = well_dir / f"img_{i}_ch{ch}.tif"
                with open(img_path, "wb") as f:
                    f.write(b"test image data")
    
    yield test_dir
    
    # Clean up
    shutil.rmtree(test_dir)


@pytest.fixture
def overlay_dir(temp_test_dir):
    """Create an overlay directory for testing."""
    overlay_dir = temp_test_dir / "overlay"
    overlay_dir.mkdir(exist_ok=True)
    
    yield overlay_dir


@pytest.fixture
def mock_file_manager():
    """Create a mock file manager."""
    file_manager = MagicMock()
    
    # Mock file operations
    file_manager.exists.return_value = True
    file_manager.ensure_directory.return_value = True
    file_manager.load_image.return_value = np.ones((10, 10), dtype=np.uint8)
    file_manager.save_image.return_value = True
    file_manager.list_files.return_value = []
    
    return file_manager


@pytest.fixture
def mock_microscope_handler():
    """Create a mock microscope handler."""
    microscope_handler = MagicMock()
    
    # Mock parser
    parser = MagicMock()
    parser.path_list_from_pattern.return_value = ["img_0_ch0.tif", "img_1_ch0.tif", "img_2_ch0.tif"]
    microscope_handler.parser = parser
    
    # Mock auto_detect_patterns
    microscope_handler.auto_detect_patterns.return_value = {
        "A01": ["*_ch0.tif", "*_ch1.tif"],
        "A02": ["*_ch0.tif", "*_ch1.tif"],
        "B01": ["*_ch0.tif", "*_ch1.tif"],
        "B02": ["*_ch0.tif", "*_ch1.tif"]
    }
    
    return microscope_handler
