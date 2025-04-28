import pytest
import numpy as np
import tempfile
from pathlib import Path
import tifffile

from ezstitcher.core.file_system_manager import FileSystemManager


class TestFileSystemManager:
    """Tests for the FileSystemManager class."""

    def test_load_image_2d(self):
        """Test loading a 2D image."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a 2D test image
            img_path = Path(temp_dir) / "test_2d.tif"
            test_img = np.ones((100, 100), dtype=np.uint16) * 1000
            tifffile.imwrite(str(img_path), test_img)

            # Load the image
            loaded_img = FileSystemManager.load_image(img_path)

            # Verify the image was loaded correctly
            assert loaded_img is not None
            assert loaded_img.shape == test_img.shape
            assert loaded_img.dtype == test_img.dtype
            assert np.array_equal(loaded_img, test_img)

    def test_load_image_3d(self):
        """Test loading a 3D image raises an error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a 3D test image
            img_path = Path(temp_dir) / "test_3d.tif"
            test_img = np.ones((3, 100, 100), dtype=np.uint16) * 1000
            tifffile.imwrite(str(img_path), test_img)

            # Load the image - should return None due to the error
            loaded_img = FileSystemManager.load_image(img_path)

            # Verify the image was not loaded
            assert loaded_img is None

    def test_load_image_nonexistent(self):
        """Test loading a non-existent image."""
        # Try to load a non-existent image
        loaded_img = FileSystemManager.load_image("nonexistent.tif")

        # Verify the image was not loaded
        assert loaded_img is None

    def test_load_image_invalid(self):
        """Test loading an invalid image file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create an invalid image file (just text)
            img_path = Path(temp_dir) / "invalid.tif"
            with open(img_path, 'w') as f:
                f.write("This is not an image file")

            # Try to load the invalid image
            loaded_img = FileSystemManager.load_image(img_path)

            # Verify the image was not loaded
            assert loaded_img is None
