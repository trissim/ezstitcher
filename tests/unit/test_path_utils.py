"""
Tests for path_utils module.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ezstitcher.io.path_utils import is_disk_path
from ezstitcher.io.virtual_path import VirtualPath

class TestIsDiskPath:
    def test_string_path(self):
        """Test is_disk_path with string paths."""
        assert is_disk_path("/path/to/file.txt") is True
        assert is_disk_path("relative/path.txt") is True
        assert is_disk_path("memory://path/to/file.txt") is False
    
    def test_path_object(self):
        """Test is_disk_path with Path objects."""
        assert is_disk_path(Path("/path/to/file.txt")) is True
        assert is_disk_path(Path("relative/path.txt")) is True
    
    def test_virtual_path(self):
        """Test is_disk_path with VirtualPath objects."""
        # Mock VirtualPath with physical path
        vp_with_physical = MagicMock(spec=VirtualPath)
        vp_with_physical.to_physical_path.return_value = Path("/path/to/file.txt")
        assert is_disk_path(vp_with_physical) is True
        
        # Mock VirtualPath without physical path
        vp_without_physical = MagicMock(spec=VirtualPath)
        vp_without_physical.to_physical_path.return_value = None
        assert is_disk_path(vp_without_physical) is False
    
    def test_object_with_to_physical_path(self):
        """Test is_disk_path with objects that have to_physical_path method."""
        # Object with to_physical_path that returns a path
        obj_with_path = MagicMock()
        obj_with_path.to_physical_path.return_value = Path("/path/to/file.txt")
        assert is_disk_path(obj_with_path) is True
        
        # Object with to_physical_path that returns None
        obj_without_path = MagicMock()
        obj_without_path.to_physical_path.return_value = None
        assert is_disk_path(obj_without_path) is False