"""
Unit tests for with_name and with_suffix methods in VirtualPath implementations.
"""

import os
import pytest
from pathlib import Path

from ezstitcher.io.virtual_path import PhysicalPath
from ezstitcher.io.memory_path import MemoryPath
from ezstitcher.io.zarr_path import ZarrPath


class TestPhysicalPathNameSuffix:
    """Test with_name and with_suffix methods for PhysicalPath."""

    def test_with_name_basic(self):
        """Test basic functionality of with_name."""
        path = PhysicalPath("foo/bar.txt")
        new_path = path.with_name("baz.txt")
        assert str(new_path) == "foo/baz.txt"
        assert new_path.name() == "baz.txt"

    def test_with_name_preserves_root_context(self):
        """Test that with_name preserves root_context."""
        path = PhysicalPath("foo/bar.txt", root_context="test")
        new_path = path.with_name("baz.txt")
        assert new_path.root_context == "test"

    def test_with_name_raises_on_empty_name(self):
        """Test that with_name raises ValueError on paths with no name component."""
        path = PhysicalPath("/")
        with pytest.raises(ValueError):
            path.with_name("new_name")

    def test_with_suffix_basic(self):
        """Test basic functionality of with_suffix."""
        path = PhysicalPath("foo/bar.txt")
        new_path = path.with_suffix(".md")
        assert str(new_path) == "foo/bar.md"
        assert new_path.suffix() == ".md"

    def test_with_suffix_no_suffix(self):
        """Test with_suffix on a path with no suffix."""
        path = PhysicalPath("foo/bar")
        new_path = path.with_suffix(".txt")
        assert str(new_path) == "foo/bar.txt"

    def test_with_suffix_empty_suffix(self):
        """Test with_suffix with an empty suffix."""
        path = PhysicalPath("foo/bar.txt")
        new_path = path.with_suffix("")
        assert str(new_path) == "foo/bar"
        assert new_path.suffix() == ""

    def test_with_suffix_preserves_root_context(self):
        """Test that with_suffix preserves root_context."""
        path = PhysicalPath("foo/bar.txt", root_context="test")
        new_path = path.with_suffix(".md")
        assert new_path.root_context == "test"


class TestMemoryPathNameSuffix:
    """Test with_name and with_suffix methods for MemoryPath."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset MemoryPath storage before and after each test."""
        MemoryPath.reset_storage()
        yield
        MemoryPath.reset_storage()

    def test_with_name_basic(self):
        """Test basic functionality of with_name."""
        path = MemoryPath("/foo/bar.txt")
        new_path = path.with_name("baz.txt")
        assert str(new_path) == "/foo/baz.txt"
        assert new_path.name() == "baz.txt"

    def test_with_name_preserves_root_context(self):
        """Test that with_name preserves root_context."""
        path = MemoryPath("/foo/bar.txt", root_context="test")
        new_path = path.with_name("baz.txt")
        assert new_path.root_context == "test"

    def test_with_name_raises_on_empty_name(self):
        """Test that with_name raises ValueError on paths with no name component."""
        path = MemoryPath("/")
        with pytest.raises(ValueError):
            path.with_name("new_name")

    def test_with_suffix_basic(self):
        """Test basic functionality of with_suffix."""
        path = MemoryPath("/foo/bar.txt")
        new_path = path.with_suffix(".md")
        assert str(new_path) == "/foo/bar.md"
        assert new_path.suffix() == ".md"

    def test_with_suffix_no_suffix(self):
        """Test with_suffix on a path with no suffix."""
        path = MemoryPath("/foo/bar")
        new_path = path.with_suffix(".txt")
        assert str(new_path) == "/foo/bar.txt"

    def test_with_suffix_empty_suffix(self):
        """Test with_suffix with an empty suffix."""
        path = MemoryPath("/foo/bar.txt")
        new_path = path.with_suffix("")
        assert str(new_path) == "/foo/bar"
        assert new_path.suffix() == ""

    def test_with_suffix_preserves_root_context(self):
        """Test that with_suffix preserves root_context."""
        path = MemoryPath("/foo/bar.txt", root_context="test")
        new_path = path.with_suffix(".md")
        assert new_path.root_context == "test"

    def test_with_suffix_adds_dot_if_missing(self):
        """Test that with_suffix adds a dot if it's missing."""
        path = MemoryPath("/foo/bar.txt")
        new_path = path.with_suffix("md")
        assert str(new_path) == "/foo/bar.md"
        assert new_path.suffix() == ".md"


class TestZarrPathNameSuffix:
    """Test with_name and with_suffix methods for ZarrPath."""

    @pytest.fixture
    def zarr_path(self):
        """Create a ZarrPath for testing."""
        path, _ = ZarrPath.create_memory_zarr()
        return path / "foo" / "bar.txt"

    @pytest.fixture
    def zarr_path_with_context(self):
        """Create a ZarrPath with root_context for testing."""
        path, _ = ZarrPath.create_memory_zarr(root_context="test")
        return path / "foo" / "bar.txt"

    def test_with_name_basic(self, zarr_path):
        """Test basic functionality of with_name."""
        new_path = zarr_path.with_name("baz.txt")
        assert str(new_path) == "/foo/baz.txt"
        assert new_path.name() == "baz.txt"

    def test_with_name_preserves_root_context(self, zarr_path_with_context):
        """Test that with_name preserves root_context."""
        new_path = zarr_path_with_context.with_name("baz.txt")
        assert new_path.root_context == "test"

    def test_with_name_raises_on_empty_name(self):
        """Test that with_name raises ValueError on paths with no name component."""
        path, _ = ZarrPath.create_memory_zarr()
        with pytest.raises(ValueError):
            path.with_name("new_name")

    def test_with_suffix_basic(self, zarr_path):
        """Test basic functionality of with_suffix."""
        new_path = zarr_path.with_suffix(".md")
        assert str(new_path) == "/foo/bar.md"
        assert new_path.suffix() == ".md"

    def test_with_suffix_no_suffix(self, zarr_path):
        """Test with_suffix on a path with no suffix."""
        no_suffix_path = zarr_path.with_suffix("")
        new_path = no_suffix_path.with_suffix(".txt")
        assert str(new_path) == "/foo/bar.txt"

    def test_with_suffix_empty_suffix(self, zarr_path):
        """Test with_suffix with an empty suffix."""
        new_path = zarr_path.with_suffix("")
        assert str(new_path) == "/foo/bar"
        assert new_path.suffix() == ""

    def test_with_suffix_preserves_root_context(self, zarr_path_with_context):
        """Test that with_suffix preserves root_context."""
        new_path = zarr_path_with_context.with_suffix(".md")
        assert new_path.root_context == "test"

    def test_with_suffix_adds_dot_if_missing(self, zarr_path):
        """Test that with_suffix adds a dot if it's missing."""
        new_path = zarr_path.with_suffix("md")
        assert str(new_path) == "/foo/bar.md"
        assert new_path.suffix() == ".md"
