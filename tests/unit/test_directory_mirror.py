"""
Tests for the directory_mirror module.
"""

import os
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from ezstitcher.io.directory_mirror import (
    mirror_directory_with_symlinks,
    OverwriteStrategy,
    _ensure_source_directory_exists,
    _prepare_target_directory,
    _create_symlink,
    _process_directory_items
)


@pytest.fixture
def setup_test_dirs(tmp_path):
    """Create test directories and files for testing."""
    # Create source directory structure
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    
    # Create some files in the source directory
    (source_dir / "file1.txt").write_text("content1")
    (source_dir / "file2.txt").write_text("content2")
    
    # Create a subdirectory with files
    subdir = source_dir / "subdir"
    subdir.mkdir()
    (subdir / "file3.txt").write_text("content3")
    
    # Create target directory
    target_dir = tmp_path / "target"
    
    return source_dir, target_dir


def test_ensure_source_directory_exists(tmp_path):
    """Test _ensure_source_directory_exists function."""
    # Test with existing directory
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    assert _ensure_source_directory_exists(existing_dir) is True
    
    # Test with non-existing directory
    non_existing_dir = tmp_path / "non_existing"
    assert _ensure_source_directory_exists(non_existing_dir) is False


def test_prepare_target_directory(tmp_path):
    """Test _prepare_target_directory function with different strategies."""
    # Test with non-existing directory
    non_existing_dir = tmp_path / "non_existing"
    assert _prepare_target_directory(non_existing_dir, OverwriteStrategy.REPLACE) is True
    assert non_existing_dir.exists()
    
    # Test with existing directory and REPLACE strategy
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    (existing_dir / "old_file.txt").write_text("old content")
    
    assert _prepare_target_directory(existing_dir, OverwriteStrategy.REPLACE) is True
    assert existing_dir.exists()
    assert not (existing_dir / "old_file.txt").exists()
    
    # Test with existing directory and SKIP strategy
    existing_dir = tmp_path / "existing_skip"
    existing_dir.mkdir()
    (existing_dir / "old_file.txt").write_text("old content")
    
    assert _prepare_target_directory(existing_dir, OverwriteStrategy.SKIP) is False
    assert existing_dir.exists()
    assert (existing_dir / "old_file.txt").exists()
    
    # Test with existing directory and MERGE strategy
    existing_dir = tmp_path / "existing_merge"
    existing_dir.mkdir()
    (existing_dir / "old_file.txt").write_text("old content")
    
    assert _prepare_target_directory(existing_dir, OverwriteStrategy.MERGE) is True
    assert existing_dir.exists()
    assert (existing_dir / "old_file.txt").exists()


def test_create_symlink(tmp_path):
    """Test _create_symlink function."""
    # Create source file
    source_file = tmp_path / "source_file.txt"
    source_file.write_text("test content")
    
    # Create target path
    target_path = tmp_path / "target_file.txt"
    
    # Create symlink
    assert _create_symlink(source_file, target_path) is True
    assert target_path.exists()
    assert target_path.is_symlink()
    assert target_path.resolve() == source_file.resolve()
    
    # Test overwriting existing symlink
    new_source_file = tmp_path / "new_source_file.txt"
    new_source_file.write_text("new content")
    
    assert _create_symlink(new_source_file, target_path) is True
    assert target_path.exists()
    assert target_path.is_symlink()
    assert target_path.resolve() == new_source_file.resolve()


def test_mirror_directory_with_symlinks_basic(setup_test_dirs):
    """Test basic functionality of mirror_directory_with_symlinks."""
    source_dir, target_dir = setup_test_dirs
    
    # Mirror directory
    symlinks_created = mirror_directory_with_symlinks(source_dir, target_dir)
    
    # Check results
    assert symlinks_created == 3  # 2 files in source dir + 1 file in subdir
    assert (target_dir / "file1.txt").exists()
    assert (target_dir / "file2.txt").exists()
    assert (target_dir / "subdir" / "file3.txt").exists()
    
    # Verify they are symlinks
    assert (target_dir / "file1.txt").is_symlink()
    assert (target_dir / "file2.txt").is_symlink()
    assert (target_dir / "subdir" / "file3.txt").is_symlink()
    
    # Verify they point to the correct files
    assert (target_dir / "file1.txt").resolve() == (source_dir / "file1.txt").resolve()
    assert (target_dir / "file2.txt").resolve() == (source_dir / "file2.txt").resolve()
    assert (target_dir / "subdir" / "file3.txt").resolve() == (source_dir / "subdir" / "file3.txt").resolve()


def test_mirror_directory_with_symlinks_non_recursive(setup_test_dirs):
    """Test mirror_directory_with_symlinks with recursive=False."""
    source_dir, target_dir = setup_test_dirs
    
    # Mirror directory non-recursively
    symlinks_created = mirror_directory_with_symlinks(source_dir, target_dir, recursive=False)
    
    # Check results
    assert symlinks_created == 2  # Only 2 files in source dir
    assert (target_dir / "file1.txt").exists()
    assert (target_dir / "file2.txt").exists()
    assert not (target_dir / "subdir").exists()


def test_mirror_directory_with_symlinks_overwrite_strategies(setup_test_dirs):
    """Test mirror_directory_with_symlinks with different overwrite strategies."""
    source_dir, target_dir = setup_test_dirs
    
    # Create existing file in target directory
    target_dir.mkdir()
    (target_dir / "existing.txt").write_text("existing content")
    
    # Test with REPLACE strategy
    symlinks_created = mirror_directory_with_symlinks(
        source_dir, target_dir, strategy=OverwriteStrategy.REPLACE
    )
    
    # Check results
    assert symlinks_created == 3
    assert (target_dir / "file1.txt").exists()
    assert not (target_dir / "existing.txt").exists()  # Should be removed
    
    # Reset target directory
    shutil.rmtree(target_dir)
    target_dir.mkdir()
    (target_dir / "existing.txt").write_text("existing content")
    
    # Test with MERGE strategy
    symlinks_created = mirror_directory_with_symlinks(
        source_dir, target_dir, strategy=OverwriteStrategy.MERGE
    )
    
    # Check results
    assert symlinks_created == 3
    assert (target_dir / "file1.txt").exists()
    assert (target_dir / "existing.txt").exists()  # Should still exist
    
    # Reset target directory
    shutil.rmtree(target_dir)
    target_dir.mkdir()
    (target_dir / "existing.txt").write_text("existing content")
    
    # Test with SKIP strategy
    symlinks_created = mirror_directory_with_symlinks(
        source_dir, target_dir, strategy=OverwriteStrategy.SKIP
    )
    
    # Check results
    assert symlinks_created == 0  # Should skip because target exists
    assert not (target_dir / "file1.txt").exists()
    assert (target_dir / "existing.txt").exists()  # Should still exist


def test_mirror_directory_with_symlinks_progress_callback(setup_test_dirs):
    """Test mirror_directory_with_symlinks with progress callback."""
    source_dir, target_dir = setup_test_dirs
    
    # Create mock callback
    mock_callback = MagicMock()
    
    # Mirror directory with callback
    symlinks_created = mirror_directory_with_symlinks(
        source_dir, target_dir, progress_callback=mock_callback
    )
    
    # Check results
    assert symlinks_created == 3
    assert mock_callback.call_count > 0  # Should be called multiple times


def test_mirror_directory_with_symlinks_string_strategy(setup_test_dirs):
    """Test mirror_directory_with_symlinks with string strategy."""
    source_dir, target_dir = setup_test_dirs
    
    # Mirror directory with string strategy
    symlinks_created = mirror_directory_with_symlinks(
        source_dir, target_dir, strategy="merge"
    )
    
    # Check results
    assert symlinks_created == 3
    assert (target_dir / "file1.txt").exists()


def test_mirror_directory_with_symlinks_invalid_strategy(setup_test_dirs):
    """Test mirror_directory_with_symlinks with invalid strategy."""
    source_dir, target_dir = setup_test_dirs
    
    # Mirror directory with invalid strategy (should default to REPLACE)
    symlinks_created = mirror_directory_with_symlinks(
        source_dir, target_dir, strategy="invalid"
    )
    
    # Check results
    assert symlinks_created == 3
    assert (target_dir / "file1.txt").exists()


def test_mirror_directory_with_symlinks_source_not_exists(tmp_path):
    """Test mirror_directory_with_symlinks with non-existent source directory."""
    source_dir = tmp_path / "non_existent"
    target_dir = tmp_path / "target"
    
    # Mirror directory with non-existent source
    symlinks_created = mirror_directory_with_symlinks(source_dir, target_dir)
    
    # Check results
    assert symlinks_created == 0
    assert not target_dir.exists()
