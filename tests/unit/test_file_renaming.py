"""
Test file renaming functionality.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path

from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.filename_parser import ImageXpressFilenameParser
from ezstitcher.core.plate_processor import PlateProcessor
from ezstitcher.core.config import PlateProcessorConfig
from ezstitcher.core.main import process_plate_auto


def create_test_files(directory, filenames):
    """Create test files in a directory."""
    for filename in filenames:
        with open(os.path.join(directory, filename), 'w') as f:
            f.write("Test file")


def test_file_system_manager_rename_files():
    """Test FileSystemManager.rename_files_with_consistent_padding."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with mixed padding
        filenames = [
            "A01_s1_w1.tif",
            "A01_s2_w1.tif",
            "A01_s10_w1.tif",
            "A01_s001_w2.tif",  # Already padded
        ]
        create_test_files(temp_dir, filenames)

        # Create FileSystemManager and FilenameParser
        fs_manager = FileSystemManager()
        parser = ImageXpressFilenameParser()

        # Test dry run
        rename_map = fs_manager.rename_files_with_consistent_padding(
            temp_dir,
            parser=parser,
            width=3,
            dry_run=True
        )

        # Check that the rename map is correct
        assert len(rename_map) == 3  # 3 files need renaming
        assert "A01_s1_w1.tif" in rename_map
        assert rename_map["A01_s1_w1.tif"] == "A01_s001_w1.tif"
        assert "A01_s2_w1.tif" in rename_map
        assert rename_map["A01_s2_w1.tif"] == "A01_s002_w1.tif"
        assert "A01_s10_w1.tif" in rename_map
        assert rename_map["A01_s10_w1.tif"] == "A01_s010_w1.tif"

        # Check that the files were not actually renamed
        assert os.path.exists(os.path.join(temp_dir, "A01_s1_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s2_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s10_w1.tif"))

        # Test actual renaming
        rename_map = fs_manager.rename_files_with_consistent_padding(
            temp_dir,
            parser=parser,
            width=3,
            dry_run=False
        )

        # Check that the files were renamed
        assert not os.path.exists(os.path.join(temp_dir, "A01_s1_w1.tif"))
        assert not os.path.exists(os.path.join(temp_dir, "A01_s2_w1.tif"))
        assert not os.path.exists(os.path.join(temp_dir, "A01_s10_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s001_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s002_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s010_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s001_w2.tif"))  # Already padded


def test_plate_processor_rename_files():
    """Test PlateProcessor.rename_files_with_consistent_padding."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with mixed padding
        filenames = [
            "A01_s1_w1.tif",
            "A01_s2_w1.tif",
            "A01_s10_w1.tif",
        ]
        create_test_files(temp_dir, filenames)

        # Create PlateProcessor
        config = PlateProcessorConfig()
        config.microscope_type = "ImageXpress"
        processor = PlateProcessor(config)
        processor._current_plate_folder = temp_dir

        # Test renaming
        rename_map = processor.rename_files_with_consistent_padding(
            width=3,
            dry_run=False
        )

        # Check that the files were renamed
        assert not os.path.exists(os.path.join(temp_dir, "A01_s1_w1.tif"))
        assert not os.path.exists(os.path.join(temp_dir, "A01_s2_w1.tif"))
        assert not os.path.exists(os.path.join(temp_dir, "A01_s10_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s001_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s002_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s010_w1.tif"))


def test_process_plate_auto_rename_files():
    """Test process_plate_auto with rename_files=True."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with mixed padding
        filenames = [
            "A01_s1_w1.tif",
            "A01_s2_w1.tif",
            "A01_s10_w1.tif",
        ]
        create_test_files(temp_dir, filenames)

        # Test process_plate_auto with rename_files=True and rename_only=True
        success = process_plate_auto(
            temp_dir,
            microscope_type="ImageXpress",
            rename_files=True,
            rename_only=True
        )

        # Check that the process was successful
        assert success

        # Check that the files were renamed
        assert not os.path.exists(os.path.join(temp_dir, "A01_s1_w1.tif"))
        assert not os.path.exists(os.path.join(temp_dir, "A01_s2_w1.tif"))
        assert not os.path.exists(os.path.join(temp_dir, "A01_s10_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s001_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s002_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s010_w1.tif"))


def test_conflict_handling():
    """Test handling of filename conflicts."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files with a conflict
        filenames = [
            "A01_s1_w1.tif",
            "A01_s001_w1.tif",  # This will conflict with the padded version of s1
        ]
        create_test_files(temp_dir, filenames)

        # Create FileSystemManager and FilenameParser
        fs_manager = FileSystemManager()
        parser = ImageXpressFilenameParser()

        # Test renaming
        rename_map = fs_manager.rename_files_with_consistent_padding(
            temp_dir,
            parser=parser,
            width=3,
            dry_run=False
        )

        # Check that no files were renamed due to the conflict
        assert len(rename_map) == 0
        assert os.path.exists(os.path.join(temp_dir, "A01_s1_w1.tif"))
        assert os.path.exists(os.path.join(temp_dir, "A01_s001_w1.tif"))
