"""Integration tests for pattern format adapters."""

import pytest
from pathlib import Path
import tempfile
import os
import shutil

# Import ashlar modules for mocking
import ashlar.fileseries
import ashlar.reg

from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.microscope_interfaces import create_microscope_handler
from ezstitcher.io.filemanager import FileManager


class TestPatternAdapterIntegration:
    """Integration tests for pattern format adapters."""

    @pytest.fixture
    def setup_test_files(self):
        """Set up test files."""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Create test files
        for i in range(1, 5):
            with open(os.path.join(temp_dir, f"test_s{i}.tif"), "w") as f:
                f.write(f"Test file {i}")

        yield temp_dir

        # Clean up
        shutil.rmtree(temp_dir)

    def test_filename_parser_with_adapter(self, setup_test_files):
        """Test FilenameParser with pattern adapter."""
        temp_dir = setup_test_files

        # Create a file manager
        file_manager = FileManager()

        # Create a microscope handler with Ashlar pattern format
        handler = create_microscope_handler(
            microscope_type="imagexpress",
            plate_folder=temp_dir,
            file_manager=file_manager,
            pattern_format="ashlar"
        )

        # Test with Ashlar pattern
        ashlar_pattern = "test_s{series}.tif"
        files = handler.parser.path_list_from_pattern(temp_dir, ashlar_pattern)

        # Should find all test files
        assert len(files) == 4
        assert "test_s1.tif" in files
        assert "test_s2.tif" in files
        assert "test_s3.tif" in files
        assert "test_s4.tif" in files

        # Test with internal pattern
        internal_pattern = "test_s{iii}.tif"
        files = handler.parser.path_list_from_pattern(temp_dir, internal_pattern)

        # Should also find all test files
        assert len(files) == 4

    def test_stitcher_with_adapter(self, setup_test_files, mocker):
        """Test Stitcher with pattern adapter."""
        temp_dir = setup_test_files

        # Mock Ashlar components
        mocker.patch("ashlar.fileseries.FileSeriesReader")
        mocker.patch("ashlar.reg.EdgeAligner")
        mocker.patch("ashlar.reg.Mosaic")

        # Create a file manager
        file_manager = FileManager()

        # Create a microscope handler
        handler = create_microscope_handler(
            microscope_type="imagexpress",
            plate_folder=temp_dir,
            file_manager=file_manager
        )

        # Create a stitcher with Ashlar pattern format
        stitcher = Stitcher(
            file_manager=file_manager,
            filename_parser=handler.parser,
            pattern_format="ashlar"
        )

        # Mock the generate_positions_df method
        mocker.patch.object(
            stitcher,
            "generate_positions_df",
            return_value="mocked_df"
        )

        # Mock the save_positions_df method
        mocker.patch.object(
            stitcher,
            "save_positions_df"
        )

        # Test with internal pattern
        internal_pattern = "test_s{iii}.tif"
        positions_path = Path(temp_dir) / "positions.csv"

        # Should convert to Ashlar pattern internally
        result = stitcher._generate_positions_ashlar(
            temp_dir,
            internal_pattern,
            positions_path,
            2,
            2
        )

        # Should succeed
        assert result is True

        # Should have called FileSeriesReader with Ashlar pattern
        ashlar.fileseries.FileSeriesReader.assert_called_once()
        args, kwargs = ashlar.fileseries.FileSeriesReader.call_args
        assert kwargs["pattern"] == "test_s{series}.tif"
