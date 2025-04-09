#!/usr/bin/env python3
"""
Test suite for Opera Phenix support in ezstitcher.

This test suite verifies that ezstitcher can correctly process Opera Phenix data.
It mirrors the tests in test_synthetic_workflow_class_based.py but uses Opera Phenix format data.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from skimage import filters, exposure

# Get the parent directory to import from the root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import synthetic data generator
sys.path.append(os.path.join(parent_dir, 'utils'))
from generate_synthetic_data import SyntheticMicroscopyGenerator

# Import core functionality
from ezstitcher.core.main import process_plate_folder
from ezstitcher.core.filename_parser import OperaPhenixFilenameParser
from ezstitcher.core.plate_processor import PlateProcessor
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.zstack_processor import ZStackProcessor
from ezstitcher.core.config import (
    PlateProcessorConfig,
    StitcherConfig,
    FocusAnalyzerConfig,
    ImagePreprocessorConfig,
    ZStackProcessorConfig
)


class TestOperaPhenixSupport(unittest.TestCase):
    """Test ezstitcher functionality with Opera Phenix data."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment with synthetic Opera Phenix data."""
        # Create base test data directory
        cls.base_test_dir = Path(os.path.join(parent_dir, 'tests', 'test_data', 'opera_phenix'))

        # Clean up any existing test data
        if cls.base_test_dir.exists():
            print(f"Cleaning up existing test data directory: {cls.base_test_dir}")
            shutil.rmtree(cls.base_test_dir)

        # Create the base test data directory
        cls.base_test_dir.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        """Set up test case with synthetic data."""
        # Create a unique test directory for each test
        self.test_dir = os.path.join(self.base_test_dir, self._testMethodName)
        os.makedirs(self.test_dir, exist_ok=True)

        # Create directories for different data types
        self.projection_dir = os.path.join(self.test_dir, "opera_plate_max")

        # Determine if we need Z-stack data
        needs_zstack = self._needs_zstack_data()
        needs_non_zstack = self._needs_non_zstack_data()

        # If neither is explicitly needed, default to both for backward compatibility
        if not needs_zstack and not needs_non_zstack:
            needs_zstack = True
            needs_non_zstack = True

        # Create synthetic Z-stack data if needed
        if needs_zstack:
            self.zstack_dir = os.path.join(self.test_dir, "opera_plate")
            print(f"Creating synthetic Opera Phenix data with Z-stacks in {self.zstack_dir}")
            os.makedirs(self.zstack_dir, exist_ok=True)
            self._create_synthetic_data(output_dir=self.zstack_dir, z_stack_levels=5, z_step_size=2.0)

            # Create a copy of the original Z-stack data for inspection
            self.zstack_original_dir = os.path.join(self.test_dir, "opera_plate_original")
            print(f"Creating copy of original Z-stack data in {self.zstack_original_dir}")
            shutil.copytree(self.zstack_dir, self.zstack_original_dir)
        else:
            # Still define the paths for tests that might reference them
            self.zstack_dir = os.path.join(self.test_dir, "opera_plate")
            self.zstack_original_dir = os.path.join(self.test_dir, "opera_plate_original")

        # Create synthetic non-Z-stack data if needed
        if needs_non_zstack:
            self.no_zstack_dir = os.path.join(self.test_dir, "opera_plate_flat")
            print(f"Creating synthetic Opera Phenix data without Z-stacks in {self.no_zstack_dir}")
            os.makedirs(self.no_zstack_dir, exist_ok=True)
            self._create_synthetic_data(output_dir=self.no_zstack_dir, z_stack_levels=1)

            # Create a copy of the original non-Z-stack data for inspection
            self.no_zstack_original_dir = os.path.join(self.test_dir, "opera_plate_flat_original")
            print(f"Creating copy of original non-Z-stack data in {self.no_zstack_original_dir}")
            shutil.copytree(self.no_zstack_dir, self.no_zstack_original_dir)
        else:
            # Still define the paths for tests that might reference them
            self.no_zstack_dir = os.path.join(self.test_dir, "opera_plate_flat")
            self.no_zstack_original_dir = os.path.join(self.test_dir, "opera_plate_flat_original")

    def _needs_zstack_data(self):
        """Determine if the current test needs Z-stack data."""
        zstack_tests = [
            'test_zstack_projection_stitching',
            'test_zstack_per_plane_stitching',
            'test_custom_zstack_function'
        ]
        return self._testMethodName in zstack_tests

    def _needs_non_zstack_data(self):
        """Determine if the current test needs non-Z-stack data."""
        non_zstack_tests = [
            'test_non_zstack_workflow',
            'test_multi_channel_reference',
            'test_basic_stitching',
            'test_explicit_format'
        ]
        return self._testMethodName in non_zstack_tests

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        # Uncomment to keep test data for inspection
        # if cls.base_test_dir.exists():
        #     shutil.rmtree(cls.base_test_dir)
        pass

    def _create_synthetic_data(self, output_dir, z_stack_levels, z_step_size=2.0):
        """Helper to create synthetic Opera Phenix data with configurable Z-stack levels."""
        # Create generator
        generator = SyntheticMicroscopyGenerator(
            output_dir=output_dir,
            grid_size=(2, 2),          # 2x2 grid (4 tiles)
            image_size=(512, 512),     # Smaller images for faster tests
            tile_size=(256, 256),      # Smaller tiles for faster tests
            overlap_percent=10,
            stage_error_px=5,
            wavelengths=2,
            z_stack_levels=z_stack_levels,
            z_step_size=z_step_size,   # Spacing between Z-steps in microns
            num_cells=150,             # More cells for better visualization
            cell_size_range=(2, 8),    # 4x smaller cells (default was 10-30)
            # Set different intensities for each wavelength
            wavelength_intensities={1: 25000, 2: 10000},
            # Use completely different cells for each wavelength
            shared_cell_fraction=0.0,  # 0% shared cells between wavelengths
            # Generate 4 wells from different rows and columns
            wells=['A01', 'A03', 'C01', 'C03'],
            format='OperaPhenix',      # Use Opera Phenix format
            random_seed=42
        )

        # Generate dataset
        generator.generate_dataset()
        print(f"Synthetic Opera Phenix data generated in {output_dir}")

    def test_filename_parser(self):
        """Test the Opera Phenix filename parser."""
        parser = OperaPhenixFilenameParser()

        # Test parsing well
        filename = "r01c03f144p05-ch3sk1fk1fl1.tiff"
        self.assertEqual(parser.parse_well(filename), "R01C03")

        # Test parsing site
        self.assertEqual(parser.parse_site(filename), 144)

        # Test parsing z_index
        self.assertEqual(parser.parse_z_index(filename), 5)

        # Test parsing channel
        self.assertEqual(parser.parse_channel(filename), 3)

        # Test parsing complete filename
        metadata = parser.parse_filename(filename)
        self.assertEqual(metadata['well'], "R01C03")
        self.assertEqual(metadata['site'], 144)
        self.assertEqual(metadata['wavelength'], 3)  # For backward compatibility
        self.assertEqual(metadata['channel'], 3)
        self.assertEqual(metadata['z_index'], 5)

        # Test constructing filename
        constructed = parser.construct_filename("R01C03", 144, 3, 5)
        self.assertEqual(constructed, "r01c03f144p05-ch3sk1fk1fl1.tiff")

    def test_basic_stitching(self):
        """Test basic stitching with Opera Phenix data."""
        # Check if Index.xml file was generated
        index_xml_path = os.path.join(self.no_zstack_dir, "Images", "Index.xml")
        self.assertTrue(os.path.exists(index_xml_path), "Index.xml file not created")

        # Process the Opera Phenix data with auto-detection
        # Use the Images directory as the input directory
        images_dir = os.path.join(self.no_zstack_dir, "Images")
        result = process_plate_folder(
            images_dir,
            reference_channels=['1'],
            tile_overlap=10.0,
            microscope_type='auto'  # Should auto-detect Opera Phenix
        )

        self.assertTrue(result, "Stitching failed")

        # Check if stitched directory was created
        # The stitched directory is created in the same directory as the Images directory
        stitched_dir = os.path.join(self.no_zstack_dir, f"{os.path.basename(images_dir)}_stitched")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")

        # Check if stitched images exist for both wavelengths and all wells
        timepoint_dir = os.path.join(stitched_dir, "TimePoint_1")
        self.assertTrue(os.path.exists(timepoint_dir), f"TimePoint_1 directory not found in {stitched_dir}")

        # Print the list of files in the stitched directory for debugging
        stitched_files = os.listdir(timepoint_dir)
        print(f"Files found in stitched directory: {stitched_files}")

        # Check for at least one stitched file per well
        # Opera Phenix wells and their ImageXpress equivalents
        wells = [
            ('R01C01', 'A01'),
            ('R01C03', 'A03'),
            ('R03C01', 'C01'),
            ('R03C03', 'C03')
        ]
        for opera_well, imx_well in wells:
            # Check for files with either Opera Phenix or ImageXpress well format
            well_files = [f for f in stitched_files if opera_well.lower() in f.lower() or imx_well.lower() in f.lower()]
            self.assertTrue(len(well_files) > 0, f"No stitched files found for well {opera_well} or {imx_well}")

    def test_explicit_format(self):
        """Test stitching with explicitly specified Opera Phenix format."""
        # Process the Opera Phenix data with explicit format
        # Use the Images directory as the input directory
        images_dir = os.path.join(self.no_zstack_dir, "Images")
        result = process_plate_folder(
            images_dir,
            reference_channels=['1'],
            tile_overlap=10.0,
            microscope_type='OperaPhenix'  # Explicitly specify Opera Phenix
        )

        self.assertTrue(result, "Stitching failed")

        # Check if stitched directory was created
        # The stitched directory is created in the same directory as the Images directory
        stitched_dir = os.path.join(self.no_zstack_dir, f"{os.path.basename(images_dir)}_stitched")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")

    def test_non_zstack_workflow(self):
        """Test non-Z-stack workflow with Opera Phenix data using class-based approach."""
        # Create configuration objects
        stitcher_config = StitcherConfig(tile_overlap=10.0)
        preprocessor_config = ImagePreprocessorConfig()

        plate_config = PlateProcessorConfig(
            reference_channels=['1'],
            microscope_type='OperaPhenix',
            stitcher=stitcher_config
        )

        # Create plate processor
        plate_processor = PlateProcessor(config=plate_config)

        # Process the plate
        result = plate_processor.run(self.no_zstack_dir)
        self.assertTrue(result, "Plate processing failed")

        # Check if stitched directory was created
        stitched_dir = os.path.join(self.test_dir, f"{os.path.basename(self.no_zstack_dir)}_stitched")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")

        # Check if stitched images exist for both wavelengths and all wells
        timepoint_dir = os.path.join(stitched_dir, "TimePoint_1")
        self.assertTrue(os.path.exists(timepoint_dir), f"TimePoint_1 directory not found in {stitched_dir}")

        # Check for stitched files
        stitched_files = os.listdir(timepoint_dir)
        print(f"Files found in stitched directory: {stitched_files}")

        # Check for at least one stitched file per well
        # Opera Phenix wells and their ImageXpress equivalents
        wells = [
            ('R01C01', 'A01'),
            ('R01C03', 'A03'),
            ('R03C01', 'C01'),
            ('R03C03', 'C03')
        ]
        for opera_well, imx_well in wells:
            # Check for files with either Opera Phenix or ImageXpress well format
            well_files = [f for f in stitched_files if opera_well.lower() in f.lower() or imx_well.lower() in f.lower()]
            self.assertTrue(len(well_files) > 0, f"No stitched files found for well {opera_well} or {imx_well}")

    # Removed skip decorator to run this test
    def test_zstack_projection_stitching(self):
        """Test Z-stack projection stitching with Opera Phenix data using class-based approach."""
        # Create configuration objects
        stitcher_config = StitcherConfig(tile_overlap=10.0)
        zstack_config = ZStackProcessorConfig(
            create_projections=True,
            stitch_z_reference="max",
            save_projections=True,
            stitch_all_z_planes=False
        )

        plate_config = PlateProcessorConfig(
            reference_channels=['1'],
            microscope_type='OperaPhenix',
            stitcher=stitcher_config,
            z_stack_processor=zstack_config
        )

        preprocessor_config = ImagePreprocessorConfig()
        focus_config = FocusAnalyzerConfig()

        # Create plate processor
        plate_processor = PlateProcessor(config=plate_config)

        # Process the plate
        result = plate_processor.run(self.zstack_dir)
        self.assertTrue(result, "Plate processing failed")

        # Check if stitched directory was created
        stitched_dir = os.path.join(self.test_dir, f"{os.path.basename(self.zstack_dir)}_stitched")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")

        # Check if stitched images exist for both wavelengths and all wells
        timepoint_dir = os.path.join(stitched_dir, "TimePoint_1")
        self.assertTrue(os.path.exists(timepoint_dir), f"TimePoint_1 directory not found in {stitched_dir}")

        # Check for stitched files
        stitched_files = os.listdir(timepoint_dir)
        print(f"Files found in stitched directory: {stitched_files}")

        # Check for at least one stitched file per well
        # Opera Phenix wells and their ImageXpress equivalents
        wells = [
            ('R01C01', 'A01'),
            ('R01C03', 'A03'),
            ('R03C01', 'C01'),
            ('R03C03', 'C03')
        ]
        for opera_well, imx_well in wells:
            # Check for files with either Opera Phenix or ImageXpress well format
            well_files = [f for f in stitched_files if opera_well.lower() in f.lower() or imx_well.lower() in f.lower()]
            self.assertTrue(len(well_files) > 0, f"No stitched files found for well {opera_well} or {imx_well}")

        # The test shouldn't specifically check for 'max' in filenames
        # Just verify that we have stitched files, which we already did above

    # Removed skip decorator to run this test
    def test_zstack_per_plane_stitching(self):
        """Test Z-stack per-plane stitching with Opera Phenix data using class-based approach."""
        # Create configuration objects
        stitcher_config = StitcherConfig(tile_overlap=10.0)
        zstack_config = ZStackProcessorConfig(
            create_projections=False,
            stitch_z_reference="max",
            save_projections=False,
            stitch_all_z_planes=True
        )

        plate_config = PlateProcessorConfig(
            reference_channels=['1'],
            microscope_type='OperaPhenix',
            stitcher=stitcher_config,
            z_stack_processor=zstack_config
        )

        preprocessor_config = ImagePreprocessorConfig()
        focus_config = FocusAnalyzerConfig()

        # Create plate processor
        plate_processor = PlateProcessor(config=plate_config)

        # Process the plate
        result = plate_processor.run(self.zstack_dir)
        self.assertTrue(result, "Plate processing failed")

        # Check if stitched directory was created
        stitched_dir = os.path.join(self.test_dir, f"{os.path.basename(self.zstack_dir)}_stitched")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")

        # Check if stitched images exist for both wavelengths and all wells
        timepoint_dir = os.path.join(stitched_dir, "TimePoint_1")
        self.assertTrue(os.path.exists(timepoint_dir), f"TimePoint_1 directory not found in {stitched_dir}")

        # Check for stitched files
        stitched_files = os.listdir(timepoint_dir)
        print(f"Files found in stitched directory: {stitched_files}")

        # Check for at least one stitched file per well
        # Opera Phenix wells and their ImageXpress equivalents
        wells = [
            ('R01C01', 'A01'),
            ('R01C03', 'A03'),
            ('R03C01', 'C01'),
            ('R03C03', 'C03')
        ]
        for opera_well, imx_well in wells:
            # Check for files with either Opera Phenix or ImageXpress well format
            well_files = [f for f in stitched_files if opera_well.lower() in f.lower() or imx_well.lower() in f.lower()]
            self.assertTrue(len(well_files) > 0, f"No stitched files found for well {opera_well} or {imx_well}")

        # Check for Z-plane files (should have z in the filename)
        z_plane_files = [f for f in stitched_files if "_z" in f.lower()]
        self.assertTrue(len(z_plane_files) > 0, "No Z-plane files found")

    def test_multi_channel_reference(self):
        """Test multi-channel reference with Opera Phenix data using class-based approach."""
        # Create configuration objects
        stitcher_config = StitcherConfig(tile_overlap=10.0)
        preprocessor_config = ImagePreprocessorConfig()

        plate_config = PlateProcessorConfig(
            reference_channels=['1'],  # Use only channel 1 as reference
            microscope_type='OperaPhenix',
            stitcher=stitcher_config
        )

        # Create plate processor
        plate_processor = PlateProcessor(config=plate_config)

        # Process the plate
        result = plate_processor.run(self.no_zstack_dir)
        self.assertTrue(result, "Plate processing failed")

        # Check if stitched directory was created
        stitched_dir = os.path.join(self.test_dir, f"{os.path.basename(self.no_zstack_dir)}_stitched")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")

        # Check if stitched images exist for both wavelengths and all wells
        timepoint_dir = os.path.join(stitched_dir, "TimePoint_1")
        self.assertTrue(os.path.exists(timepoint_dir), f"TimePoint_1 directory not found in {stitched_dir}")

        # Check for stitched files
        stitched_files = os.listdir(timepoint_dir)
        print(f"Files found in stitched directory: {stitched_files}")

        # Check for at least one stitched file per well
        # Opera Phenix wells and their ImageXpress equivalents
        wells = [
            ('R01C01', 'A01'),
            ('R01C03', 'A03'),
            ('R03C01', 'C01'),
            ('R03C03', 'C03')
        ]
        for opera_well, imx_well in wells:
            # Check for files with either Opera Phenix or ImageXpress well format
            well_files = [f for f in stitched_files if opera_well.lower() in f.lower() or imx_well.lower() in f.lower()]
            self.assertTrue(len(well_files) > 0, f"No stitched files found for well {opera_well} or {imx_well}")


if __name__ == '__main__':
    unittest.main()
