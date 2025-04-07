#!/usr/bin/env python3
"""
Test Z-stack workflow for ezstitcher.

This simplified test:
1. Generates synthetic Z-stack and non-Z-stack test data
2. Tests the Z-stack handler functionality
3. Verifies the directory structure and outputs

Usage:
    python -m tests.test_z_stack_workflow
"""

import os
import sys
import shutil
import unittest
from pathlib import Path

# Add parent directory to path so we can import from ezstitcher
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from ezstitcher
from ezstitcher.core.main import modified_process_plate_folder

# Import synthetic data generator
sys.path.append(parent_dir)
from utils.generate_synthetic_data import SyntheticMicroscopyGenerator

class TestZStackWorkflow(unittest.TestCase):
    """Test ezstitcher functionality with synthetic Z-stack and non-Z-stack data."""

    def setUp(self):
        """Set up test environment by generating synthetic data."""
        # Use a designated test directory
        self.test_dir = Path(__file__).resolve().parent / "test_data"

        # Clean up any existing test data
        if os.path.exists(self.test_dir):
            print(f"Cleaning up existing test data directory: {self.test_dir}")
            shutil.rmtree(self.test_dir)

        # Create the test data directory
        self.test_dir.mkdir(exist_ok=True)

        # Create Z-stack data
        self.zstack_dir = self.test_dir / "synthetic_plate"
        print(f"Creating synthetic data with Z-stacks in {self.zstack_dir}")
        os.makedirs(self.zstack_dir, exist_ok=True)
        self._generate_zstack_data()

        # Create non-Z-stack data
        self.no_zstack_dir = self.test_dir / "synthetic_plate_flat"
        print(f"Creating synthetic data without Z-stacks in {self.no_zstack_dir}")
        os.makedirs(self.no_zstack_dir, exist_ok=True)
        self._generate_non_zstack_data()

    def _generate_zstack_data(self):
        """Generate synthetic Z-stack microscopy data."""
        # Simple cell parameters optimized for testing
        wavelength_params = {
            1: {
                'num_cells': 30,
                'cell_size_range': (10, 20),
                'cell_intensity_range': (8000, 20000),
                'background_intensity': 500,
            },
            2: {
                'num_cells': 15,
                'cell_size_range': (5, 15),
                'cell_intensity_range': (3000, 10000),
                'background_intensity': 300,
            }
        }

        # Create a generator with Z-stack levels
        generator = SyntheticMicroscopyGenerator(
            output_dir=self.zstack_dir,
            grid_size=(2, 2),          # 2x2 grid (4 tiles)
            image_size=(384, 384),     # Small images for faster tests
            tile_size=(192, 192),      # Small tiles for faster tests
            overlap_percent=10,
            wavelengths=2,
            z_stack_levels=3,
            wavelength_params=wavelength_params,
            random_seed=42
        )

        # Generate the dataset
        generator.generate_dataset()
        print(f"Z-stack synthetic data generated in {self.zstack_dir}")

    def _generate_non_zstack_data(self):
        """Generate synthetic non-Z-stack microscopy data."""
        # Simple cell parameters optimized for testing
        wavelength_params = {
            1: {
                'num_cells': 30,
                'cell_size_range': (10, 20),
                'cell_intensity_range': (8000, 20000),
                'background_intensity': 500,
            },
            2: {
                'num_cells': 15,
                'cell_size_range': (5, 15),
                'cell_intensity_range': (3000, 10000),
                'background_intensity': 300,
            }
        }

        # Create a generator without Z-stack levels
        generator = SyntheticMicroscopyGenerator(
            output_dir=self.no_zstack_dir,
            grid_size=(2, 2),          # 2x2 grid (4 tiles)
            image_size=(384, 384),     # Small images for faster tests
            tile_size=(192, 192),      # Small tiles for faster tests
            overlap_percent=10,
            wavelengths=2,
            z_stack_levels=1,
            wavelength_params=wavelength_params,
            random_seed=42
        )

        # Generate the dataset
        generator.generate_dataset()
        print(f"Non-Z-stack synthetic data generated in {self.no_zstack_dir}")

    def test_zstack_workflow(self):
        """Test complete Z-stack workflow with synthetic data."""
        print("\nTesting Z-stack workflow...")

        # Verify Z-stack structure exists
        zstep_folder = os.path.join(self.zstack_dir, "TimePoint_1", "ZStep_1")
        self.assertTrue(os.path.exists(zstep_folder),
                       f"Z-stack folder structure not found at {zstep_folder}")

        # Process Z-stack data
        modified_process_plate_folder(
            self.zstack_dir,
            reference_channels=["1"],
            tile_overlap=10,
            focus_detect=True,
            focus_method="combined",
            create_projections=True,
            projection_types=["max", "mean"],
            stitch_z_reference="best_focus"
        )

        # Get the test directory so we can check folders at the correct level
        test_dir = self.test_dir
        plate_name = os.path.basename(self.zstack_dir)

        # Verify output directories were created (at the same level as synthetic_plate)
        best_focus_dir = os.path.join(test_dir, f"{plate_name}_BestFocus")
        projections_dir = os.path.join(test_dir, f"{plate_name}_Projections")
        stitched_dir = os.path.join(test_dir, f"{plate_name}_stitched")

        # Main check is that we correctly create ALL required directories
        self.assertTrue(os.path.exists(best_focus_dir), "BestFocus directory not created")
        self.assertTrue(os.path.exists(projections_dir), "Projections directory not created")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")

        # Verify TimePoint_1 subdirectories exist in each folder
        best_focus_timepoint = os.path.join(best_focus_dir, "TimePoint_1")
        projections_timepoint = os.path.join(projections_dir, "TimePoint_1")
        stitched_timepoint = os.path.join(stitched_dir, "TimePoint_1")

        self.assertTrue(os.path.exists(best_focus_timepoint),
                        "TimePoint_1 not created in BestFocus directory")
        self.assertTrue(os.path.exists(projections_timepoint),
                        "TimePoint_1 not created in Projections directory")
        self.assertTrue(os.path.exists(stitched_timepoint),
                        "TimePoint_1 not created in stitched directory")

        # Verify HTD file was copied to both the root and TimePoint_1 subdirectories
        self.assertTrue(os.path.exists(os.path.join(best_focus_dir, "test_A01.HTD")),
                        "HTD file not copied to BestFocus root directory")
        self.assertTrue(os.path.exists(os.path.join(best_focus_timepoint, "test_A01.HTD")),
                        "HTD file not copied to BestFocus TimePoint_1 directory")

        self.assertTrue(os.path.exists(os.path.join(projections_dir, "test_A01.HTD")),
                        "HTD file not copied to Projections root directory")
        self.assertTrue(os.path.exists(os.path.join(projections_timepoint, "test_A01.HTD")),
                        "HTD file not copied to Projections TimePoint_1 directory")

        # Count files in output directories
        best_focus_files = os.listdir(best_focus_timepoint)
        projection_files = os.listdir(projections_timepoint)

        # Verify files were created
        self.assertGreater(len(best_focus_files), 0, "No best focus files created")
        self.assertGreater(len(projection_files), 0, "No projection files created")

        # Should have both max and mean projections
        max_proj_files = [f for f in projection_files if "maxproj" in f]
        mean_proj_files = [f for f in projection_files if "meanproj" in f]
        self.assertGreater(len(max_proj_files), 0, "No max projection files created")
        self.assertGreater(len(mean_proj_files), 0, "No mean projection files created")

        print(f"  Z-stack workflow test passed")
        print(f"  Best focus files: {len(best_focus_files)}")
        print(f"  Projection files: {len(projection_files)}")

    def test_non_zstack_workflow(self):
        """Test standard workflow with non-Z-stack data."""
        print("\nTesting non-Z-stack workflow...")

        # Verify folder structure (no ZStep folders for non-Z-stack data)
        timepoint_dir = os.path.join(self.no_zstack_dir, "TimePoint_1")
        zstep_folders = [d for d in os.listdir(timepoint_dir) if d.startswith("ZStep_") and
                         os.path.isdir(os.path.join(timepoint_dir, d))]
        self.assertEqual(len(zstep_folders), 0, "Z-stack folders unexpectedly found in non-Z-stack data")

        # Process non-Z-stack data
        modified_process_plate_folder(
            self.no_zstack_dir,
            reference_channels=["1"],
            tile_overlap=10,
            # Include Z-stack parameters that should be ignored for non-Z-stack data
            focus_detect=True,
            create_projections=True
        )

        # For non-Z-stack data, these directories should NOT be created
        best_focus_dir = os.path.join(self.no_zstack_dir, "TimePoint_1_BestFocus")
        projections_dir = os.path.join(self.no_zstack_dir, "TimePoint_1_Projections")

        self.assertFalse(os.path.exists(best_focus_dir),
                        "Best focus directory incorrectly created for non-Z-stack data")
        self.assertFalse(os.path.exists(projections_dir),
                        "Projections directory incorrectly created for non-Z-stack data")

        # Stitched directory should still be created
        stitched_dir = os.path.join(str(self.no_zstack_dir) + "_stitched", "TimePoint_1")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")

        print("  Non-Z-stack workflow test passed")

    def tearDown(self):
        """Clean up after tests."""
        # Keep the test data for now
        # If you want to clean up automatically, uncomment the following:
        # if os.path.exists(self.test_dir):
        #     shutil.rmtree(self.test_dir)
        pass

if __name__ == "__main__":
    unittest.main()