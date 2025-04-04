#!/usr/bin/env python3
"""
Test ezstitcher with synthetic microscopy data.

This script:
1. Generates synthetic microscopy data (both Z-stack and non-Z-stack)
2. Tests the Z-stack handling and stitching functionality
3. Validates that correct output folders are created

Usage:
    python test_with_synthetic_data.py
"""

import os
import sys
import shutil
import unittest
import numpy as np
from pathlib import Path
import tifffile

# Add parent directory to path so we can import from ezstitcher
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from ezstitcher
from ezstitcher.core.z_stack_handler import modified_process_plate_folder

# Import synthetic data generator
sys.path.append(os.path.join(parent_dir, 'utils'))
from generate_synthetic_data import SyntheticMicroscopyGenerator

class TestWithSyntheticData(unittest.TestCase):
    def setUp(self):
        """Set up test environment with both Z-stack and non-Z-stack synthetic data."""
        # Use a persistent directory in the project folder
        project_dir = Path(__file__).resolve().parent.parent
        self.test_dir = project_dir / "tests/test_data"
        
        # Always clean up the test_dir to ensure a fresh start
        if os.path.exists(self.test_dir):
            print(f"Removing existing test data directory: {self.test_dir}")
            shutil.rmtree(self.test_dir)
        
        # Create the test data directory
        self.test_dir.mkdir(exist_ok=True)
        
        # Create synthetic Z-stack data
        self.zstack_dir = os.path.join(self.test_dir, "synthetic_zstack")
        print(f"Creating synthetic data with Z-stacks in {self.zstack_dir}")
        os.makedirs(self.zstack_dir, exist_ok=True)
        self._create_synthetic_data(output_dir=self.zstack_dir, z_stack_levels=3)
        
        # Create synthetic non-Z-stack data
        self.no_zstack_dir = os.path.join(self.test_dir, "synthetic_no_zstack")
        print(f"Creating synthetic data without Z-stacks in {self.no_zstack_dir}")
        os.makedirs(self.no_zstack_dir, exist_ok=True)
        self._create_synthetic_data(output_dir=self.no_zstack_dir, z_stack_levels=1)
    
    def _create_synthetic_data(self, output_dir, z_stack_levels):
        """Helper to create synthetic data with configurable Z-stack levels."""
        wavelength_params = {
            1: {
                'num_cells': 50, 
                'cell_size_range': (15, 30),
                'cell_intensity_range': (8000, 20000),
                'background_intensity': 500,
            },
            2: {
                'num_cells': 20,
                'cell_size_range': (5, 15),
                'cell_intensity_range': (3000, 10000),
                'background_intensity': 300,
            }
        }
        
        generator = SyntheticMicroscopyGenerator(
            output_dir=output_dir,
            grid_size=(2, 2),          # 2x2 grid (4 tiles)
            image_size=(512, 512),     # Smaller images for faster tests
            tile_size=(256, 256),      # Smaller tiles for faster tests
            overlap_percent=10,
            wavelengths=2,
            z_stack_levels=z_stack_levels,
            wavelength_params=wavelength_params,
            random_seed=42
        )
        
        generator.generate_dataset()
        print(f"Synthetic data generated in {output_dir}")
        
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
        
        # Verify output directories were created
        best_focus_dir = os.path.join(self.zstack_dir, "TimePoint_1_BestFocus")
        projections_dir = os.path.join(self.zstack_dir, "TimePoint_1_Projections")
        
        # Main check is that we correctly create ALL required directories
        self.assertTrue(os.path.exists(best_focus_dir), "Best focus directory not created")
        self.assertTrue(os.path.exists(projections_dir), "Projections directory not created")
        
        # Check if stitched directory was created (either main location or BestFocus location)
        stitched_dir = os.path.join(self.zstack_dir + "_stitched", "TimePoint_1")
        bestfocus_stitched_dir = os.path.join(self.zstack_dir, "TimePoint_1_BestFocus_stitched", "TimePoint_1")
        
        stitched_exists = os.path.exists(stitched_dir) or os.path.exists(bestfocus_stitched_dir)
        self.assertTrue(stitched_exists, "Stitched directory not created")
        
        # Count files in output directories
        best_focus_files = os.listdir(best_focus_dir)
        projection_files = os.listdir(projections_dir)
        
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
        zstep_folders = [d for d in os.listdir(timepoint_dir) if d.startswith("ZStep_")]
        self.assertEqual(len(zstep_folders), 0, "Z-stack folders unexpectedly found in non-Z-stack data")
        
        # Process non-Z-stack data using modified_process_plate_folder
        modified_process_plate_folder(
            self.no_zstack_dir,
            reference_channels=["1"],
            tile_overlap=10,
            # Include Z-stack parameters that should be ignored
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
        stitched_dir = os.path.join(self.no_zstack_dir + "_stitched", "TimePoint_1")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")
        
        print("  Non-Z-stack workflow test passed")
        
if __name__ == "__main__":
    unittest.main()
