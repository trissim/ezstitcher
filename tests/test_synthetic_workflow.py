#!/usr/bin/env python3
"""
Comprehensive test for ezstitcher using synthetic microscopy data.

This test:
1. Generates synthetic microscopy data with multiple wavelengths and Z-stacks
2. Tests the core functionality of ezstitcher:
   - Z-stack detection and organization
   - Best focus selection
   - Projection creation
   - Composite image creation
   - Stitching with various reference methods

Usage:
    python -m unittest tests/test_synthetic_workflow.py
"""

import unittest
import os
import sys
import shutil
import logging
import numpy as np
from pathlib import Path
import tifffile

# Add parent directory to path so we can import from ezstitcher
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import synthetic data generator
sys.path.append(os.path.join(parent_dir, 'utils'))
from generate_synthetic_data import SyntheticMicroscopyGenerator

# Import core functionality
from ezstitcher.core.stitcher import process_plate_folder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class TestSyntheticWorkflow(unittest.TestCase):
    """Test ezstitcher functionality using synthetic microscopy data."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment with synthetic data."""
        # Create test data directory
        cls.test_dir = Path(os.path.join(parent_dir, 'tests', 'test_data'))

        # Clean up any existing test data
        if cls.test_dir.exists():
            print(f"Cleaning up existing test data directory: {cls.test_dir}")
            shutil.rmtree(cls.test_dir)

        # Create the test data directory
        cls.test_dir.mkdir(exist_ok=True)

        # Create synthetic Z-stack data
        cls.zstack_dir = os.path.join(cls.test_dir, "synthetic_plate")
        print(f"Creating synthetic data with Z-stacks in {cls.zstack_dir}")
        os.makedirs(cls.zstack_dir, exist_ok=True)
        cls._create_synthetic_data(output_dir=cls.zstack_dir, z_stack_levels=3)

        # Create synthetic non-Z-stack data
        cls.no_zstack_dir = os.path.join(cls.test_dir, "synthetic_plate_flat")
        print(f"Creating synthetic data without Z-stacks in {cls.no_zstack_dir}")
        os.makedirs(cls.no_zstack_dir, exist_ok=True)
        cls._create_synthetic_data(output_dir=cls.no_zstack_dir, z_stack_levels=1)

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        # Uncomment to keep test data for inspection
        # if cls.test_dir.exists():
        #     shutil.rmtree(cls.test_dir)
        pass

    @classmethod
    def _create_synthetic_data(cls, output_dir, z_stack_levels):
        """Helper to create synthetic data with configurable Z-stack levels."""
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
            num_cells=100,
            random_seed=42
        )

        # Generate dataset
        generator.generate_dataset()
        print(f"Synthetic data generated in {output_dir}")

    def test_comprehensive_workflow(self):
        """Test comprehensive workflow with process_plate_folder."""
        print("\nTesting comprehensive workflow with process_plate_folder...")

        # Test Z-stack workflow
        print("\n1. Testing Z-stack workflow...")

        # Process Z-stack data with all features enabled
        process_plate_folder(
            self.zstack_dir,
            reference_channels=["1"],
            tile_overlap=10,
            max_shift=20,
            focus_detect=True,
            focus_method="combined",
            create_projections=True,
            projection_types=["max", "mean"],
            stitch_z_reference="best_focus"
        )

        # Check if best focus directory was created
        best_focus_dir = os.path.join(self.test_dir, f"{os.path.basename(self.zstack_dir)}_BestFocus")
        self.assertTrue(os.path.exists(best_focus_dir), "Best focus directory not created")

        # Check if best focus images were created
        timepoint_dir = os.path.join(best_focus_dir, "TimePoint_1")
        self.assertTrue(os.path.exists(timepoint_dir), f"TimePoint_1 directory not found in {best_focus_dir}")

        # Check if best focus images exist for all sites
        for site in range(1, 5):  # 2x2 grid = 4 sites
            filename = f"A01_s{site:03d}_w1.tif"
            file_path = os.path.join(timepoint_dir, filename)
            self.assertTrue(os.path.exists(file_path), f"Best focus image not found: {file_path}")

        # Check if projections directory was created
        projections_dir = os.path.join(self.test_dir, f"{os.path.basename(self.zstack_dir)}_Projections")
        self.assertTrue(os.path.exists(projections_dir), "Projections directory not created")

        # Check if projection images were created
        timepoint_dir = os.path.join(projections_dir, "TimePoint_1")
        self.assertTrue(os.path.exists(timepoint_dir), f"TimePoint_1 directory not found in {projections_dir}")

        # Check if projection images exist for all sites, wavelengths, and projection types
        for site in range(1, 5):  # 2x2 grid = 4 sites
            for wavelength in [1, 2]:
                for proj_type in ["max", "mean"]:
                    filename = f"A01_s{site:03d}_w{wavelength}_{proj_type}proj.tif"
                    file_path = os.path.join(timepoint_dir, filename)
                    self.assertTrue(os.path.exists(file_path), f"Projection image not found: {file_path}")

        # Check if stitched directory was created
        stitched_dir = os.path.join(self.test_dir, f"{os.path.basename(best_focus_dir)}_stitched")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")

        # Test non-Z-stack workflow
        print("\n2. Testing non-Z-stack workflow...")

        # Process non-Z-stack data
        process_plate_folder(
            self.no_zstack_dir,
            reference_channels=["1"],
            tile_overlap=10,
            max_shift=20
        )

        # Check if stitched directory was created
        stitched_dir = os.path.join(self.test_dir, f"{os.path.basename(self.no_zstack_dir)}_stitched")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")

        # Test multi-channel reference
        print("\n3. Testing multi-channel reference...")

        # Process with multiple reference channels
        process_plate_folder(
            self.no_zstack_dir,
            reference_channels=["1", "2"],
            composite_weights={"1": 0.3, "2": 0.7},
            tile_overlap=10,
            max_shift=20
        )

        # Check if processed directory was created
        processed_dir = os.path.join(self.test_dir, f"{os.path.basename(self.no_zstack_dir)}_processed")
        self.assertTrue(os.path.exists(processed_dir), "Processed directory not created")

if __name__ == "__main__":
    unittest.main()
