"""
Integration tests for the ezstitcher package.

These tests verify that the core components work together correctly.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
import tifffile

from ezstitcher.core.stitcher import process_plate_folder
from ezstitcher.core.z_stack_handler import modified_process_plate_folder
from ezstitcher.core.image_process import process_bf


class TestIntegration(unittest.TestCase):
    def create_test_htd_file(self, folder_path, well_name, grid_size_x, grid_size_y):
        """Create a fake HTD file for testing."""
        htd_content = f"""HTD,1.0
Description,Test HTD file for {well_name}
Wells,{well_name}
GridSizeX,{grid_size_x}
GridSizeY,{grid_size_y}
SiteSelection,{well_name},"""
        
        # Add site selection rows (all set to True)
        for y in range(grid_size_y):
            row = []
            for x in range(grid_size_x):
                row.append("True")
            htd_content += "\nSiteSelection," + ",".join(row)
        
        # Write the HTD file
        htd_file = os.path.join(folder_path, f"test_{well_name}.HTD")
        with open(htd_file, 'w') as f:
            f.write(htd_content)
        
        return htd_file
    
    def setUp(self):
        """Set up test environment with temporary directory structure."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create a plate folder structure
        self.plate_dir = os.path.join(self.test_dir, "test_plate")
        self.timepoint_dir = os.path.join(self.plate_dir, "TimePoint_1")
        os.makedirs(self.timepoint_dir, exist_ok=True)
        
        # Create sample image data (small 50x50 images)
        self.img_size = 50
        
        # Create a 2x2 grid of images for well A01
        self.grid_size = 2
        self.well_name = "A01"
        
        # Create HTD file for grid detection
        self.htd_file = self.create_test_htd_file(
            self.plate_dir, 
            self.well_name, 
            self.grid_size, 
            self.grid_size
        )
        
        # Create images for two wavelengths
        for wavelength in [1, 2]:
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    # Create a unique pattern for each site
                    img = np.ones((self.img_size, self.img_size), dtype=np.uint16) * 1000
                    
                    # Add a unique pattern to identify the site
                    site_id = y * self.grid_size + x + 1
                    
                    # Add a square in the center with intensity based on site
                    center = self.img_size // 2
                    size = self.img_size // 4
                    img[center-size:center+size, center-size:center+size] = 5000 * site_id
                    
                    # Add wavelength-specific intensity
                    if wavelength == 2:
                        img = img * 1.5
                    
                    # Save the image
                    filename = f"{self.well_name}_s{site_id:03d}_w{wavelength}.tif"
                    file_path = os.path.join(self.timepoint_dir, filename)
                    tifffile.imwrite(file_path, img)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_basic_stitching(self):
        """Test basic stitching functionality."""
        # Process the plate folder
        process_plate_folder(
            self.plate_dir,
            reference_channels=["1"],
            tile_overlap=10,
            max_shift=10
        )
        
        # Check if stitched images were created
        stitched_dir = os.path.join(self.plate_dir + "_stitched", "TimePoint_1")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")
        
        # Check if stitched images exist for both wavelengths
        for wavelength in [1, 2]:
            stitched_file = f"A01_w{wavelength}.tif"
            stitched_path = os.path.join(stitched_dir, stitched_file)
            self.assertTrue(os.path.exists(stitched_path), f"Stitched file {stitched_file} not created")
            
            # Load the stitched image and check its properties
            stitched_img = tifffile.imread(stitched_path)
            
            # The stitched image should be larger than the individual tiles
            # With 2x2 grid and 10% overlap, size should be approximately 1.9x the original
            expected_size = int(self.img_size * 1.9)
            self.assertGreaterEqual(stitched_img.shape[0], expected_size)
            self.assertGreaterEqual(stitched_img.shape[1], expected_size)

    def test_preprocessing_and_stitching(self):
        """Test stitching with preprocessing functions."""
        # Process the plate folder with preprocessing
        process_plate_folder(
            self.plate_dir,
            reference_channels=["1"],
            preprocessing_funcs={"1": process_bf},
            tile_overlap=10,
            max_shift=10
        )
        
        # Check if processed directory was created
        processed_dir = os.path.join(self.plate_dir, "processed")
        self.assertTrue(os.path.exists(processed_dir), "Processed directory not created")
        
        # Check if stitched images were created
        stitched_dir = os.path.join(self.plate_dir + "_stitched", "TimePoint_1")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")
        
        # Check if stitched images exist for both wavelengths
        for wavelength in [1, 2]:
            stitched_file = f"A01_w{wavelength}.tif"
            stitched_path = os.path.join(stitched_dir, stitched_file)
            self.assertTrue(os.path.exists(stitched_path), f"Stitched file {stitched_file} not created")

    def test_multi_channel_reference(self):
        """Test stitching with multiple reference channels."""
        # Process the plate folder with multiple reference channels
        process_plate_folder(
            self.plate_dir,
            reference_channels=["1", "2"],
            composite_weights={"1": 0.3, "2": 0.7},
            tile_overlap=10,
            max_shift=10
        )
        
        # Check if composite directory was created
        composite_dir = os.path.join(self.plate_dir, "composite")
        self.assertTrue(os.path.exists(composite_dir), "Composite directory not created")
        
        # Check if stitched images were created
        stitched_dir = os.path.join(self.plate_dir + "_stitched", "TimePoint_1")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")
        
        # Check if stitched images exist for both wavelengths and composite
        for wavelength in [1, 2, "composite"]:
            stitched_file = f"A01_w{wavelength}.tif"
            stitched_path = os.path.join(stitched_dir, stitched_file)
            self.assertTrue(os.path.exists(stitched_path), f"Stitched file {stitched_file} not created")


class TestZStackIntegration(unittest.TestCase):
    def create_test_htd_file(self, folder_path, well_name, grid_size_x, grid_size_y):
        """Create a fake HTD file for testing."""
        htd_content = f"""HTD,1.0
Description,Test HTD file for {well_name}
Wells,{well_name}
GridSizeX,{grid_size_x}
GridSizeY,{grid_size_y}
SiteSelection,{well_name},"""
        
        # Add site selection rows (all set to True)
        for y in range(grid_size_y):
            row = []
            for x in range(grid_size_x):
                row.append("True")
            htd_content += "\nSiteSelection," + ",".join(row)
        
        # Write the HTD file
        htd_file = os.path.join(folder_path, f"test_{well_name}.HTD")
        with open(htd_file, 'w') as f:
            f.write(htd_content)
        
        return htd_file
        
    def setUp(self):
        """Set up test environment with Z-stack images."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create a plate folder structure
        self.plate_dir = os.path.join(self.test_dir, "test_plate")
        self.timepoint_dir = os.path.join(self.plate_dir, "TimePoint_1")
        os.makedirs(self.timepoint_dir, exist_ok=True)
        
        # Create sample image data (small 50x50 images)
        self.img_size = 50
        
        # Create a 2x2 grid of images for well A01
        self.grid_size = 2
        self.well_name = "A01"
        
        # Create Z-stack with 3 levels
        self.z_levels = 3
        
        # Create HTD file for grid detection
        self.htd_file = self.create_test_htd_file(
            self.plate_dir, 
            self.well_name, 
            self.grid_size, 
            self.grid_size
        )
        
        # Create images for two wavelengths
        for wavelength in [1, 2]:
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    # Create a unique pattern for each site
                    site_id = y * self.grid_size + x + 1
                    
                    # Create Z-stack images with different focus levels
                    for z in range(1, self.z_levels + 1):
                        img = np.ones((self.img_size, self.img_size), dtype=np.uint16) * 1000
                        
                        # Add a square in the center with intensity based on site
                        center = self.img_size // 2
                        size = self.img_size // 4
                        
                        # Make the middle Z-plane (z=2) the sharpest (highest contrast)
                        if z == 2:
                            # Sharp image (high contrast)
                            img[center-size:center+size, center-size:center+size] = 10000 * site_id
                        else:
                            # Blurry image (lower contrast)
                            img[center-size:center+size, center-size:center+size] = 3000 * site_id
                        
                        # Add wavelength-specific intensity
                        if wavelength == 2:
                            img = img * 1.5
                        
                        # Save the image
                        filename = f"{self.well_name}_s{site_id:03d}_w{wavelength}_z{z:03d}.tif"
                        file_path = os.path.join(self.timepoint_dir, filename)
                        tifffile.imwrite(file_path, img)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_zstack_processing(self):
        """Test Z-stack processing and focus detection."""
        # Process the plate folder with Z-stack handling
        modified_process_plate_folder(
            self.plate_dir,
            reference_channels=["1"],
            tile_overlap=10,
            max_shift=10,
            focus_detect=True,
            focus_method="combined",
            create_projections=True,
            projection_types=["max", "mean"],
            stitch_z_reference="best_focus"
        )
        
        # Check if best focus directory was created (matches the directory name used in our implementation)
        best_focus_dir = os.path.join(self.plate_dir, "TimePoint_1_BestFocus")
        self.assertTrue(os.path.exists(best_focus_dir), "Best focus directory not created")
        
        # Check if projections directory was created
        projections_dir = os.path.join(self.plate_dir, "TimePoint_1_Projections")
        self.assertTrue(os.path.exists(projections_dir), "Projections directory not created")
        
        # Check if stitched images were created
        stitched_dir = os.path.join(self.plate_dir + "_stitched", "TimePoint_1")
        self.assertTrue(os.path.exists(stitched_dir), "Stitched directory not created")
        
        # Check if best focus images exist for both wavelengths
        for wavelength in [1, 2]:
            # Check for best focus images
            bf_file = f"{self.well_name}_s001_w{wavelength}.tif"  # First tile's best focus image
            bf_path = os.path.join(best_focus_dir, bf_file)
            self.assertTrue(os.path.exists(bf_path), f"Best focus file {bf_file} not created")
            
            # Check for projection images
            for proj_type in ["max", "mean"]:
                proj_file = f"{self.well_name}_s001_w{wavelength}_{proj_type}proj.tif"
                proj_path = os.path.join(projections_dir, proj_file)
                self.assertTrue(os.path.exists(proj_path), f"Projection file {proj_file} not created")


if __name__ == '__main__':
    unittest.main()
