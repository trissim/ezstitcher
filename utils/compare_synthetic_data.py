#!/usr/bin/env python3
"""
Compare synthetic microscopy images from before and after the changes.

This script:
1. Loads images from both the original and new synthetic data
2. Displays them side by side
3. Computes basic statistics for comparison
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile

# Add parent directory to path so we can import from ezstitcher
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def load_and_compare_images(original_path, new_path):
    """Load and compare two images."""
    original_img = tifffile.imread(original_path)
    new_img = tifffile.imread(new_path)
    
    # Compute statistics
    original_mean = np.mean(original_img)
    original_std = np.std(original_img)
    original_min = np.min(original_img)
    original_max = np.max(original_img)
    
    new_mean = np.mean(new_img)
    new_std = np.std(new_img)
    new_min = np.min(new_img)
    new_max = np.max(new_img)
    
    print(f"Original image: mean={original_mean:.2f}, std={original_std:.2f}, min={original_min}, max={original_max}")
    print(f"New image:      mean={new_mean:.2f}, std={new_std:.2f}, min={new_min}, max={new_max}")
    
    # Plot images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Use the same color scale for both images
    vmin = min(original_min, new_min)
    vmax = max(original_max, new_max)
    
    ax1.imshow(original_img, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title(f"Original - {Path(original_path).name}")
    ax1.axis('off')
    
    ax2.imshow(new_img, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title(f"New (wavelength-specific) - {Path(new_path).name}")
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Create output directory
    output_dir = Path(parent_dir) / "tests" / "comparison_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save figure
    filename = f"comparison_{Path(original_path).name.replace('.tif', '')}.png"
    output_path = output_dir / filename
    plt.savefig(output_path)
    print(f"Saved comparison to {output_path}")
    
    plt.close()

def main():
    # Define paths
    project_dir = Path(parent_dir)
    original_dir = project_dir / "tests" / "reference_data" / "synthetic_plate_original" / "TimePoint_1"
    new_dir = project_dir / "tests" / "test_data" / "synthetic_plate" / "TimePoint_1"
    
    # Compare a Z-stack image from wavelength 1
    original_w1 = original_dir / "A01_s001_w1_z002.tif"
    new_w1 = new_dir / "A01_s001_w1_z002.tif"
    
    print("\nComparing wavelength 1 images:")
    load_and_compare_images(original_w1, new_w1)
    
    # Compare a Z-stack image from wavelength 2
    original_w2 = original_dir / "A01_s001_w2_z002.tif"
    new_w2 = new_dir / "A01_s001_w2_z002.tif"
    
    print("\nComparing wavelength 2 images:")
    load_and_compare_images(original_w2, new_w2)
    
    print("\nComparison complete. Check the tests/comparison_output directory for visual comparisons.")

if __name__ == "__main__":
    main()