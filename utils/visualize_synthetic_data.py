#!/usr/bin/env python3
"""
Generate and visualize synthetic microscopy data.

This script generates synthetic microscopy data and displays it for visual inspection.

Usage:
    python visualize_synthetic_data.py output_dir
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import synthetic data generator
from generate_synthetic_data import SyntheticMicroscopyGenerator


def visualize_dataset(dataset_dir):
    """
    Visualize the generated dataset.
    
    Args:
        dataset_dir: Directory containing the generated dataset
    """
    timepoint_dir = Path(dataset_dir) / "TimePoint_1"
    
    # Find all TIFF files
    tiff_files = sorted(list(timepoint_dir.glob("*.tif")))
    
    if not tiff_files:
        print(f"No TIFF files found in {timepoint_dir}")
        return
    
    # Group files by wavelength and Z level
    files_by_wz = {}
    for file_path in tiff_files:
        filename = file_path.name
        
        # Parse filename to extract wavelength and Z level
        parts = filename.split('_')
        site = parts[1][1:]  # Remove 's' prefix
        
        wavelength = None
        z_level = None
        
        for part in parts:
            if part.startswith('w'):
                wavelength = part[1:]
            elif part.startswith('z'):
                z_level = part[1:].split('.')[0]  # Remove file extension
        
        if z_level is None:
            z_level = '001'  # Default Z level if not specified
        
        key = (wavelength, z_level)
        if key not in files_by_wz:
            files_by_wz[key] = []
        
        files_by_wz[key].append((site, file_path))
    
    # Sort keys for consistent display
    sorted_keys = sorted(files_by_wz.keys())
    
    # Create figure
    num_wavelengths = len(set(k[0] for k in sorted_keys))
    num_z_levels = len(set(k[1] for k in sorted_keys))
    
    fig, axes = plt.subplots(num_z_levels, num_wavelengths, 
                             figsize=(4*num_wavelengths, 4*num_z_levels))
    
    # Handle case with only one subplot
    if num_z_levels == 1 and num_wavelengths == 1:
        axes = np.array([[axes]])
    elif num_z_levels == 1:
        axes = axes.reshape(1, -1)
    elif num_wavelengths == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each wavelength and Z level
    for i, (wavelength, z_level) in enumerate(sorted_keys):
        # Calculate row and column in the subplot grid
        row = int(z_level) - 1
        col = int(wavelength) - 1
        
        # Get the first image for this wavelength and Z level
        _, file_path = files_by_wz[(wavelength, z_level)][0]
        image = tifffile.imread(file_path)
        
        # Display image
        ax = axes[row, col]
        im = ax.imshow(image, cmap='gray')
        ax.set_title(f"W{wavelength}, Z{z_level}")
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = Path(dataset_dir) / "synthetic_data_preview.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Preview saved to {fig_path}")
    
    # Show figure
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate and visualize synthetic microscopy data")
    
    parser.add_argument("output_dir", help="Directory to save generated images")
    parser.add_argument("--grid-size", type=int, nargs=2, default=[2, 2], help="Grid size (rows cols)")
    parser.add_argument("--wavelengths", type=int, default=2, help="Number of wavelength channels")
    parser.add_argument("--z-stack", type=int, default=3, help="Number of Z-stack levels")
    parser.add_argument("--overlap", type=float, default=10.0, help="Percentage of overlap between tiles")
    parser.add_argument("--num-cells", type=int, default=100, help="Number of cells to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    print(f"Generating synthetic microscopy data in {output_dir}")
    generator = SyntheticMicroscopyGenerator(
        output_dir=output_dir,
        grid_size=tuple(args.grid_size),
        image_size=(1024, 1024),
        tile_size=(512, 512),
        overlap_percent=args.overlap,
        stage_error_px=5,
        wavelengths=args.wavelengths,
        z_stack_levels=args.z_stack,
        num_cells=args.num_cells,
        random_seed=args.seed
    )
    
    generator.generate_dataset()
    
    # Visualize the generated data
    print("Visualizing generated data...")
    visualize_dataset(output_dir)


if __name__ == "__main__":
    main()
