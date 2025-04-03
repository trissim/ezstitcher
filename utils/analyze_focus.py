#!/usr/bin/env python3
"""
Utility script for analyzing focus quality in microscopy images.
"""

import os
import sys
import argparse
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path so we can import from ezstitcher
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ezstitcher.core.focus_detect import (
    tenengrad_variance,
    normalized_variance,
    laplacian_energy,
    find_best_focus
)

# Define our own extract_z_index function
def extract_z_index(filename):
    """
    Extract Z-index from a filename.
    
    Args:
        filename (str): Filename including _z{zzz} pattern
        
    Returns:
        int or None: Z-index if found, None otherwise
    """
    match = re.search(r'_z(\d{3})', filename)
    if match:
        return int(match.group(1))
    return None

def analyze_z_series(file_pattern, output_file=None, methods=None, z_pattern=r'_z(\d{3})'):
    """
    Analyze a series of Z-stack images and find the best focused one.
    
    Args:
        file_pattern: Glob pattern to match image files
        output_file: File to save visualization (if None, displays plot)
        methods: List of focus detection methods to use
        z_pattern: Regular expression pattern to extract Z-index
    """
    if methods is None:
        methods = ["combined", "tenengrad", "laplacian", "normalized_variance"]
    
    # Find matching files
    image_paths = sorted(glob.glob(file_pattern))
    if not image_paths:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Extract Z-indices if possible
    z_indices = []
    z_regex = re.compile(z_pattern)
    for path in image_paths:
        filename = os.path.basename(path)
        match = z_regex.search(filename)
        if match:
            z_indices.append(int(match.group(1)))
        else:
            # If no z-index found, use the file's position in the sorted list
            z_indices.append(len(z_indices))
    
    # Load images
    images = []
    valid_paths = []
    first_shape = None
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            # Check if this is the first image or has the same shape as the first image
            if first_shape is None:
                first_shape = img.shape
                images.append(img)
                valid_paths.append(path)
                print(f"Loaded {os.path.basename(path)}: {img.shape}")
            elif img.shape == first_shape:
                images.append(img)
                valid_paths.append(path)
                print(f"Loaded {os.path.basename(path)}: {img.shape}")
            else:
                print(f"Skipping {os.path.basename(path)} due to size mismatch: {img.shape} vs {first_shape}")
        else:
            print(f"Failed to load {path}")
    
    if not images:
        print("No valid images loaded")
        return
    
    # Update image_paths to only include paths of valid images
    image_paths = valid_paths
    
    # Update z_indices to match the valid images
    if z_indices:
        z_indices = z_indices[:len(images)]
    
    # Calculate focus measures
    measures = {}
    for method in methods:
        # Calculating focus measures for each method
        try:
            # Use find_best_focus function
            _, best_idx, best_z, all_scores = find_best_focus(images, method=method)
            
            # Get scores based on method
            if isinstance(all_scores[0], dict):
                # If all_scores contains dictionaries
                scores = [score.get('combined', 0) for score in all_scores]
            elif isinstance(all_scores[0], tuple):
                # If all_scores contains tuples (idx, score)
                scores = [score[1] for score in all_scores]
            else:
                # Otherwise just use the scores directly
                scores = all_scores
                
            measures[method] = {
                'scores': scores,
                'best_idx': best_idx,
                'best_z': best_z if best_z is not None else z_indices[best_idx] if z_indices else best_idx
            }
        except Exception as e:
            print(f"Error with method '{method}': {e}")
            continue
        
        print(f"Method '{method}': Best focus at z={best_z} (index {best_idx})")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot focus measures
    plt.subplot(2, 2, 1)
    for method, data in measures.items():
        if z_indices:
            plt.plot(z_indices, data['scores'], 'o-', label=method)
            plt.axvline(x=data['best_z'], color='r', linestyle='--', alpha=0.5)
        else:
            plt.plot(data['scores'], 'o-', label=method)
            plt.axvline(x=data['best_idx'], color='r', linestyle='--', alpha=0.5)
    
    plt.title('Focus Measures by Z-position')
    plt.xlabel('Z-position')
    plt.ylabel('Focus Score')
    plt.legend()
    plt.grid(True)
    
    # Show best focused image
    plt.subplot(2, 2, 2)
    best_method = methods[0]  # Use the first method as default
    best_idx = measures[best_method]['best_idx']
    best_img = images[best_idx]
    plt.imshow(cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB))
    best_z = measures[best_method]['best_z'] if measures[best_method]['best_z'] is not None else best_idx
    plt.title(f'Best Focus (z={best_z})')
    plt.axis('off')
    
    # Show worst focused image (lowest score)
    plt.subplot(2, 2, 3)
    worst_idx = np.argmin(measures[best_method]['scores'])
    worst_img = images[worst_idx]
    plt.imshow(cv2.cvtColor(worst_img, cv2.COLOR_BGR2RGB))
    worst_z = z_indices[worst_idx] if z_indices else worst_idx
    plt.title(f'Worst Focus (z={worst_z})')
    plt.axis('off')
    
    # Add info about the images
    plt.subplot(2, 2, 4)
    plt.axis('off')
    info_text = f"File pattern: {file_pattern}\n"
    info_text += f"Number of images: {len(images)}\n\n"
    
    for method, data in measures.items():
        info_text += f"Method: {method}\n"
        info_text += f"Best focus: z={data['best_z']} (index {data['best_idx']})\n"
        info_text += f"Score: {data['scores'][data['best_idx']]:.4f}\n\n"
    
    plt.text(0.05, 0.95, info_text, va='top', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Results saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze focus quality in Z-stack images")
    
    parser.add_argument("file_pattern", help="Glob pattern to match image files (e.g., '/path/to/images/*.tif')")
    parser.add_argument("--output", "-o", help="Output file to save visualization (if not specified, displays plot)")
    parser.add_argument("--methods", "-m", nargs="+", default=["combined", "tenengrad", "laplacian", "normalized_variance"],
                       help="Focus detection methods to use")
    parser.add_argument("--z-pattern", default=r'_z(\d{3})',
                       help="Regular expression pattern to extract Z-index from filenames")
    
    args = parser.parse_args()
    
    analyze_z_series(args.file_pattern, args.output, args.methods, args.z_pattern)