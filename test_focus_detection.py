#!/usr/bin/env python3
"""
Test script for focus detection on a folder of Z-stack images.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import focus detection functionality
from ezstitcher.core.focus_detect import (
    combined_focus_measure,
    find_best_focus,
    tenengrad_variance,
    normalized_variance,
    laplacian_energy,
    adaptive_fft_focus
)

# Import Z-stack handler for handling focus detection across stacks
from ezstitcher.core.z_stack_handler import (
    detect_zstack_images,
    load_image_stack,
    find_best_focus_in_stack,
    create_best_focus_images
)

def display_focus_scores(image_files, scores, best_idx, method_name, output_dir=None):
    """Display focus scores and best image."""
    # Plot focus scores
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(scores, 'o-')
    plt.axvline(x=best_idx, color='r', linestyle='--')
    plt.title(f'Focus Scores ({method_name})')
    plt.xlabel('Image Index')
    plt.ylabel('Focus Score')
    plt.grid(True)
    
    # Display best image
    plt.subplot(1, 2, 2)
    best_img_path = image_files[best_idx]
    best_img = cv2.imread(str(best_img_path))
    if best_img is not None:
        best_img = cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB)
        plt.imshow(best_img)
        plt.title(f'Best Focus Image: {Path(best_img_path).name}')
        plt.axis('off')
    else:
        plt.text(0.5, 0.5, 'Could not load image', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save or display
    if output_dir:
        output_path = os.path.join(output_dir, f'focus_scores_{method_name}.png')
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

def test_focus_methods(input_dir, output_dir=None, methods=None):
    """
    Test different focus detection methods on a folder of images.
    
    Args:
        input_dir: Directory containing Z-stack images
        output_dir: Directory to save results (if None, will display plots)
        methods: List of focus methods to test, defaults to all
    """
    if methods is None:
        methods = ['combined', 'tenengrad', 'laplacian', 'normalized_variance', 'adaptive_fft']
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in ['.tif', '.TIF', '.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']:
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Load images
    images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
            print(f"Loaded {img_path.name}: {img.shape}")
        else:
            print(f"Failed to load {img_path}")
    
    if not images:
        print("No valid images loaded")
        return
    
    # Test each method
    results = {}
    for method in methods:
        print(f"\nTesting method: {method}")
        
        if method == 'combined':
            # Calculate scores for each image individually
            scores = []
            for img in images:
                score = combined_focus_measure(img)
                scores.append(score)
            best_idx = np.argmax(scores)
        elif method == 'tenengrad':
            scores = []
            for img in images:
                score = tenengrad_variance(img)
                scores.append(score)
            best_idx = np.argmax(scores)
        elif method == 'laplacian':
            scores = []
            for img in images:
                score = laplacian_energy(img)
                scores.append(score)
            best_idx = np.argmax(scores)
        elif method == 'normalized_variance':
            scores = []
            for img in images:
                score = normalized_variance(img)
                scores.append(score)
            best_idx = np.argmax(scores)
        elif method == 'adaptive_fft':
            scores = []
            for img in images:
                score = adaptive_fft_focus(img)
                scores.append(score)
            best_idx = np.argmax(scores)
        else:
            # Use the find_best_focus function directly
            best_idx, focus_scores = find_best_focus(images, method=method)
            scores = [score[1] for score in focus_scores]
        
        results[method] = {
            'best_idx': best_idx,
            'best_image': image_files[best_idx].name,
            'scores': scores
        }
        
        print(f"Best focus image for '{method}': {image_files[best_idx].name} (index {best_idx})")
        
        # Display or save results
        display_focus_scores(image_files, scores, best_idx, method, output_dir)
    
    # Save results to a text file if output_dir is specified
    if output_dir:
        with open(os.path.join(output_dir, 'focus_results.txt'), 'w') as f:
            f.write(f"Results for {len(image_files)} images in {input_dir}\n\n")
            for method, result in results.items():
                f.write(f"Method: {method}\n")
                f.write(f"Best image: {result['best_image']} (index {result['best_idx']})\n")
                f.write(f"Scores: {', '.join(map(str, result['scores']))}\n\n")

def test_select_best_focused_images(input_dir, output_dir, focus_method='combined'):
    """
    Test Z-stack best focus selection and create images in the output directory.
    
    Args:
        input_dir: Directory containing Z-stack images
        output_dir: Directory to save best focused images
        focus_method: Focus detection method to use
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if directory contains Z-stack images
    has_zstack, z_indices_map = detect_zstack_images(input_dir)
    
    if not has_zstack:
        print(f"No Z-stack images detected in {input_dir}")
        return
        
    # Create best focus images
    print(f"Finding best focused images in {input_dir} using method '{focus_method}'")
    best_focus_results = create_best_focus_images(
        input_dir,
        output_dir,
        focus_wavelength='all',
        focus_method=focus_method
    )
    
    # Report results
    if best_focus_results:
        print(f"Created {len(best_focus_results)} best focus images in {output_dir}")
        for img_coords, best_z in best_focus_results.items():
            print(f"Image at {img_coords}: selected z={best_z}")
    else:
        print("No best focused images created")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test focus detection on a folder of Z-stack images")
    parser.add_argument("input_dir", help="Directory containing Z-stack images")
    parser.add_argument("--output-dir", help="Directory to save results (optional)")
    parser.add_argument("--mode", choices=["methods", "select"], default="methods",
                       help="Test mode: compare methods or select best images")
    parser.add_argument("--method", default="combined", 
                       help="Focus detection method to use (for --mode=select)")
    
    args = parser.parse_args()
    
    if args.mode == "methods":
        test_focus_methods(args.input_dir, args.output_dir)
    else:
        if not args.output_dir:
            args.output_dir = os.path.join(args.input_dir, "best_focus")
        test_select_best_focused_images(args.input_dir, args.output_dir, args.method)