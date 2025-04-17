"""
Example demonstrating how to use percentile normalization for Z-stack processing.

This example shows how to:
1. Load a Z-stack of images
2. Apply global percentile normalization across the entire stack
3. Create a maximum intensity projection from the normalized stack
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile

from ezstitcher.core.image_preprocessor import ImagePreprocessor

def percentile_normalized_projection(z_stack_folder, output_folder, low_percentile=2, high_percentile=98):
    """
    Create a percentile-normalized projection of a Z-stack.
    
    This function normalizes the entire stack using percentile-based contrast stretching,
    then creates a maximum intensity projection.
    
    Args:
        z_stack_folder (str): Path to folder containing Z-stack images
        output_folder (str): Path to save the output projection
        low_percentile (float): Lower percentile for normalization (0-100)
        high_percentile (float): Upper percentile for normalization (0-100)
    
    Returns:
        numpy.ndarray: Normalized projection image
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create an ImagePreprocessor instance
    preprocessor = ImagePreprocessor()
    
    # Load all images in the Z-stack folder
    image_files = sorted([f for f in os.listdir(z_stack_folder) if f.endswith('.tif')])
    z_stack = []
    
    for img_file in image_files:
        img_path = os.path.join(z_stack_folder, img_file)
        img = tifffile.imread(img_path)
        z_stack.append(img)
    
    # Convert to numpy array
    z_stack = np.array(z_stack)
    
    # Normalize the stack using percentile-based contrast stretching
    normalized_stack = preprocessor.stack_percentile_normalize(
        z_stack, 
        low_percentile=low_percentile, 
        high_percentile=high_percentile
    )
    
    # Create a maximum intensity projection
    projection = np.max(normalized_stack, axis=0)
    
    # Save the projection
    output_path = os.path.join(output_folder, 'percentile_normalized_projection.tif')
    tifffile.imwrite(output_path, projection)
    
    # Also save a standard max projection for comparison
    standard_projection = np.max(z_stack, axis=0)
    standard_output_path = os.path.join(output_folder, 'standard_max_projection.tif')
    tifffile.imwrite(standard_output_path, standard_projection)
    
    # Create a visualization comparing the two projections
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display standard max projection
    axes[0].imshow(standard_projection, cmap='gray')
    axes[0].set_title('Standard Max Projection')
    axes[0].axis('off')
    
    # Display percentile normalized projection
    axes[1].imshow(projection, cmap='gray')
    axes[1].set_title('Percentile Normalized Projection')
    axes[1].axis('off')
    
    # Save the comparison figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'projection_comparison.png'))
    
    return projection

def main():
    """
    Main function to demonstrate percentile normalization for Z-stack processing.
    """
    # Example usage with a synthetic Z-stack
    # In a real application, you would use actual Z-stack data
    
    # Create a synthetic Z-stack with varying contrast
    z_stack_folder = 'synthetic_z_stack'
    output_folder = 'normalized_projections'
    
    os.makedirs(z_stack_folder, exist_ok=True)
    
    # Create synthetic Z-stack images with different intensity ranges
    for z in range(5):
        # Create base image with some features
        img = np.zeros((512, 512), dtype=np.uint16)
        
        # Add some circles with varying intensity
        for i in range(10):
            x = np.random.randint(50, 462)
            y = np.random.randint(50, 462)
            r = np.random.randint(10, 50)
            intensity = np.random.randint(1000, 65000)
            
            # Create circle
            y_grid, x_grid = np.ogrid[-y:512-y, -x:512-x]
            mask = x_grid*x_grid + y_grid*y_grid <= r*r
            
            # Add to image with z-dependent intensity
            # Make some planes brighter than others to simulate varying contrast
            z_factor = 0.5 + z * 0.25  # Varies from 0.5 to 1.5
            img[mask] = min(65535, int(intensity * z_factor))
        
        # Save the image
        tifffile.imwrite(os.path.join(z_stack_folder, f'z_{z:02d}.tif'), img)
    
    # Process the Z-stack
    projection = percentile_normalized_projection(z_stack_folder, output_folder)
    
    print(f"Processed Z-stack and saved projections to {output_folder}")
    print(f"Projection shape: {projection.shape}")

if __name__ == "__main__":
    main()
