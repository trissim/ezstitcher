"""
Utility functions for ezstitcher.

This module contains common utility functions used across the ezstitcher package.
"""

import os
import re
import numpy as np
import logging
from pathlib import Path
import tifffile

logger = logging.getLogger(__name__)

def ensure_directory(directory):
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory (str or Path): Directory path to ensure exists

    Returns:
        Path: Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def list_image_files(directory, extensions=None):
    """
    List all image files in a directory with specified extensions.

    Args:
        directory (str or Path): Directory to search
        extensions (list): List of file extensions to include (default: common image formats)

    Returns:
        list: List of Path objects for image files
    """
    if extensions is None:
        extensions = ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

    directory = Path(directory)
    image_files = []

    for ext in extensions:
        image_files.extend(list(directory.glob(f"*{ext}")))

    return sorted(image_files)

def parse_filename(filename):
    """
    Parse a microscopy image filename to extract well, site, wavelength, and z-index.

    Args:
        filename (str): Filename to parse

    Returns:
        dict: Dictionary with extracted components or None if parsing fails
    """
    # Common pattern: WellID_sXXX_wY_zZZZ.tif
    # Example: A01_s001_w1_z001.tif
    pattern = re.compile(r'([A-Z]\d+)_s(\d+)_w(\d+)(?:_z(\d+))?\..*')
    match = pattern.match(filename)

    if match:
        well = match.group(1)
        site = int(match.group(2))
        wavelength = int(match.group(3))
        z_index = int(match.group(4)) if match.group(4) else None

        return {
            'well': well,
            'site': site,
            'wavelength': wavelength,
            'z_index': z_index
        }

    return None

def load_image(file_path):
    """
    Load an image and ensure it's 2D grayscale.

    Args:
        file_path (str or Path): Path to the image file

    Returns:
        numpy.ndarray: 2D grayscale image
    """
    try:
        img = tifffile.imread(str(file_path))

        # Convert to 2D grayscale if needed
        if img.ndim == 3:
            # Check if it's a channel-first format (C, H, W)
            if img.shape[0] <= 4:  # Assuming max 4 channels (RGBA)
                # Convert channel-first to 2D by taking mean across channels
                img = np.mean(img, axis=0).astype(img.dtype)
            # Check if it's a channel-last format (H, W, C)
            elif img.shape[2] <= 4:  # Assuming max 4 channels (RGBA)
                # Convert channel-last to 2D by taking mean across channels
                img = np.mean(img, axis=2).astype(img.dtype)
            else:
                # If it's a 3D image with a different structure, use the first slice
                img = img[0].astype(img.dtype)

        return img
    except Exception as e:
        logger.error(f"Error loading image {file_path}: {e}")
        return None

def save_image(file_path, image, compression=None):
    """
    Save an image to disk.

    Args:
        file_path (str or Path): Path to save the image
        image (numpy.ndarray): Image to save
        compression (str or None): Compression method (default: None)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        directory = Path(file_path).parent
        directory.mkdir(parents=True, exist_ok=True)

        # Save image
        tifffile.imwrite(str(file_path), image, compression=compression)
        return True
    except Exception as e:
        logger.error(f"Error saving image {file_path}: {e}")
        return False

def path_list_from_pattern(directory, pattern):
    """
    Get a list of filenames matching a pattern in a directory.

    Args:
        directory (str or Path): Directory to search
        pattern (str): Pattern to match with {iii} placeholder for site index

    Returns:
        list: List of matching filenames
    """
    directory = Path(directory)

    # Convert pattern to regex
    # Replace {iii} with (\d+) to match any number
    regex_pattern = pattern.replace('{iii}', '(\\d+)')
    regex = re.compile(regex_pattern)

    # Find all matching files
    matching_files = []
    for file_path in directory.glob('*'):
        if regex.match(file_path.name):
            matching_files.append(file_path.name)

    return sorted(matching_files)

def create_linear_weight_mask(height, width, margin_ratio=0.1):
    """
    Create a 2D weight mask that linearly ramps from 0 at the edges
    to 1 in the center.

    Args:
        height (int): Height of the mask
        width (int): Width of the mask
        margin_ratio (float): Ratio of the margin to the image size

    Returns:
        numpy.ndarray: 2D weight mask
    """
    margin_y = int(np.floor(height * margin_ratio))
    margin_x = int(np.floor(width * margin_ratio))

    weight_y = np.ones(height, dtype=np.float32)
    if margin_y > 0:
        ramp_top = np.linspace(0, 1, margin_y, endpoint=False)
        ramp_bottom = np.linspace(1, 0, margin_y, endpoint=False)
        weight_y[:margin_y] = ramp_top
        weight_y[-margin_y:] = ramp_bottom

    weight_x = np.ones(width, dtype=np.float32)
    if margin_x > 0:
        ramp_left = np.linspace(0, 1, margin_x, endpoint=False)
        ramp_right = np.linspace(1, 0, margin_x, endpoint=False)
        weight_x[:margin_x] = ramp_left
        weight_x[-margin_x:] = ramp_right

    mask_2d = np.outer(weight_y, weight_x)
    return mask_2d

def parse_positions_csv(csv_path):
    """
    Parse a CSV file with lines of the form:
      file: <filename>; grid: (col, row); position: (x, y)

    Args:
        csv_path (str or Path): Path to the CSV file

    Returns:
        list: List of tuples (filename, x_float, y_float)
    """
    entries = []
    with open(csv_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Example line:
            # file: some_image.tif; grid: (0, 0); position: (123.45, 67.89)
            file_match = re.search(r'file:\s*([^;]+);', line)
            pos_match = re.search(r'position:\s*\(([^,]+),\s*([^)]+)\)', line)
            if file_match and pos_match:
                fname = file_match.group(1).strip()
                x_val = float(pos_match.group(1).strip())
                y_val = float(pos_match.group(2).strip())
                entries.append((fname, x_val, y_val))
    return entries
