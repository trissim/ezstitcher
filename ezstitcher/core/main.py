"""
Main module for ezstitcher.

This module provides the main entry point for the ezstitcher package.
"""

import logging
from pathlib import Path

from ezstitcher.core.stitcher_manager import StitcherManager
from ezstitcher.core.z_stack_manager import ZStackManager
from ezstitcher.core.image_processor import ImageProcessor

logger = logging.getLogger(__name__)

def process_plate_folder(plate_folder, reference_channels=['1'],
                         preprocessing_funcs=None, margin_ratio=0.1,
                         composite_weights=None, well_filter=None,
                         tile_overlap=6.5, tile_overlap_x=None, tile_overlap_y=None,
                         max_shift=50, focus_detect=False, focus_method="combined",
                         create_projections=False, stitch_z_reference='best_focus',
                         save_projections=True, stitch_all_z_planes=False):
    """
    Process an entire plate folder with microscopy images.

    This function is a wrapper around StitcherManager.process_plate_folder.

    Args:
        plate_folder (str or Path): Base folder for the plate
        reference_channels (list): List of channels to use as reference
        preprocessing_funcs (dict): Dict mapping wavelength/channel to preprocessing function
        margin_ratio (float): Blending margin ratio for stitching
        composite_weights (dict): Dict mapping channels to weights for composite
        well_filter (list): Optional list of wells to process
        tile_overlap (float): Percentage of overlap between tiles
        tile_overlap_x (float): Horizontal overlap percentage
        tile_overlap_y (float): Vertical overlap percentage
        max_shift (int): Maximum shift allowed between tiles in microns
        focus_detect (bool): Whether to enable focus detection for Z-stacks
        focus_method (str): Focus detection method to use
        create_projections (bool): Whether to create projections from Z-stacks
        stitch_z_reference (str): Z-plane to use for stitching ('best_focus', 'max', 'mean')
        save_projections (bool): Whether to save projection images after stitching
        stitch_all_z_planes (bool): Whether to stitch all Z-planes using reference positions

    Returns:
        bool: True if successful, False otherwise
    """
    return StitcherManager.process_plate_folder(
        plate_folder=plate_folder,
        reference_channels=reference_channels,
        preprocessing_funcs=preprocessing_funcs,
        margin_ratio=margin_ratio,
        composite_weights=composite_weights,
        well_filter=well_filter,
        tile_overlap=tile_overlap,
        tile_overlap_x=tile_overlap_x,
        tile_overlap_y=tile_overlap_y,
        max_shift=max_shift,
        focus_detect=focus_detect,
        focus_method=focus_method,
        create_projections=create_projections,
        stitch_z_reference=stitch_z_reference,
        save_projections=save_projections,
        stitch_all_z_planes=stitch_all_z_planes
    )

def modified_process_plate_folder(plate_folder, **kwargs):
    """
    Process a plate folder with Z-stack handling.

    This function is a wrapper around ZStackManager.stitch_across_z.

    Args:
        plate_folder (str or Path): Path to the plate folder
        **kwargs: Additional arguments to pass to process_plate_folder

    Returns:
        bool: Success status
    """
    # First preprocess to organize z-stacks if needed
    has_zstack, z_info = ZStackManager.preprocess_plate_folder(plate_folder)

    if not has_zstack:
        logger.warning(f"No Z-stack detected in {plate_folder}, using standard stitching")
        return process_plate_folder(plate_folder, **kwargs)

    # Get reference_z from kwargs or use default
    reference_z = kwargs.pop('stitch_z_reference', 'best_focus')

    # Use the Z-stack manager to handle stitching
    return ZStackManager.stitch_across_z(plate_folder, reference_z=reference_z, **kwargs)

def process_bf(imgs):
    """
    Process brightfield images.

    This function is a wrapper around ImageProcessor.process_bf.

    Args:
        imgs (list): List of brightfield images

    Returns:
        list: List of processed images
    """
    return ImageProcessor.process_bf(imgs)

def find_best_focus(image_stack, method='combined', roi=None):
    """
    Find the best focused image in a stack.

    This function is a wrapper around FocusDetector.find_best_focus.

    Args:
        image_stack (list): List of images
        method (str): Focus detection method
        roi (tuple): Optional region of interest

    Returns:
        tuple: (best_focus_index, focus_scores)
    """
    from ezstitcher.core.focus_detector import FocusDetector
    return FocusDetector.find_best_focus(image_stack, method, roi)
