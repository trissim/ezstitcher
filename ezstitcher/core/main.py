"""
Main module for ezstitcher.

This module provides the main entry point for the ezstitcher package.
"""

import logging
from pathlib import Path

from ezstitcher.core.config import (
    PlateProcessorConfig, StitcherConfig, ZStackProcessorConfig,
    FocusAnalyzerConfig, ImagePreprocessorConfig
)
from ezstitcher.core.plate_processor import PlateProcessor
from ezstitcher.core.zstack_processor import ZStackProcessor
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor

# Legacy imports for backward compatibility
# Removed imports for static method-based classes
# Removed import for static method-based ImageProcessor

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

    This function creates and uses a PlateProcessor instance with the specified configuration.

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
    # Create configurations
    stitcher_config = StitcherConfig(
        tile_overlap=tile_overlap,
        tile_overlap_x=tile_overlap_x,
        tile_overlap_y=tile_overlap_y,
        max_shift=max_shift,
        margin_ratio=margin_ratio
    )

    zstack_config = ZStackProcessorConfig(
        focus_detect=focus_detect,
        focus_method=focus_method,
        create_projections=create_projections,
        stitch_z_reference=stitch_z_reference,
        save_projections=save_projections,
        stitch_all_z_planes=stitch_all_z_planes
    )

    focus_config = FocusAnalyzerConfig(
        method=focus_method
    )

    image_preprocessor_config = ImagePreprocessorConfig(
        preprocessing_funcs=preprocessing_funcs or {},
        composite_weights=composite_weights
    )

    plate_config = PlateProcessorConfig(
        reference_channels=reference_channels,
        well_filter=well_filter,
        stitcher=stitcher_config,
        focus_analyzer=focus_config,
        image_preprocessor=image_preprocessor_config,
        z_stack_processor=zstack_config
    )

    # Create and run the plate processor
    processor = PlateProcessor(plate_config)

    # No fallback to static methods - using only instance-based implementation
    return processor.run(plate_folder)

def modified_process_plate_folder(plate_folder, **kwargs):
    """
    Process a plate folder with Z-stack handling.

    This function uses ZStackProcessor to handle Z-stack detection and processing.

    Args:
        plate_folder (str or Path): Path to the plate folder
        **kwargs: Additional arguments to pass to process_plate_folder

    Returns:
        bool: Success status
    """
    # Create a ZStackProcessor with default config
    z_config = ZStackProcessorConfig()
    z_processor = ZStackProcessor(z_config)

    # Detect Z-stacks
    has_zstack = z_processor.detect_z_stacks(plate_folder)

    if not has_zstack:
        logger.warning(f"No Z-stack detected in {plate_folder}, using standard stitching")
        return process_plate_folder(plate_folder, **kwargs)

    # Get reference_z from kwargs or use default
    reference_z = kwargs.pop('stitch_z_reference', 'best_focus')

    # Create a PlateProcessor with the appropriate configuration
    stitcher_config = StitcherConfig(
        tile_overlap=kwargs.get('tile_overlap', 6.5),
        tile_overlap_x=kwargs.get('tile_overlap_x', None),
        tile_overlap_y=kwargs.get('tile_overlap_y', None),
        max_shift=kwargs.get('max_shift', 50),
        margin_ratio=kwargs.get('margin_ratio', 0.1)
    )

    zstack_config = ZStackProcessorConfig(
        focus_detect=kwargs.get('focus_detect', False),
        focus_method=kwargs.get('focus_method', 'combined'),
        create_projections=kwargs.get('create_projections', False),
        stitch_z_reference=reference_z,
        save_projections=kwargs.get('save_projections', True),
        stitch_all_z_planes=kwargs.get('stitch_all_z_planes', False)
    )

    focus_config = FocusAnalyzerConfig(
        method=kwargs.get('focus_method', 'combined')
    )

    image_preprocessor_config = ImagePreprocessorConfig(
        preprocessing_funcs=kwargs.get('preprocessing_funcs', {}),
        composite_weights=kwargs.get('composite_weights', None)
    )

    plate_config = PlateProcessorConfig(
        reference_channels=kwargs.get('reference_channels', ['1']),
        well_filter=kwargs.get('well_filter', None),
        stitcher=stitcher_config,
        focus_analyzer=focus_config,
        image_preprocessor=image_preprocessor_config,
        z_stack_processor=zstack_config
    )

    # Create and run the plate processor
    processor = PlateProcessor(plate_config)

    # No fallback to static methods - using only instance-based implementation
    return processor.run(plate_folder)

def process_bf(imgs):
    """
    Process brightfield images.

    This function uses ImagePreprocessor to process brightfield images.

    Args:
        imgs (list): List of brightfield images

    Returns:
        list: List of processed images
    """
    # Create an ImagePreprocessor with default config
    preprocessor = ImagePreprocessor()
    return preprocessor.process_bf(imgs)

def find_best_focus(image_stack, method='combined', roi=None):
    """
    Find the best focused image in a stack.

    This function uses FocusAnalyzer to find the best focused image.

    Args:
        image_stack (list): List of images
        method (str): Focus detection method
        roi (tuple): Optional region of interest

    Returns:
        tuple: (best_focus_index, focus_scores)
    """
    # Create a FocusAnalyzer with the specified method
    config = FocusAnalyzerConfig(method=method, roi=roi)
    analyzer = FocusAnalyzer(config)
    return analyzer.find_best_focus(image_stack)
