"""
Main module for ezstitcher.

This module provides the main entry point for the ezstitcher package.
"""

import logging
from pathlib import Path

# Import both dataclass and pydantic configs for backward compatibility
from ezstitcher.core.config import (
    PlateProcessorConfig, StitcherConfig, ZStackProcessorConfig,
    FocusAnalyzerConfig, ImagePreprocessorConfig
)

# Import Pydantic configs for new code
from ezstitcher.core.pydantic_config import (
    PlateProcessorConfig as PydanticPlateProcessorConfig,
    StitcherConfig as PydanticStitcherConfig,
    ZStackProcessorConfig as PydanticZStackProcessorConfig,
    FocusAnalyzerConfig as PydanticFocusAnalyzerConfig,
    ImagePreprocessorConfig as PydanticImagePreprocessorConfig,
    ConfigPresets
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
                         max_shift=50, focus_method="combined",
                         stitch_all_z_planes=False, use_reference_positions=False,
                         microscope_type='auto',
                         # New parameters
                         z_reference_function='max_projection',
                         save_reference=True,
                         additional_projections=None,
                         # Deprecated parameters (kept for backward compatibility)
                         focus_detect=None,
                         create_projections=None,
                         stitch_z_reference=None,
                         reference_method=None,
                         save_projections=None):
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
        focus_method (str): Focus detection method to use when using best_focus

        # New parameters
        z_reference_function (str or callable): Function that converts a 3D stack to a 2D image
            Can be a string name of a standard function or a callable
            Standard functions: "max_projection", "mean_projection", "best_focus"
        save_reference (bool): Whether to save the reference image
        additional_projections (list): Types of additional projections to create

        # Deprecated parameters
        focus_detect (bool, optional): Deprecated. Use z_reference_function="best_focus" instead
        create_projections (bool, optional): Deprecated. Use save_reference instead
        stitch_z_reference (str, optional): Deprecated. Use z_reference_function instead
        reference_method (str, optional): Deprecated. Use z_reference_function instead
        save_projections (bool, optional): Deprecated. Use save_reference instead
        stitch_all_z_planes (bool): Whether to stitch all Z-planes using reference positions
        use_reference_positions (bool): Whether to use existing reference positions
        microscope_type (str): Type of microscope ('auto', 'ImageXpress', 'OperaPhenix')

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

    # Handle backward compatibility
    if any(param is not None for param in [stitch_z_reference, focus_detect, create_projections, save_projections, reference_method]):
        # Use deprecated parameters if provided
        zstack_config = ZStackProcessorConfig(
            # Deprecated parameters
            focus_detect=focus_detect,
            focus_method=focus_method,
            create_projections=create_projections,
            stitch_z_reference=stitch_z_reference,
            save_projections=save_projections,
            reference_method=reference_method,
            stitch_all_z_planes=stitch_all_z_planes
        )
    else:
        # Use new parameters
        zstack_config = ZStackProcessorConfig(
            z_reference_function=z_reference_function,
            focus_method=focus_method,
            save_reference=save_reference,
            additional_projections=additional_projections,
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
        use_reference_positions=use_reference_positions,
        microscope_type=microscope_type,
        preprocessing_funcs=preprocessing_funcs,
        composite_weights=composite_weights,
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

    # Get microscope_type from kwargs or use default
    microscope_type = kwargs.pop('microscope_type', 'auto')

    # Create a PlateProcessor with the appropriate configuration
    stitcher_config = StitcherConfig(
        tile_overlap=kwargs.get('tile_overlap', 6.5),
        tile_overlap_x=kwargs.get('tile_overlap_x', None),
        tile_overlap_y=kwargs.get('tile_overlap_y', None),
        max_shift=kwargs.get('max_shift', 50),
        margin_ratio=kwargs.get('margin_ratio', 0.1)
    )

    # Handle backward compatibility
    if any(param in kwargs for param in ['stitch_z_reference', 'focus_detect', 'create_projections', 'save_projections', 'reference_method']):
        # Use deprecated parameters if provided
        zstack_config = ZStackProcessorConfig(
            # Deprecated parameters
            focus_detect=kwargs.get('focus_detect', None),
            focus_method=kwargs.get('focus_method', 'combined'),
            create_projections=kwargs.get('create_projections', None),
            stitch_z_reference=kwargs.get('stitch_z_reference', None),
            save_projections=kwargs.get('save_projections', None),
            reference_method=kwargs.get('reference_method', None),
            stitch_all_z_planes=kwargs.get('stitch_all_z_planes', False)
        )
    else:
        # Use new parameters
        zstack_config = ZStackProcessorConfig(
            z_reference_function=kwargs.get('z_reference_function', 'max_projection'),
            focus_method=kwargs.get('focus_method', 'combined'),
            save_reference=kwargs.get('save_reference', True),
            additional_projections=kwargs.get('additional_projections', None),
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
        use_reference_positions=kwargs.get('use_reference_positions', False),
        microscope_type=microscope_type,
        preprocessing_funcs=kwargs.get('preprocessing_funcs', {}),
        composite_weights=kwargs.get('composite_weights', None),
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


def process_plate_folder_with_config(plate_folder, config_file=None, config_preset=None, **kwargs):
    """
    Process a plate folder using a configuration file or preset.

    This function provides a more flexible way to configure the processing pipeline
    using either a configuration file (JSON or YAML) or a predefined preset.

    Args:
        plate_folder (str or Path): Path to the plate folder
        config_file (str or Path, optional): Path to a configuration file (JSON or YAML)
        config_preset (str, optional): Name of a predefined configuration preset
            ('default', 'z_stack_best_focus', 'z_stack_per_plane', 'high_resolution')
        **kwargs: Additional arguments to override configuration values

    Returns:
        bool: Success status
    """
    # Load configuration
    if config_file is not None:
        # Load from file
        config_path = Path(config_file)
        if config_path.suffix.lower() in ('.json', '.jsn'):
            config = PydanticPlateProcessorConfig.from_json(config_path)
        elif config_path.suffix.lower() in ('.yaml', '.yml'):
            config = PydanticPlateProcessorConfig.from_yaml(config_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    elif config_preset is not None:
        # Use a predefined preset
        if config_preset == 'default':
            config = ConfigPresets.default()
        elif config_preset == 'z_stack_best_focus':
            config = ConfigPresets.z_stack_best_focus()
        elif config_preset == 'z_stack_per_plane':
            config = ConfigPresets.z_stack_per_plane()
        elif config_preset == 'high_resolution':
            config = ConfigPresets.high_resolution()
        else:
            raise ValueError(f"Unknown configuration preset: {config_preset}")
    else:
        # Use default configuration
        config = PydanticPlateProcessorConfig()

    # Override configuration with kwargs
    if kwargs:
        # Update configuration with kwargs
        # This is a simplified approach - in a real implementation,
        # you would need to handle nested configurations
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Convert Pydantic config to dataclass config for compatibility
    # In a future version, you could modify PlateProcessor to accept Pydantic configs directly
    plate_config = PlateProcessorConfig(
        reference_channels=config.reference_channels,
        well_filter=config.well_filter,
        use_reference_positions=config.use_reference_positions,
        microscope_type=config.microscope_type,
        output_dir_suffix=config.output_dir_suffix,
        positions_dir_suffix=config.positions_dir_suffix,
        stitched_dir_suffix=config.stitched_dir_suffix,
        best_focus_dir_suffix=config.best_focus_dir_suffix,
        projections_dir_suffix=config.projections_dir_suffix,
        timepoint_dir_name=config.timepoint_dir_name,
        preprocessing_funcs=config.preprocessing_funcs,
        composite_weights=config.composite_weights,
        stitcher=StitcherConfig(
            tile_overlap=config.stitcher.tile_overlap,
            tile_overlap_x=config.stitcher.tile_overlap_x,
            tile_overlap_y=config.stitcher.tile_overlap_y,
            max_shift=config.stitcher.max_shift,
            margin_ratio=config.stitcher.margin_ratio,
            pixel_size=config.stitcher.pixel_size
        ),
        focus_analyzer=FocusAnalyzerConfig(
            method=config.focus_analyzer.method,
            roi=config.focus_analyzer.roi,
            weights=config.focus_analyzer.weights
        ),
        image_preprocessor=ImagePreprocessorConfig(
            preprocessing_funcs=config.image_preprocessor.preprocessing_funcs,
            composite_weights=config.image_preprocessor.composite_weights
        ),
        z_stack_processor=ZStackProcessorConfig(
            z_reference_function=config.z_stack_processor.z_reference_function,
            focus_method=config.z_stack_processor.focus_method,
            save_reference=config.z_stack_processor.save_reference,
            stitch_all_z_planes=config.z_stack_processor.stitch_all_z_planes,
            additional_projections=config.z_stack_processor.additional_projections
        )
    )

    # Create and run the plate processor
    processor = PlateProcessor(plate_config)
    return processor.run(plate_folder)
