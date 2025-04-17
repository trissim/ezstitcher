import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

from ezstitcher.core.config import StitcherConfig, PipelineConfig, FocusAnalyzerConfig
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.microscope_interfaces import MicroscopeHandler, create_microscope_handler

# Default padding width for consistent file naming
DEFAULT_PADDING = 3
from ezstitcher.core.image_locator import ImageLocator

# Set up logger
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    A robust pipeline orchestrator for microscopy image processing.

    The pipeline follows a clear, linear flow:
    1. Load and organize images
    2. Process tiles (per well, per site, per channel)
    3. Select or compose channels
    4. Flatten Z-stacks (if present)
    5. Generate stitching positions
    6. Stitch images
    """

    def __init__(self, config: PipelineConfig):
        """Initialize with configuration."""
        self.config = config
        self.fs_manager = FileSystemManager()
        self.image_preprocessor = ImagePreprocessor()

        # Initialize focus analyzer directly
        focus_config = config.focus_config or FocusAnalyzerConfig(method=config.focus_method)
        self.focus_analyzer = FocusAnalyzer(focus_config)

        self.microscope_handler = None
        self.stitcher = None

    def _prepare_images(self, plate_path):
        """
        Prepare images by padding filenames and organizing Z-stack folders.

        Args:
            plate_path: Path to the plate folder
            parser: FilenameParser to use for file operations

        Returns:
            Path: Path to the image directory
        """
        # Find the image directory
        image_dir = ImageLocator.find_image_directory(plate_path)
        logger.info(f"Found image directory: {image_dir}")

        # Rename files with consistent padding and force missing suffixes
        self.fs_manager.rename_files_with_consistent_padding(
            image_dir,
            parser=self.microscope_handler,
            width=DEFAULT_PADDING,  # Use consistent padding width
            force_suffixes=True  # Force missing suffixes to be added
        )

        # Detect and organize Z-stack folders
        has_zstack_folders, _ = self.fs_manager.detect_zstack_folders(image_dir)
        if has_zstack_folders:
            logger.info("Organizing Z-stack folders in %s", image_dir)
            self.fs_manager.organize_zstack_folders(image_dir, filename_parser=self.microscope_handler)

        # Return the image directory (which may have changed if Z-stack folders were organized)
        return ImageLocator.find_image_directory(plate_path)

    def run(self, plate_folder):
        """
        Process a plate through the complete pipeline.

        Args:
            plate_folder: Path to the plate folder

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Setup
            plate_path = Path(plate_folder)
            self.microscope_handler = create_microscope_handler('auto', plate_folder=plate_path)
            self.config.grid_size = self.microscope_handler.get_grid_dimensions(plate_path)
            self.config.pixel_size = self.microscope_handler.get_pixel_size(plate_path)
            self.stitcher = Stitcher(self.config.stitcher, filename_parser=self.microscope_handler.parser)

            # Prepare images (pad filenames and organize Z-stack folders)
            input_dir = self._prepare_images(plate_path)

            # Create directory structure
            dirs = {
                'input': input_dir,
                'processed': plate_path.parent / f"{plate_path.name}{self.config.processed_dir_suffix}",
                'post_processed': plate_path.parent / f"{plate_path.name}{self.config.post_processed_dir_suffix}",
                'positions': plate_path.parent / f"{plate_path.name}{self.config.positions_dir_suffix}",
                'stitched': plate_path.parent / f"{plate_path.name}{self.config.stitched_dir_suffix}"
            }

            for dir_path in dirs.values():
                self.fs_manager.ensure_directory(dir_path)

            # Get patterns by well
            patterns_by_well = self.microscope_handler.auto_detect_patterns(
                dirs['input'],
                well_filter=self.config.well_filter,
                variable_components=['site']
            )
            patterns_by_well_z = self.microscope_handler.auto_detect_patterns(
                dirs['input'],
                well_filter=self.config.well_filter,
                variable_components=['z_index']
            )
            # Well filter is already applied in auto_detect_patterns

            # Process each well
            #for well, wavelength_patterns in patterns_by_well.items():
            for well in patterns_by_well.keys():
                wavelength_patterns = patterns_by_well[well]
                wavelength_patterns_z = patterns_by_well_z[well]
                self.process_well(well, wavelength_patterns, wavelength_patterns_z, dirs)

            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return False

    def process_well(self, well, wavelength_patterns, wavelength_patterns_z, dirs):
        """
        Process a single well through the pipeline.

        Args:
            well: Well identifier
            wavelength_patterns: Dictionary mapping wavelengths to varying site patterns
            wavelength_patterns: Dictionary mapping wavelengths to varying z_index patterns
            dirs: Dictionary of directories
        """
        logger.info(f"Processing well {well}")

        # 1. Process reference images (for position generation)
        self.process_reference_images(well, wavelength_patterns, wavelength_patterns_z, dirs)

        # 2. Generate stitching positions
        positions_file, stitch_pattern = self.generate_positions(well, dirs)

        # 3. Process final images (for stitching)
        self.process_final_images(well, wavelength_patterns, wavelength_patterns_z, dirs)

        # 4. Stitch final images
        self.stitch_images(well, wavelength_patterns, dirs, positions_file)

    def process_reference_images(self, well, wavelength_patterns, wavelength_patterns_z, dirs):
        """
        Process images for position generation.

        Args:
            well: Well identifier
            wavelength_patterns: Dictionary mapping wavelengths to patterns grouped by variable sites
            wavelength_patterns_z: Dictionary mapping wavelengths to patterns grouped by variable z_index
            dirs: Dictionary of directories
        """
        logger.info(f"Processing reference images for well {well}")

        # Determine which channels to use as reference
        reference_channels = [ch for ch in self.config.reference_channels if ch in wavelength_patterns]
        if not reference_channels:
            # Fall back to first available channel
            reference_channels = [next(iter(wavelength_patterns.keys()))]

        # Track processed files for cleanup
        processed_files = []

        # Process each reference channel
        for channel in reference_channels:
            patterns = wavelength_patterns_z[channel]

            # Apply tile processing
            tile_files = self.process_tiles(
                dirs['input'],
                dirs['processed'],
                patterns,
                channel
            )
            processed_files.extend(tile_files)

        # Create composite or select single channel
        composite_files = self.create_composite(
            well,
            dirs['processed'],
            {ch: wavelength_patterns[ch] for ch in reference_channels},
            self.config.reference_composite_weights
        )

        # Clean up processed tiles after composition
        if composite_files:
            logger.info(f"Cleaning up processed tiles after reference composition")
            self.fs_manager.cleanup_processed_files(processed_files, composite_files)

        # Flatten Z-stacks if needed
        flatten_patterns = []
        for pattern in patterns:
            sample = pattern.replace('{iii}', '001')
            meta = self.microscope_handler.parser.parse_filename(sample)
            flatten_patterns.append(self.microscope_handler.parser.construct_filename(
                well=meta['well'],
                site=meta['site'],
                z_index='{iii}',
                extension='.tif',
                site_padding=DEFAULT_PADDING,  # Use consistent padding
                z_padding=DEFAULT_PADDING      # Use consistent padding
            ))
        flatten_method = self.config.reference_flatten
        flattened_files = self.flatten_zstacks(dirs['processed'],dirs['processed'],flatten_patterns,method=flatten_method)

        if flattened_files:
            logger.info(f"Cleaning up processed tiles after reference composition")
            self.fs_manager.cleanup_processed_files(composite_files, flattened_files)

    def process_final_images(self, well, wavelength_patterns, wavelength_patterns_z, dirs):
        """
        Process images for final stitching.

        Args:
            well: Well identifier
            wavelength_patterns: Dictionary mapping wavelengths to patterns grouped by variable sites
            wavelength_patterns_z: Dictionary mapping wavelengths to patterns grouped by variable z_index
            dirs: Dictionary of directories
        """
        logger.info(f"Processing final images for well {well}")

        # Always process all available channels for final stitching
        channels_to_process = list(wavelength_patterns.keys())
        logger.info(f"Processing all {len(channels_to_process)} available channels for well {well}")

        # Track processed files for cleanup
        processed_files = []

        # Process each channel
        for channel in channels_to_process:
            patterns = wavelength_patterns_z[channel]

            # Apply tile processing
            tile_files = self.process_tiles(
                dirs['input'],
                dirs['post_processed'],
                patterns,
                channel
            )
            processed_files.extend(tile_files)

        flatten_patterns = []
        for pattern in patterns:
            sample = pattern.replace('{iii}', '001')
            meta = self.microscope_handler.parser.parse_filename(sample)
            flatten_patterns.append(self.microscope_handler.parser.construct_filename(
                well=meta['well'],
                site=meta['site'],
                channel=meta['channel'],
                z_index='{iii}',
                extension='.tif',
                site_padding=DEFAULT_PADDING,
                z_padding=DEFAULT_PADDING
            ))

        flatten_method = self.config.stitch_flatten
        flattened_files = self.flatten_zstacks(dirs['post_processed'],dirs['post_processed'],flatten_patterns,method=flatten_method)

        if flattened_files:
            logger.info(f"Cleaning up processed tiles after reference composition")
            self.fs_manager.cleanup_processed_files(processed_files, flattened_files)

    def process_tiles(self, input_dir, output_dir, patterns, channel):
        """
        Unified processing using zstack_processor.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            pattern: List of file pattern
            channel: Channel identifier

        Returns:
            list: Paths to created images
        """
        output_files = []

        for pattern in patterns:
            matching_files = self.microscope_handler.parser.path_list_from_pattern(input_dir, pattern)
            images = [self.fs_manager.load_image(input_dir / filename) for filename in matching_files]
            images = [img for img in images if img is not None]
            # Apply preprocessing if specified
            preprocess_func = None
            if hasattr(self.config, 'preprocessing_funcs') and self.config.preprocessing_funcs:
                preprocess_func = self.config.preprocessing_funcs.get(channel)

            if preprocess_func and images:
                images = self.image_preprocessor.apply_function_to_stack(images, preprocess_func)

            for image, matching_file in zip (images, matching_files):
                output_path = output_dir / matching_file
                self.fs_manager.save_image(output_path, image)
                output_files.append(output_path)

        return output_files

    def create_composite(self, well, input_dir, channel_patterns, weights=None):
        """
        Create a composite image from multiple channels for each site and z-index.

        Args:
            well: Well identifier
            input_dir: Input directory
            channel_patterns: Dictionary mapping channels to patterns
            weights: Dictionary mapping channels to weights, or None to use first channel as reference

        Returns:
            list: Paths to created composite images
        """
        output_files = []
        images_by_site_z = {}

        # Collect all images by site, z-index, and channel
        for channel, patterns in channel_patterns.items():
            patterns = [patterns] if not isinstance(patterns, list) else patterns

            for pattern in patterns:
                for filename in self.microscope_handler.parser.path_list_from_pattern(input_dir, pattern):
                    metadata = self.microscope_handler.parser.parse_filename(filename)
                    if not metadata or 'site' not in metadata:
                        continue

                    site = metadata['site']
                    z_index = metadata.get('z_index', 1)  # Default to 1 if no z-index
                    key = (site, z_index)

                    if key not in images_by_site_z:
                        images_by_site_z[key] = {}

                    image = self.fs_manager.load_image(input_dir / filename)
                    if image is not None:
                        images_by_site_z[key][channel] = image

        # Process each site and z-index combination
        for (site, z_index), channel_images in images_by_site_z.items():
            if not channel_images:
                continue

            # Create output filename with site and z-index
            output_path = input_dir / self.microscope_handler.parser.construct_filename(
                well=well,
                site=site,
                z_index=z_index,
                extension='.tif',
                site_padding=DEFAULT_PADDING,
                z_padding=DEFAULT_PADDING
            )

            # Save image based on available channels
            if len(channel_patterns) == 1:
                # Single channel or no weights - use first channel
                channel = next(iter(channel_images.keys()))
                self.fs_manager.save_image(output_path, channel_images[channel])
            else:
                # Create weighted composite
                composite = self.image_preprocessor.create_composite(channel_images, weights)
                self.fs_manager.save_image(output_path, composite)

            output_files.append(output_path)

        return output_files

    def flatten_zstacks(self, input_dir, output_dir, patterns,method="max"):
        """
        Finds planes of the same tile and flattens them into a single image using zstack_processor.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            pattern: List of file pattern

        Returns:
            list: Paths to created images
        """
        output_files = []
        for pattern in patterns:
            matching_files = self.microscope_handler.parser.path_list_from_pattern(input_dir, pattern)
            images = [self.fs_manager.load_image(input_dir / filename) for filename in matching_files]
            images = [img for img in images if img is not None]

            # Create a projection from the Z-stack
            if method is not None:
                projected_image = self.image_preprocessor.create_projection(
                    images,
                    method=method,
                    focus_analyzer=self.focus_analyzer
                )

                # Get the output filename
                pattern_with_site = pattern.replace('{iii}', '001')
                metadata = self.microscope_handler.parser.parse_filename(pattern_with_site)
                fname = self.microscope_handler.parser.construct_filename(
                    well=metadata['well'],
                    site=metadata['site'],
                    channel=metadata['channel'],
                    extension='.tif',
                    site_padding=DEFAULT_PADDING,  # Use consistent padding
                    z_padding=DEFAULT_PADDING      # Use consistent padding
                )

                # Save the projected image
                output_path = output_dir / fname
                output_files.append(output_path)
                self.fs_manager.save_image(output_path, projected_image)

        return output_files


    def generate_positions(self, well, dirs):
        """
        Generate stitching positions for a well.

        Args:
            well: Well identifier
            dirs: Dictionary of directories

        Returns:
            Path to positions file
        """
        logger.info(f"Generating positions for well {well}")

        # Ensure positions directory exists
        self.fs_manager.ensure_directory(dirs['positions'])

        # Generate positions file path with well name and .csv extension
        positions_file = dirs['positions'] / f"{well}.csv"

        # Use standardized reference pattern (without channel information)
        # After processing, all images are saved with well_s### format
        reference_pattern = self.microscope_handler.parser.construct_filename(
            well=well,
            site="{iii}",
            extension='.tif',
            site_padding=DEFAULT_PADDING,
            z_padding=DEFAULT_PADDING
        )

        # Generate positions
        self.stitcher.generate_positions(
            dirs['processed'],
            reference_pattern,
            positions_file,  # Pass the file path, not the directory
            self.config.grid_size[0],
            self.config.grid_size[1],
        )

        return positions_file, reference_pattern

    def stitch_images(self, well, wavelength_patterns, dirs, positions_file):
        """
        Stitch images for a well.

        Args:
            well: Well identifier
            wavelength_patterns: Dictionary mapping wavelengths to patterns
            dirs: Dictionary of directories
            positions_file: Path to positions file
        """
        logger.info(f"Stitching images for well {well}")

        # Get grid dimensions
        grid_dims = self.get_grid_dimensions(dirs['input'])

        # Always stitch all available channels
        channels_to_stitch = list(wavelength_patterns.keys())
        logger.info(f"Stitching all {len(channels_to_stitch)} available channels for well {well}")

        # Add composite if needed
        if len(channels_to_stitch) > 1 and self.config.final_composite_weights:
            channels_to_stitch.append("composite")

        # Use auto_detect_patterns to find all patterns for this well
        patterns_by_well = self.microscope_handler.parser.auto_detect_patterns(
            dirs['post_processed'],
            well_filter=[well],
            variable_components=['site']
        )

        if not patterns_by_well or well not in patterns_by_well:
            logger.warning(f"No patterns found for well {well} in {dirs['post_processed']}")
            return

        # Get all patterns for this well and flatten them into a single list
        patterns_by_channel = patterns_by_well[well]
        all_patterns = []
        for patterns in patterns_by_channel.values():
            all_patterns.extend(patterns)

        logger.info(f"Found {len(all_patterns)} patterns for well {well}")

        # Stitch each pattern
        for pattern in all_patterns:
            # Find all matching files for this pattern
            matching_files = self.microscope_handler.parser.path_list_from_pattern(dirs['post_processed'], pattern)

            if not matching_files:
                logger.warning(f"No files found for pattern {pattern}")
                continue

            # Extract pattern suffix to determine output filename
            parsable = pattern.replace('{iii}','001')
            metadata = self.microscope_handler.parser.parse_filename(parsable)
            output_filename = self.microscope_handler.parser.construct_filename(
                well=metadata['well'],
                site=metadata['site'],
                channel=metadata['channel'],
                z_index=metadata['z_index'],
                extension='.tif',
                site_padding=DEFAULT_PADDING,
                z_padding=DEFAULT_PADDING
            )
            # Example: A01_s{iii}_w1.tif -> _w1.tif
            #suffix = pattern.replace(f"{well}_s{{iii}}", "")

            # Create output filename based on the pattern
            #output_filename = f"{well}{suffix}"
            output_path = dirs['stitched'] / output_filename
            logger.info(f"Stitching pattern {pattern} to {output_path}")

            # Assemble the stitched image
            self.stitcher.assemble_image(
                positions_path=positions_file,
                images_dir=dirs['post_processed'],
                output_path=output_path,
                override_names=matching_files
            )

    def check_for_zstack(self, input_dir, pattern):
        """
        Check if a pattern contains Z-stacks.

        Args:
            input_dir: Input directory
            pattern: File pattern

        Returns:
            bool: True if Z-stacks are present, False otherwise
        """
        # Check if the directory has any Z-stack folders
        has_zstack, _ = self.fs_manager.detect_zstack_folders(input_dir)
        return has_zstack



    def get_grid_dimensions(self, input_dir):
        """
        Get grid dimensions for stitching.

        Args:
            input_dir: Input directory

        Returns:
            tuple: (grid_size_x, grid_size_y)
        """
        # Use the microscope handler to get grid dimensions
        return self.microscope_handler.get_grid_dimensions(input_dir)
