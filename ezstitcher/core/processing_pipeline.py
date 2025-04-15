import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field

from ezstitcher.core.config import StitcherConfig
from ezstitcher.core.zstack_processor import ZStackProcessor, ZStackProcessorConfig
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.filename_parser import FilenameParser, detect_parser, create_parser
from ezstitcher.core.image_locator import ImageLocator
from ezstitcher.core.pattern_matcher import PatternMatcher

# Set up logger
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""
    # Input/output configuration
    processed_dir_suffix: str = "_processed"
    post_processed_dir_suffix: str = "_post_processed"
    positions_dir_suffix: str = "_positions"
    stitched_dir_suffix: str = "_stitched"

    # Well filtering
    well_filter: Optional[List[str]] = None

    # Reference processing (for position generation)
    reference_channels: List[str] = field(default_factory=lambda: ["1"])
    reference_preprocessing: Optional[Dict[str, Callable]] = None
    reference_composite_weights: Optional[Dict[str, float]] = None
    reference_z_method: str = "max_projection"  # or "best_focus", "mean_projection"
    focus_method: str = "combined"  # Used for best_focus

    # Final processing (for stitched output)
    # Note: All available channels are always processed and stitched
    preserve_z_planes: bool = False  # If True, preserve Z-stack structure
    final_preprocessing: Optional[Dict[str, Callable]] = None
    final_composite_weights: Optional[Dict[str, float]] = None
    final_z_method: Optional[str] = None  # If None, uses reference_z_method

    # Stitching configuration
    stitcher: StitcherConfig = field(default_factory=StitcherConfig)


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
        self.zstack_processor = ZStackProcessor(ZStackProcessorConfig())
        self.filename_parser = None
        self.pattern_matcher = None
        self.stitcher = None

    def _prepare_images(self, plate_path):
        """
        Prepare images by padding filenames and organizing Z-stack folders.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Path: Path to the image directory
        """
        # Find the image directory
        image_dir = ImageLocator.find_image_directory(plate_path)
        logger.info(f"Found image directory: {image_dir}")

        # Rename files with consistent padding
        self.fs_manager.rename_files_with_consistent_padding(
            image_dir,
            parser=self.filename_parser,
            width=3  # Default padding width
        )

        # Detect and organize Z-stack folders
        has_zstack_folders, _ = self.zstack_processor.detect_zstack_folders(image_dir)
        if has_zstack_folders:
            logger.info(f"Organizing Z-stack folders in {image_dir}")
            self.zstack_processor.organize_zstack_folders(image_dir)

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
            self.filename_parser = create_parser('auto', plate_folder=plate_path)
            self.pattern_matcher = PatternMatcher(filename_parser=self.filename_parser)
            self.stitcher = Stitcher(self.config.stitcher, filename_parser=self.filename_parser)

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
            patterns_by_well = self.pattern_matcher.auto_detect_patterns(dirs['input'], well_filter=self.config.well_filter)

            # Well filter is already applied in auto_detect_patterns

            # Process each well
            for well, wavelength_patterns in patterns_by_well.items():
                self.process_well(well, wavelength_patterns, dirs)

            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return False

    def process_well(self, well, wavelength_patterns, dirs):
        """
        Process a single well through the pipeline.

        Args:
            well: Well identifier
            wavelength_patterns: Dictionary mapping wavelengths to patterns
            dirs: Dictionary of directories
        """
        logger.info(f"Processing well {well}")

        # 1. Process reference images (for position generation)
        self.process_reference_images(well, wavelength_patterns, dirs)

        # 2. Generate stitching positions
        positions_file = self.generate_positions(well, dirs)

        # 3. Process final images (for stitching)
        self.process_final_images(well, wavelength_patterns, dirs)

        # 4. Stitch final images
        self.stitch_images(well, wavelength_patterns, dirs, positions_file)

    def process_reference_images(self, well, wavelength_patterns, dirs):
        """
        Process images for position generation.

        Args:
            well: Well identifier
            wavelength_patterns: Dictionary mapping wavelengths to patterns
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
            pattern = wavelength_patterns[channel]

            # Check if this pattern has Z-stacks
            has_zstack = self.check_for_zstack(dirs['input'], pattern)

            # Apply tile processing
            tile_files = self.process_tiles(
                dirs['input'],
                dirs['processed'],
                pattern,
                channel,
                has_zstack
            )
            processed_files.extend(tile_files)

        # Create composite if multiple reference channels
        #if len(reference_channels) > 1 and self.config.reference_composite_weights:
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

    def process_final_images(self, well, wavelength_patterns, dirs):
        """
        Process images for final stitching.

        Args:
            well: Well identifier
            wavelength_patterns: Dictionary mapping wavelengths to patterns
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
            pattern = wavelength_patterns[channel]

            # Check if this pattern has Z-stacks
            has_zstack = self.check_for_zstack(dirs['input'], pattern)

            # Apply tile processing
            tile_files = self.process_tiles(
                dirs['input'],
                dirs['post_processed'],
                pattern,
                channel,
                has_zstack and not self.config.preserve_z_planes
            )
            processed_files.extend(tile_files)

        # Create composite if needed
        if len(channels_to_process) > 1 and self.config.final_composite_weights:
            composite_files = self.create_composite(
                well,
                dirs['post_processed'],
                {ch: wavelength_patterns[ch] for ch in channels_to_process},
                self.config.final_composite_weights
            )

            # Clean up processed tiles after composition
            if composite_files:
                logger.info(f"Cleaning up processed tiles after final composition")
                self.fs_manager.cleanup_processed_files(processed_files, composite_files)

    def process_tiles(self, input_dir, output_dir, patterns, channel, flatten_zstack=False):
        """
        Process tiles for a specific pattern.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            pattern: List of file pattern
            channel: Channel identifier
            flatten_zstack: Whether to flatten Z-stacks

        Returns:
            list: Paths to created images
        """
        output_files = []
        # Get preprocessing function for this channel
        preprocess_func = None
        if hasattr(self.config, 'preprocessing_funcs') and self.config.preprocessing_funcs:
            preprocess_func = self.config.preprocessing_funcs.get(channel)

        # Check if this pattern has Z-stacks
        has_zstack, z_indices_map = self.zstack_processor.detect_zstack_images(input_dir)

        if has_zstack:
            # Process Z-stacks
            for base_name, z_indices in z_indices_map.items():
                # Load Z-stack
                # Create a dictionary with just this base_name and its z_indices
                single_stack_map = {base_name: z_indices}
                loaded_stacks = self.zstack_processor.load_z_stacks(input_dir, single_stack_map)
                z_stack = loaded_stacks.get(base_name)
                if z_stack is None or len(z_stack) == 0:
                    continue

                # Apply preprocessing if specified
                if preprocess_func:
                    z_stack = self.apply_function_to_stack(z_stack, preprocess_func)

                if flatten_zstack:
                    # Flatten Z-stack
                    z_method = self.config.z_method if hasattr(self.config, 'z_method') else "max_projection"

                    if z_method == "max_projection":
                        result = self.image_preprocessor.max_projection(z_stack)
                    elif z_method == "mean_projection":
                        result = self.image_preprocessor.mean_projection(z_stack)
                    elif z_method == "best_focus":
                        # Find best focused image
                        best_idx, _ = self.zstack_processor.focus_analyzer.find_best_focus(z_stack)
                        result = z_stack[best_idx]
                    else:
                        # Default to max projection
                        result = self.image_preprocessor.max_projection(z_stack)

                    # Save flattened result
                    output_path = output_dir / f"{base_name}.tif"
                    self.fs_manager.save_image(output_path, result)
                else:
                    # Save each Z-plane
                    metadata = self.filename_parser.parse_filename(f"{base_name}_z1.tif")
                    if not metadata:
                        continue

                    for i, z_index in enumerate(z_indices):
                        # Construct filename
                        filename = self.filename_parser.construct_filename(
                            well=metadata['well'],
                            site=metadata['site'],
                            channel=metadata['channel'],
                            z_index=z_index,
                            extension='.tif'
                        )

                        # Save processed image
                        output_path = output_dir / filename
                        self.fs_manager.save_image(output_path, z_stack[i])
        else:
            # Process single images
            # Use PatternMatcher to handle {iii} placeholder in pattern
            #must loop
            for pattern in patterns:
                matching_filenames = self.pattern_matcher.path_list_from_pattern(input_dir, pattern)
                image_files = [input_dir / filename for filename in matching_filenames]

                for img_path in image_files:
                    # Load image
                    image = self.fs_manager.load_image(img_path)
                    if image is None:
                        continue

                    # Apply preprocessing if specified
                    if preprocess_func:
                        image = preprocess_func(image)

                    # Save processed image
                    output_path = output_dir / img_path.name
                    self.fs_manager.save_image(output_path, image)
                    output_files.append(output_path)

        return output_files

    def create_composite(self, well, input_dir, channel_patterns, weights=None):
        """
        Create a composite image from multiple channels or save reference channel.

        Args:
            well: Well identifier
            input_dir: Input directory
            channel_patterns: Dictionary mapping channels to patterns
            weights: Dictionary mapping channels to weights, or None to use first channel as reference

        Returns:
            list: Paths to created composite images
        """
        output_files = []
        # Find all images by site and channel
        site_images = {}
        site_z_indices = {}

        # Process each channel
        for channel, patterns in channel_patterns.items():
            # Handle both single pattern (string) and multiple patterns (list)
            pattern_list = patterns if isinstance(patterns, list) else [patterns]

            # Find all matching files for this channel
            for pattern in pattern_list:
                matching_filenames = self.pattern_matcher.path_list_from_pattern(input_dir, pattern)

                # Process each file
                for filename in matching_filenames:
                    metadata = self.filename_parser.parse_filename(filename)
                    if metadata and 'site' in metadata:
                        site = metadata['site']
                        file_path = input_dir / filename

                        # Initialize site data structures if needed
                        if site not in site_images:
                            site_images[site] = {}
                            site_z_indices[site] = {}

                        # Load the image
                        image = self.fs_manager.load_image(file_path)
                        if image is not None:
                            site_images[site][channel] = image

                            # Track Z-index if present
                            if metadata and 'z_index' in metadata:
                                site_z_indices[site][channel] = metadata['z_index']

        # Process each site
        for site, images in site_images.items():
            if not images:
                continue

            # Create standardized output filename with Z-suffix if needed
            z_suffix = ''
            if site in site_z_indices and site_z_indices[site]:
                # Use the z-index from the first channel
                first_channel = next(iter(site_z_indices[site].keys()))
                z_suffix = f"z{int(site_z_indices[site][first_channel]):03d}"

            # Create output path with standardized naming
            output_filename = f"{well}_s{int(site):03d}{z_suffix}.tif"
            output_path = input_dir / output_filename

            # Save the image (composite or reference)
            if weights and len(images) == len(channel_patterns) and all(channel in images for channel in weights):
                # Create composite if weights are provided and all channels are available
                composite = self.image_preprocessor.create_composite(images, weights)
                self.fs_manager.save_image(output_path, composite)
            else:
                # Otherwise, use the first available channel as reference
                reference_channel = next(iter(images.keys()))
                self.fs_manager.save_image(output_path, images[reference_channel])

            output_files.append(output_path)

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

        # Get grid dimensions
        grid_dims = self.get_grid_dimensions(dirs['input'])

        # Ensure positions directory exists
        self.fs_manager.ensure_directory(dirs['positions'])

        # Generate positions file path with well name and .csv extension
        positions_file = dirs['positions'] / f"{well}.csv"

        # Use standardized reference pattern (without channel information)
        # After processing, all images are saved with well_s### format
        reference_pattern = f"{well}_s{{iii}}.tif"

        # Generate positions
        self.stitcher.generate_positions(
            dirs['processed'],
            reference_pattern,
            positions_file,  # Pass the file path, not the directory
            grid_dims[0],
            grid_dims[1],
        )

        return positions_file

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
        patterns_by_well = self.pattern_matcher.auto_detect_patterns(dirs['post_processed'], well_filter=[well])

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
            matching_files = self.pattern_matcher.path_list_from_pattern(dirs['post_processed'], pattern)

            if not matching_files:
                logger.warning(f"No files found for pattern {pattern}")
                continue

            # Extract pattern suffix to determine output filename
            # Example: A01_s{iii}_w1.tif -> _w1.tif
            suffix = pattern.replace(f"{well}_s{{iii}}", "")

            # Create output filename based on the pattern
            output_filename = f"{well}{suffix}"
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
        # ZStackProcessor.detect_zstack_images doesn't accept a pattern parameter
        # We'll just check if the directory has any Z-stack images
        has_zstack, _ = self.zstack_processor.detect_zstack_images(input_dir)
        return has_zstack

    def apply_function_to_stack(self, z_stack, func):
        """
        Apply a function to a Z-stack, handling both stack and single-image functions.

        Args:
            z_stack: Z-stack of images
            func: Function to apply

        Returns:
            Processed Z-stack
        """
        try:
            # Try to apply to the whole stack
            result = func(z_stack)
            if isinstance(result, list) or (isinstance(result, np.ndarray) and result.ndim > 2):
                return result
        except:
            pass

        # Apply to each image individually
        return [func(img) for img in z_stack]

    def get_grid_dimensions(self, input_dir):
        """
        Get grid dimensions for stitching.

        Args:
            input_dir: Input directory

        Returns:
            tuple: (grid_size_x, grid_size_y)
        """
        # Find HTD file
        htd_file = self.fs_manager.find_htd_file(input_dir)

        if htd_file:
            # Parse HTD file
            parsed = self.fs_manager.parse_htd_file(htd_file)
            if parsed:
                grid_size_x, grid_size_y = parsed
                logger.info(f"Using grid dimensions from HTD file: {grid_size_x}x{grid_size_y}")
                return grid_size_x, grid_size_y

        # Default grid dimensions
        logger.warning("Using default grid dimensions: 2x2")
        return 2, 2
