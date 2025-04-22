import logging
import os
import copy
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable, Any

from ezstitcher.core.microscope_interfaces import create_microscope_handler
from ezstitcher.core.image_locator import ImageLocator
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig

logger = logging.getLogger(__name__)

DEFAULT_PADDING = 3

class PipelineOrchestrator:
    """Orchestrates the complete image processing and stitching pipeline."""

    def __init__(self, config=None, fs_manager=None, image_preprocessor=None, focus_analyzer=None):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Pipeline configuration
            fs_manager: File system manager
            image_preprocessor: Image preprocessor
            focus_analyzer: Focus analyzer
        """
        self.config = config or PipelineConfig()
        self.fs_manager = fs_manager or FileSystemManager()
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()

        # Initialize focus analyzer
        focus_config = self.config.focus_config or FocusAnalyzerConfig(method=self.config.focus_method)
        self.focus_analyzer = focus_analyzer or FocusAnalyzer(focus_config)

        self.microscope_handler = None
        self.stitcher = None

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
            # Use workspace_path from config if provided, otherwise create a default path
            workspace_path = (
                self.config.workspace_path
                if self.config.workspace_path is not None
                else plate_path.parent / f"{plate_path.name}_workspace"
            )
            logger.info("Detecting microscope type")
            self.microscope_handler = create_microscope_handler('auto', plate_folder=plate_path)
            logger.info("Initializing workspace: %s", workspace_path)
            self.microscope_handler.init_workspace(plate_path, workspace_path)
            self.config.grid_size = self.microscope_handler.get_grid_dimensions(workspace_path)
            logger.info("Grid size: %s", self.config.grid_size)
            self.config.pixel_size = self.microscope_handler.get_pixel_size(workspace_path)
            logger.info("Pixel size: %s", self.config.pixel_size)
            self.stitcher = Stitcher(self.config.stitcher, filename_parser=self.microscope_handler.parser)

            logger.info("Preparing images through renaming")
            # Prepare images (pad filenames and organize Z-stack folders)
            input_dir = self._prepare_images(workspace_path)

            # Create directory structure
            dirs = self._setup_directories(workspace_path, input_dir)

            # Get wells to process
            wells = self._get_wells_to_process(dirs['input'])

            # Process wells using ThreadPoolExecutor
            num_workers = self.config.num_workers
            # Use only one worker if there's only one well
            effective_workers = min(num_workers, len(wells)) if len(wells) > 0 else 1
            logger.info(
                "Processing %d wells using %d worker threads",
                len(wells),
                effective_workers
            )

            # Create a thread pool with the appropriate number of workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                # Submit all well processing tasks
                # Create a mapping of futures to wells
                future_to_well = {}
                logger.info("About to submit %d wells for parallel processing", len(wells))
                for well in wells:
                    # Create a deep copy of dirs for each well to avoid shared state issues
                    well_dirs = copy.deepcopy(dirs)
                    logger.info("Submitting well %s to thread pool", well)
                    future = executor.submit(self.process_well, well, well_dirs)
                    future_to_well[future] = well

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_well):
                    well = future_to_well[future]
                    try:
                        future.result()  # Get the result (or exception)
                        logger.info("Completed processing well %s", well)

                        # Note: We don't have access to the well_dirs here since it's in the thread's scope
                        # We'll skip cleanup here as each thread has its own copy of dirs
                        # The main cleanup will happen at the end of processing all wells

                    except Exception as e:
                        logger.error("Error processing well %s: %s", well, e, exc_info=True)

            # Final cleanup after all wells have been processed
            if self.config.cleanup_processed:
                self.fs_manager.empty_directory(dirs['processed'])
            if self.config.cleanup_post_processed:
                self.fs_manager.empty_directory(dirs['post_processed'])

            return True

        except Exception as e:
            logger.error("Pipeline failed with unexpected error: %s", str(e))
            logger.debug("Exception details:", exc_info=True)
            return False

    def _get_wells_to_process(self, input_dir):
        """
        Get the list of wells to process based on well filter.

        Args:
            input_dir: Input directory

        Returns:
            list: List of wells to process
        """
        import time
        start_time = time.time()
        logger.info("Finding wells to process in %s", input_dir)

        # Auto-detect all wells
        all_wells = set()

        # For Opera Phenix, we can optimize by only looking at a sample of files
        # since all files in a well have the same well ID
        is_opera_phenix = hasattr(self.microscope_handler.parser, 'remap_field_in_filename')

        # Find image files - for Opera Phenix, limit to non-recursive and sample size
        if is_opera_phenix:
            # For Opera Phenix, just look at files in the root directory and immediate subdirectories
            logger.info("Detected Opera Phenix dataset. Using optimized well detection.")
            image_paths = []

        image_paths = ImageLocator.find_images_in_directory(input_dir, recursive=True)

        # Extract wells from filenames
        logger.info("Found %d image files. Extracting well information...", len(image_paths))
        for img_path in image_paths:
            metadata = self.microscope_handler.parse_filename(img_path.name)
            if metadata and 'well' in metadata:
                all_wells.add(metadata['well'])

        # Apply well filter if specified
        if self.config.well_filter:
            # Convert well filter to lowercase for case-insensitive matching
            well_filter_lower = [w.lower() for w in self.config.well_filter]
            wells_to_process = [well for well in all_wells if well.lower() in well_filter_lower]
        else:
            wells_to_process = list(all_wells)

        logger.info("Found %d wells in %.2f seconds", len(wells_to_process), time.time() - start_time)
        return wells_to_process

    def process_well(self, well, dirs):
        """
        Process a single well through the pipeline.

        Args:
            well: Well identifier
            dirs: Dictionary of directories
        """
        logger.info("Processing well %s", well)
        logger.info("Processing well %s with pixel size %s", well, self.config.pixel_size)

        # Add thread ID information for debugging
        import threading
        thread_id = threading.get_ident()
        thread_name = threading.current_thread().name
        logger.info("Processing well %s in thread %s (ID: %s)", well, thread_name, thread_id)

        # Create well-specific directories to avoid conflicts between threads
        # We need to create well-specific subdirectories for processed and post-processed
        well_dirs = copy.deepcopy(dirs)

        # Create well-specific subdirectories
        well_dirs['processed'] = dirs['processed'] / well
        well_dirs['post_processed'] = dirs['post_processed'] / well

        # Ensure the well-specific directories exist
        self.fs_manager.ensure_directory(well_dirs['processed'])
        self.fs_manager.ensure_directory(well_dirs['post_processed'])

        logger.info("Created well-specific directories for well %s: %s", well, well_dirs['processed'])

        # Create a new Stitcher instance for this thread to avoid shared state issues
        thread_stitcher = Stitcher(self.config.stitcher, filename_parser=self.microscope_handler.parser)

        # 1. Process reference images (for position generation)
        self.process_reference_images(well, well_dirs)

        # 2. Generate stitching positions
        positions_file, stitch_pattern = self.generate_positions(well, well_dirs, thread_stitcher)

        # 3. Process final images (for stitching)
        self.process_final_images(well, well_dirs)

        # 4. Stitch final images
        self.stitch_images(well, well_dirs, positions_file, thread_stitcher)



    def process_reference_images(self, well, dirs):
        """
        Process images for position generation.

        Args:
            well: Well identifier
            dirs: Dictionary of directories
        """
        logger.info("Processing reference images for well %s", well)

        # Determine which channels to use as reference
        reference_channels = self.config.reference_channels

        # Get reference processing functions from config
        processing_funcs = {}
        for channel in reference_channels:
            channel_funcs = self._get_processing_functions(
                getattr(self.config, 'reference_processing', None),
                channel
            )
            if channel_funcs:
                processing_funcs[channel] = channel_funcs


        # Flatten Z-stacks if needed - use create_projection directly
        self.process_patterns_with_variable_components(
            input_dir=dirs['input'],
            output_dir=dirs['processed'],
            well_filter=[well],
            variable_components=['z_index'],
            processing_funcs=self.image_preprocessor.create_projection,
            processing_args={
                'method': self.config.reference_flatten,
                'focus_analyzer': self.focus_analyzer
            }
        ).get(well, [])

        # Process reference images
        self.process_patterns_with_variable_components(
            input_dir=dirs['processed'],
            output_dir=dirs['processed'],
            well_filter=[well],
            variable_components=['site'],
            group_by='channel',
            processing_funcs=processing_funcs
        ).get(well, [])

        # Create composites in one step
        self.process_patterns_with_variable_components(
            input_dir=dirs['processed'],
            output_dir=dirs['processed'],
            well_filter=[well],
            variable_components=['channel'],
            group_by='site',
            processing_funcs=self.image_preprocessor.create_composite,
            processing_args={'weights': self.config.reference_composite_weights}
        ).get(well, [])


    def process_final_images(self, well, dirs):
        """
        Process images for final stitching.

        Args:
            well: Well identifier
            dirs: Dictionary of directories
        """
        logger.info("Processing final images for well %s", well)

        # Get all available channels
        channels = self._get_available_channels(dirs['input'], well)
        logger.info("Processing all %d available channels for well %s", len(channels), well)

        # Get final processing functions from config
        processing_funcs = {}
        for channel in channels:
            channel_funcs = self._get_processing_functions(
                getattr(self.config, 'final_processing', None),
                channel
            )
            if channel_funcs:
                processing_funcs[channel] = channel_funcs
            else:
                processing_funcs[channel] = []

        # Process final images
        self.process_patterns_with_variable_components(
            input_dir=dirs['input'],
            output_dir=dirs['post_processed'],
            well_filter=[well],
            variable_components=['site'],
            processing_funcs=processing_funcs
        ).get(well, [])

        # Flatten Z-stacks if needed - use create_projection directly
        self.process_patterns_with_variable_components(
            input_dir=dirs['post_processed'],
            output_dir=dirs['post_processed'],
            well_filter=[well],
            variable_components=['z_index'],
            processing_funcs=self.image_preprocessor.create_projection,
            processing_args={
                'method': self.config.stitch_flatten,
                'focus_analyzer': self.focus_analyzer
            }
        ).get(well, [])

    def _get_processing_functions(self, functions, channel=None):
        """
        Get processing functions for a channel.

        Args:
            functions: Processing functions (callable, list, or dict)
            channel: Optional channel to get specific functions for

        Returns:
            List of processing functions or None if no functions are defined
        """
        if functions is None:
            return None

        if callable(functions) or isinstance(functions, list):
            # If functions is a callable or list of functions, apply to all channels
            result = functions
        elif isinstance(functions, dict) and channel is not None and channel in functions:
            # If functions is a dict, get functions for the specified channel
            result = functions[channel]
        else:
            return None

        # Convert single function to list for consistent handling
        if callable(result):
            return [result]
        return result

    def _get_available_channels(self, input_dir, well):
        """
        Get all available channels for a well.

        Args:
            input_dir: Input directory
            well: Well identifier

        Returns:
            list: List of available channels
        """
        # Find all image files for this well
        image_paths = ImageLocator.find_images_in_directory(input_dir, recursive=True)

        # Extract channels from filenames
        channels = set()
        for img_path in image_paths:
            metadata = self.microscope_handler.parse_filename(img_path.name)
            if metadata and 'well' in metadata and metadata['well'].lower() == well.lower() and 'channel' in metadata:
                channels.add(str(metadata['channel']))

        return list(channels)

    def _group_patterns_by_component(self, input_dir, patterns, component):
        """
        Group patterns by a specific component extracted from matching files.

        Args:
            input_dir (str or Path): Input directory
            patterns (list): List of file patterns
            component (str): Component to group by (e.g., 'channel', 'z_index', 'well')

        Returns:
            dict: Dictionary mapping component values to patterns
        """
        # For flat patterns, determine all unique component values
        component_to_patterns = {}

        # First, get all unique component values from the patterns
        unique_values = set()
        for pattern in patterns:
            sample_files = self.microscope_handler.parser.path_list_from_pattern(input_dir, pattern)

            for file_path in sample_files[:5]:  # Limit to first 5 files for efficiency
                # Ensure we're passing just the filename, not the full path
                filename = os.path.basename(file_path)
                metadata = self.microscope_handler.parser.parse_filename(filename)
                if metadata and component in metadata:
                    unique_values.add(str(metadata[component]))

        # If no values found, use a default
        if not unique_values:
            unique_values = ["1"]

        # Assign all patterns to each component value
        for value in unique_values:
            component_to_patterns[value] = patterns

        return component_to_patterns

    def _prepare_patterns_and_functions(self, patterns, processing_funcs, component='default'):
        """
        Prepare patterns and processing functions for processing.

        This function handles two main tasks:
        1. Ensuring patterns are in a component-keyed dictionary format
        2. Determining which processing functions to use for each component

        Args:
            patterns (list or dict): Patterns to process, either as a flat list or grouped by component
            processing_funcs (callable, list, dict, optional): Processing functions to apply
            component (str): Component name for grouping (only used for clarity in the result)

        Returns:
            tuple: (grouped_patterns, component_to_funcs)
                - grouped_patterns: Dictionary mapping component values to patterns
                - component_to_funcs: Dictionary mapping component values to processing functions
        """
        # Fast path: If both patterns and processing_funcs are dictionaries with matching keys,
        # they're already properly structured, so return them as is
        if (isinstance(patterns, dict) and isinstance(processing_funcs, dict) and
                set(patterns.keys()).issubset(set(processing_funcs.keys()))):
            return patterns, processing_funcs

        # Ensure patterns are in a dictionary format
        # If already a dict, use as is; otherwise wrap the list in a dictionary
        grouped_patterns = patterns if isinstance(patterns, dict) else {component: patterns}

        # Determine which processing functions to use for each component
        component_to_funcs = {}

        for comp_value in grouped_patterns.keys():
            if processing_funcs is None:
                component_to_funcs[comp_value] = None
            elif isinstance(processing_funcs, dict):
                component_to_funcs[comp_value] = processing_funcs.get(comp_value)
            else:
                component_to_funcs[comp_value] = processing_funcs

        return grouped_patterns, component_to_funcs

    def process_patterns_with_variable_components(self, input_dir, output_dir, well_filter=None,
                                                 variable_components=None, group_by=None,
                                                 processing_funcs=None, processing_args=None):
        """
        Detect patterns with variable components and process them flexibly.

        Args:
            input_dir (str or Path): Input directory containing images
            output_dir (str or Path): Output directory for processed images
            well_filter (list, optional): List of wells to include
            variable_components (list, optional): Components to make variable (e.g., ['site', 'z_index'])
            group_by (str, optional): How to group patterns (e.g., 'channel', 'z_index', 'well')
            processing_funcs (callable, list, dict, optional): Processing functions to apply
            processing_args (dict, optional): Additional arguments to pass to processing functions

        Returns:
            dict: Dictionary mapping wells to processed file paths
        """
        # Default variable components if not specified
        if variable_components is None:
            variable_components = ['site']

        # Default processing args
        if processing_args is None:
            processing_args = {}

        # Auto-detect patterns with the specified variable components
        patterns_by_well = self.microscope_handler.auto_detect_patterns(
            input_dir,
            well_filter=well_filter,
            variable_components=variable_components,
            group_by=group_by
        )

        # Process each well
        results = {}
        for well, patterns in patterns_by_well.items():
            results[well] = []

            # Prepare patterns and functions - works with both grouped and flat patterns
            grouped_patterns, component_to_funcs = self._prepare_patterns_and_functions(
                patterns, processing_funcs, component=group_by or '1'
            )

            # Process each group of patterns with its corresponding function
            # Both dictionaries have the same keys, so we can iterate over the keys
            for component_value in grouped_patterns.keys():
                component_patterns = grouped_patterns[component_value]
                component_func = component_to_funcs[component_value]

                # Process tiles for this component
                files = self.process_tiles(
                    input_dir,
                    output_dir,
                    component_patterns,
                    processing_funcs=component_func,
                    **processing_args
                )
                results[well].extend(files)

        return results

    def _setup_directories(self, plate_path, input_dir):
        """
        Set up directory structure for processing.

        Args:
            plate_path: Path to the plate folder
            input_dir: Path to the input directory

        Returns:
            dict: Dictionary of directories
        """
        # Create main directories
        dirs = {
            'input': input_dir,
            'processed': plate_path.parent / f"{plate_path.name}{self.config.processed_dir_suffix}",
            'post_processed': plate_path.parent / f"{plate_path.name}{self.config.post_processed_dir_suffix}",
            'positions': plate_path.parent / f"{plate_path.name}{self.config.positions_dir_suffix}",
            'stitched': plate_path.parent / f"{plate_path.name}{self.config.stitched_dir_suffix}"
        }

        # Ensure main directories exist
        for dir_path in dirs.values():
            self.fs_manager.ensure_directory(dir_path)

        return dirs

    def _prepare_images(self, plate_path):
        """
        Prepare images by padding filenames and organizing Z-stack folders.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Path: Path to the image directory
        """
        import time
        start_time = time.time()

        # Find the image directory
        image_dir = ImageLocator.find_image_directory(plate_path)
        logger.info("Found image directory: %s", image_dir)

        # Always rename files with consistent padding, even for Opera Phenix datasets
        logger.info("Renaming files with consistent padding...")
        rename_start = time.time()
        self.fs_manager.rename_files_with_consistent_padding(
            image_dir,
            parser=self.microscope_handler,
            width=DEFAULT_PADDING,  # Use consistent padding width
            force_suffixes=True  # Force missing suffixes to be added
        )
        logger.info("Renamed files in %.2f seconds", time.time() - rename_start)

        # Detect and organize Z-stack folders
        zstack_start = time.time()
        has_zstack_folders, z_folders = self.fs_manager.detect_zstack_folders(image_dir)
        if has_zstack_folders:
            logger.info("Found %d Z-stack folders in %s", len(z_folders), image_dir)
            logger.info("Organizing Z-stack folders...")
            self.fs_manager.organize_zstack_folders(image_dir, filename_parser=self.microscope_handler)
            logger.info("Organized Z-stack folders in %.2f seconds", time.time() - zstack_start)

        # Return the image directory (which may have changed if Z-stack folders were organized)
        logger.info("Image preparation completed in %.2f seconds", time.time() - start_time)
        return ImageLocator.find_image_directory(plate_path)

    def generate_positions(self, well, dirs, stitcher=None):
        """
        Generate stitching positions for a well.

        Args:
            well: Well identifier
            dirs: Dictionary of directories
            stitcher: Optional Stitcher instance to use (for thread safety)

        Returns:
            Tuple of (positions_file, stitch_pattern)
        """
        logger.info("Generating positions for well %s", well)

        # Use the provided stitcher or the default one
        stitcher_to_use = stitcher or self.stitcher

        # Ensure positions directory exists
        self.fs_manager.ensure_directory(dirs['positions'])

        # Generate positions file path with well name and .csv extension
        positions_file = dirs['positions'] / f"{well}.csv"

        # Use auto_detect_patterns to find all patterns for this well
        patterns_by_well = self.microscope_handler.auto_detect_patterns(
            dirs['processed'],
            well_filter=[well],
            variable_components=['site']
        )

        if not patterns_by_well or well not in patterns_by_well:
            logger.warning("No patterns found for well %s in %s", well, dirs['processed'])
            # Fall back to a generic pattern if no patterns are found
            reference_pattern = self.microscope_handler.parser.construct_filename(
                well=well,
                site="{iii}",
                extension='.tif',
                site_padding=DEFAULT_PADDING,
                z_padding=DEFAULT_PADDING
            )
        else:
            # Get all patterns for this well
            all_patterns = []
            for _, patterns in patterns_by_well[well].items():
                all_patterns.extend(patterns)

            if not all_patterns:
                logger.warning("No patterns found for well %s in %s", well, dirs['processed'])
                # Fall back to a generic pattern if no patterns are found
                reference_pattern = self.microscope_handler.parser.construct_filename(
                    well=well,
                    site="{iii}",
                    extension='.tif',
                    site_padding=DEFAULT_PADDING,
                    z_padding=DEFAULT_PADDING
                )
            else:
                # Parse a sample pattern to get the components
                sample_pattern = all_patterns[0]
                metadata = self.microscope_handler.parser.parse_filename(sample_pattern)

                # Construct a reference pattern with the same format but with {iii} for site
                reference_pattern = self.microscope_handler.parser.construct_filename(
                    well=metadata['well'],
                    site="{iii}",
                    channel=metadata.get('channel'),
                    z_index=metadata.get('z_index'),
                    extension=metadata['extension'],
                    site_padding=DEFAULT_PADDING,
                    z_padding=DEFAULT_PADDING
                )

                logger.info("Using reference pattern: %s based on detected files", reference_pattern)

        # Generate positions using the appropriate stitcher
        # Log the paths being used for debugging
        logger.info("Using processed directory: %s", dirs['processed'])
        logger.info("Using reference pattern: %s", reference_pattern)
        logger.info("Using positions file: %s", positions_file)

        # We need to use the actual directory containing the files (well-specific directory)
        # and the pattern without the well subfolder
        stitcher_to_use.generate_positions(
            dirs['processed'],  # This is already the well-specific directory
            reference_pattern,  # Use the pattern without the well subfolder
            positions_file,  # Pass the file path, not the directory
            self.config.grid_size[0],
            self.config.grid_size[1],
        )

        return positions_file, reference_pattern

    def stitch_images(self, well, dirs, positions_file, stitcher=None):
        """
        Stitch images for a well.

        Args:
            well: Well identifier
            dirs: Dictionary of directories
            positions_file: Path to positions file
            stitcher: Optional Stitcher instance to use (for thread safety)
        """
        logger.info("Stitching images for well %s", well)

        # Use the provided stitcher or the default one
        stitcher_to_use = stitcher or self.stitcher

        # Use auto_detect_patterns to find all patterns for this well
        patterns_by_well = self.microscope_handler.parser.auto_detect_patterns(
            dirs['post_processed'],
            well_filter=[well],
            variable_components=['site']
        )

        if not patterns_by_well or well not in patterns_by_well:
            logger.warning("No patterns found for well %s in %s", well, dirs['post_processed'])
            return

        # Stitch each pattern
        for pattern in patterns_by_well[well]:
            # Find all matching files for this pattern
            matching_files = self.microscope_handler.parser.path_list_from_pattern(
                dirs['post_processed'], pattern)

            if not matching_files:
                logger.warning("No files found for pattern %s", pattern)
                continue

            # Extract pattern suffix to determine output filename
            parsable = pattern.replace('{iii}','001')
            metadata = self.microscope_handler.parser.parse_filename(parsable)
            output_filename = self.microscope_handler.parser.construct_filename(
                well=metadata['well'],
                site=metadata['site'],
                channel=metadata['channel'],
                z_index=metadata.get('z_index', 1),
                extension='.tif',
                site_padding=DEFAULT_PADDING,
                z_padding=DEFAULT_PADDING
            )

            # Create output filename based on the pattern
            output_path = dirs['stitched'] / output_filename
            logger.info("Stitching pattern %s to %s", pattern, output_path)

            # Assemble the stitched image using the thread-specific stitcher
            # Note: positions_file is already in the main positions directory
            # The images are in the well-specific post-processed directory
            # The output should go to the main stitched directory

            # Log the paths being used for debugging
            logger.info("Using positions file: %s", positions_file)
            logger.info("Using images directory: %s", dirs['post_processed'])
            logger.info("Using output path: %s", output_path)

            # The matching_files are relative to the well-specific post-processed directory
            # We need to construct the full paths for override_names
            full_path_matching_files = []
            for filename in matching_files:
                # The filename is relative to the well-specific post-processed directory
                # We need to construct the full path including the well subfolder
                full_path = str(dirs['post_processed'] / filename)
                full_path_matching_files.append(full_path)

            # The images_dir should be the parent of the post-processed directory
            # This is because the positions file references paths relative to this directory
            # and includes the well subfolder in the paths
            parent_dir = dirs['post_processed'].parent

            stitcher_to_use.assemble_image(
                positions_path=positions_file,
                images_dir=parent_dir,  # Use the parent directory to find the well subfolder
                output_path=output_path,
                override_names=full_path_matching_files  # Use full paths to the files
            )

    def process_tiles(self, input_dir, output_dir, patterns, processing_funcs=None, **kwargs):
        """
        Unified processing using zstack_processor.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            patterns: List of file patterns
            processing_funcs: Processing functions to apply (optional)
            **kwargs: Additional arguments to pass to processing functions

        Returns:
            list: Paths to created images
        """
        output_files = []

        for pattern in patterns:
            matching_files = self.microscope_handler.parser.path_list_from_pattern(input_dir, pattern)
            images = [self.fs_manager.load_image(input_dir / filename) for filename in matching_files]
            images = [img for img in images if img is not None]

            # Apply stack processing functions if specified
            if processing_funcs and images:
                if callable(processing_funcs):
                    # Single function - apply it with kwargs
                    # Don't pass filenames as a positional argument to avoid conflicts
                    images = [processing_funcs(images, **kwargs)]
                else:
                    # List of functions - apply each one
                    for func in processing_funcs:
                        images = self.image_preprocessor.apply_function_to_stack(images, func)

            # Save processed images and delete what we modified
            #clean_up old files if working in place
            if input_dir is output_dir:
                for filename in matching_files:
                    self.fs_manager.delete_file(output_dir/filename)
            # save flattened stack as 1 image if function flattens
            if len(images) != len(matching_files):
                output_path = output_dir / matching_files[0]
                self.fs_manager.save_image(output_path, images[0])
            # if returns stack of same sive save each image with same name
            else:
                for image,file_name in zip(images,matching_files):
                    output_path = output_dir / file_name
                    self.fs_manager.save_image(output_path, image)


        return output_files
