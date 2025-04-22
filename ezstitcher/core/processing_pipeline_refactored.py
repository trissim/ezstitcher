import logging
import os
import copy
import concurrent.futures
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable, Any

from ezstitcher.core.microscope_interfaces import create_microscope_handler
from ezstitcher.core.image_locator import ImageLocator
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig

# Import the new pipeline architecture
from ezstitcher.core.pipeline import Step, Pipeline

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
        Process images for position generation using the new Pipeline architecture.

        Args:
            well: Well identifier
            dirs: Dictionary of directories
        """
        logger.info("Processing reference images for well %s", well)

        # Create reference image processing pipeline
        reference_pipeline = Pipeline(
            steps=[
                # Step 1: Flatten Z-stacks
                Step(
                    func=self.image_preprocessor.create_projection,
                    variable_components=['z_index'],
                    processing_args={
                        'method': self.config.reference_flatten,
                        'focus_analyzer': self.focus_analyzer
                    },
                    name="Z-Stack Flattening",
                    input_dir=dirs['input'],  # Use input (which is the workspace)
                    output_dir=dirs['processed'],  # Output to processed
                ),

                # Step 2: Process channels
                Step(
                    func=getattr(self.config, 'reference_processing', None),
                    variable_components=['site'],
                    group_by='channel',
                    name="Channel Processing"
                ),

                # Step 3: Create composites
                Step(
                    func=self.image_preprocessor.create_composite,
                    variable_components=['channel'],
                    group_by='site',
                    processing_args={'weights': self.config.reference_composite_weights},
                    name="Composite Creation"
                )
            ],
            well_filter=[well],
            name=f"Reference Pipeline - {well}"
        )

        # Run the pipeline with microscope handler
        reference_pipeline.run(
#            input_dir=dirs['input'],  # Pass input directory explicitly
#            output_dir=dirs['processed'],  # Pass output directory explicitly
            well_filter=[well],  # Pass well filter explicitly
            microscope_handler=self.microscope_handler
        )

        logger.info(f"Reference pipeline completed for well {well}")

    def process_final_images(self, well, dirs):
        """
        Process images for final stitching using the new Pipeline architecture.

        Args:
            well: Well identifier
            dirs: Dictionary of directories
        """
        logger.info("Processing final images for well %s", well)

        # Get all available channels
        channels = self._get_available_channels(dirs['processed'], well)
        logger.info("Processing all %d available channels for well %s", len(channels), well)

        # Create final processing pipeline
        final_pipeline = Pipeline(
            steps=[
                # Step 1: Process channels
                Step(
                    func=getattr(self.config, 'final_processing', None),
                    variable_components=['site'],
                    group_by='channel',
                    name="Channel Processing"
                ),

                # Step 2: Flatten Z-stacks
                Step(
                    func=self.image_preprocessor.create_projection,
                    variable_components=['z_index'],
                    processing_args={
                        'method': self.config.stitch_flatten,
                        'focus_analyzer': self.focus_analyzer
                    },
                    name="Z-Stack Flattening"
                )
            ],
            input_dir=dirs['processed'],  # Use processed as input
            output_dir=dirs['post_processed'],  # Output to post_processed
            well_filter=[well],
            name=f"Final Processing Pipeline - {well}"
        )

        # Run the pipeline with microscope handler
        final_pipeline.run(
            input_dir=dirs['processed'],  # Pass input directory explicitly
            output_dir=dirs['post_processed'],  # Pass output directory explicitly
            well_filter=[well],  # Pass well filter explicitly
            microscope_handler=self.microscope_handler
        )

        logger.info("Final processing pipeline completed for well %s", well)

    # _get_processing_functions method removed - now handled by
    # pipeline_core.prepare_patterns_and_functions

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
            if (metadata and 'well' in metadata and
                metadata['well'].lower() == well.lower() and 'channel' in metadata):
                channels.add(str(metadata['channel']))

        return list(channels)

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
            'post_processed': (plate_path.parent /
                f"{plate_path.name}{self.config.post_processed_dir_suffix}"),
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
        # Import time here to avoid circular imports
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
            self.fs_manager.organize_zstack_folders(
                image_dir, filename_parser=self.microscope_handler)
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

        # Try to find a sample pattern to use as reference
        sample_pattern = None

        # Use auto_detect_patterns to find all patterns for this well
        patterns_by_well = self.microscope_handler.auto_detect_patterns(
            dirs['processed'],
            well_filter=[well],  # Filter by well
            variable_components=['site']
        )

        # Extract a sample pattern if available
        if patterns_by_well and well in patterns_by_well:
            all_patterns = []
            for _, patterns in patterns_by_well[well].items():
                all_patterns.extend(patterns)

            if all_patterns:
                sample_pattern = all_patterns[0]

        # Create reference pattern based on sample or fallback to generic
        if sample_pattern:
            # Parse sample pattern to get components
            metadata = self.microscope_handler.parser.parse_filename(sample_pattern)

            # Construct reference pattern with the same format but with {iii} for site
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
        else:
            # No patterns found, fall back to generic pattern
            logger.warning("No patterns found for well %s in %s", well, dirs['processed'])
            reference_pattern = self.microscope_handler.parser.construct_filename(
                well=well,
                site="{iii}",
                extension='.tif',
                site_padding=DEFAULT_PADDING,
                z_padding=DEFAULT_PADDING
            )

        # Log the paths being used for debugging
        logger.info("Using processed directory: %s", dirs['processed'])
        logger.info("Using reference pattern: %s", reference_pattern)
        logger.info("Using positions file: %s", positions_file)

        # Generate positions using the appropriate stitcher
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

        # Use auto_detect_patterns to find all patterns
        patterns_by_well = self.microscope_handler.auto_detect_patterns(
            dirs['post_processed'],
            well_filter=[well],  # Filter by well
            variable_components=['site']
        )

        if not patterns_by_well or well not in patterns_by_well:
            logger.warning("No patterns found for well %s in %s", well, dirs['post_processed'])
            return

        # Stitch each pattern
        all_patterns = []
        for component, patterns in patterns_by_well[well].items():
            all_patterns.extend(patterns)

        for pattern in all_patterns:
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

            # Construct full paths for override_names
            full_path_matching_files = []
            for filename in matching_files:
                # Construct the full path
                full_path = str(dirs['post_processed'] / filename)
                full_path_matching_files.append(full_path)

            # Use the post-processed directory directly
            # since we're not using well-specific subdirectories

            stitcher_to_use.assemble_image(
                positions_path=positions_file,
                images_dir=dirs['post_processed'],  # Use the post-processed directory directly
                output_path=output_path,
                override_names=full_path_matching_files  # Use full paths to the files
            )
