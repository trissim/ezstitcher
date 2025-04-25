import logging
import os
import copy
import time
import threading
import concurrent.futures
from pathlib import Path


from ezstitcher.core.microscope_interfaces import create_microscope_handler
from ezstitcher.core.image_locator import ImageLocator
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.image_processor import ImageProcessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig

# Import the pipeline architecture
from ezstitcher.core.pipeline import Step, Pipeline

logger = logging.getLogger(__name__)

DEFAULT_PADDING = 3

class PipelineOrchestrator:
    """Orchestrates the complete image processing and stitching pipeline."""

    def __init__(self, plate_path=None, workspace_path=None, config=None, fs_manager=None, image_preprocessor=None, focus_analyzer=None):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Pipeline configuration
            fs_manager: File system manager
            image_preprocessor: Image preprocessor
            focus_analyzer: Focus analyzer
        """
        self.config = config or PipelineConfig()

        self.plate_path = Path(plate_path)
        self.fs_manager = fs_manager or FileSystemManager()

        # Determine workspace path
        if workspace_path:
            workspace_path_to_use = workspace_path
        else:
            workspace_path_to_use = self.plate_path.parent / f"{self.plate_path.name}_workspace"

        # Convert to Path
        self.workspace_path = Path(workspace_path_to_use)

        logger.info("Detecting microscope type")
        self.microscope_handler = create_microscope_handler('auto', plate_folder=self.plate_path)
        logger.info("Initializing workspace: %s", workspace_path)
        self.microscope_handler.init_workspace(self.plate_path, self.workspace_path)

        logger.info("Preparing images through renaming and dir flattening")
        self.input_dir = self.prepare_images(self.workspace_path)

        self.stitcher = Stitcher(self.config.stitcher, filename_parser=self.microscope_handler.parser)
        self.image_preprocessor = image_preprocessor or ImageProcessor()

        # Initialize focus analyzer
        focus_config = self.config.focus_config or FocusAnalyzerConfig()
        self.focus_analyzer = focus_analyzer or FocusAnalyzer(focus_config)


    def run(self,pipelines=None):
        """
        Process a plate through the complete pipeline.

        Args:
            plate_folder: Path to the plate folder
            pipelines: List of pipelines to run for each well

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Setup
            self.config.grid_size = self.microscope_handler.get_grid_dimensions(self.workspace_path)
            logger.info("Grid size: %s", self.config.grid_size)
            self.config.pixel_size = self.microscope_handler.get_pixel_size(self.workspace_path)
            logger.info("Pixel size: %s", self.config.pixel_size)

            # Directory setup is handled within pipelines now.

            # Get wells to process
            wells = self._get_wells_to_process()

            # Process wells using ThreadPoolExecutor
            num_workers = self.config.num_workers
            # Use only one worker if there's only one well
            effective_workers = min(num_workers, len(wells)) if len(wells) > 0 else 1
            # Check if pipelines are provided
            if pipelines:
                logger.info("Using provided pipelines for processing")
            else:
                logger.info("No pipelines provided, using pipeline functions")

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
                    # Deep copy pipelines for thread safety as they might be stateful
                    # and lack a specific clone method. Requires 'import copy'.
                    copied_pipelines = [copy.deepcopy(p) for p in pipelines] if pipelines else []

                    logger.info("Submitting well %s to thread pool", well)
                    future = executor.submit(
                        self.process_well,
                        well,
                        copied_pipelines # Pass copied pipelines
                    )
                    future_to_well[future] = well

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_well):
                    well = future_to_well[future]
                    try:
                        future.result()  # Get the result (or exception)
                        logger.info("Completed processing well %s", well)

                    except Exception as e:
                        logger.error("Error processing well %s: %s", well, e, exc_info=True)

            # Final cleanup after all wells have been processed
            # Cleanup is now handled by individual pipelines

            return True

        except Exception as e:
            logger.error("Pipeline failed with unexpected error: %s", str(e))
            logger.debug("Exception details:", exc_info=True)
            return False

    def _get_wells_to_process(self):
        """
        Get the list of wells to process based on well filter.

        Args:
            input_dir: Input directory

        Returns:
            list: List of wells to process
        """
        input_dir = self.input_dir
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

    def process_well(self, well, pipelines=None):
        """
        Process a single well through the pipeline.

        Args:
            well: Well identifier
            pipelines: List of cloned pipelines to run sequentially for this well
        """
        logger.info("Processing well %s", well)
        logger.info("Processing well %s with pixel size %s", well, self.config.pixel_size)

        # Add thread ID information for debugging
        thread_id = threading.get_ident()
        thread_name = threading.current_thread().name
        logger.info("Processing well %s in thread %s (ID: %s)", well, thread_name, thread_id)

        # Stitcher instances will be provided on demand by the orchestrator
        # via the get_stitcher() method, if needed by pipeline steps.

        # Run the pipelines sequentially (list received is already copied)
        if pipelines:
            logger.info("Running %d pipelines for well %s", len(pipelines), well)
            for i, pipeline in enumerate(pipelines):
                logger.info("Running pipeline %d/%d for well %s: %s",
                           i+1, len(pipelines), well, pipeline.name)

                # Orchestrator is passed, allowing pipelines/steps to call
                # orchestrator.get_stitcher() if they need one.
                pipeline.run(
                    well_filter=[well],
                    orchestrator=self
                )

            logger.info("All pipelines completed for well %s", well)
        else:
            logger.warning("No pipelines provided for well %s", well)



    def get_stitcher(self):
        """
        Provides a new Stitcher instance configured for the current run.
        This ensures thread safety by creating a new instance on demand.

        Returns:
            Stitcher: A new Stitcher instance.
        """
        logger.debug("Creating new Stitcher instance for requestor.")
        # Ensure the stitcher is configured using the orchestrator's config
        return Stitcher(self.config.stitcher, filename_parser=self.microscope_handler.parser)

    def prepare_images(self, plate_path):
        """
        Prepare images by padding filenames and organizing Z-stack folders.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Path: Path to the image directory
        """
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

    def _get_patterns_for_well(self, well, directory):
        """
        Get patterns for a specific well from a directory.
        """
        patterns_by_well = self.microscope_handler.auto_detect_patterns(
            directory, well_filter=[well], variable_components=['site']
        )

        # Extract and flatten all patterns for this well
        all_patterns = []
        if patterns_by_well and well in patterns_by_well:
            for _, patterns in patterns_by_well[well].items():
                all_patterns.extend(patterns)

        return all_patterns

    def _get_reference_pattern(self, well, sample_pattern):
        """
        Create a reference pattern for stitching.
        """
        if not sample_pattern:
            raise ValueError(f"No pattern found for well {well}")

        metadata = self.microscope_handler.parser.parse_filename(sample_pattern)
        if not metadata:
            raise ValueError(f"Could not parse pattern: {sample_pattern}")

        return self.microscope_handler.parser.construct_filename(
            well=metadata['well'],
            site="{iii}",
            channel=metadata.get('channel'),
            z_index=metadata.get('z_index'),
            extension=metadata['extension'],
            site_padding=DEFAULT_PADDING,
            z_padding=DEFAULT_PADDING
        )

    def generate_positions(self, well, input_dir, positions_dir):
        """
        Generate stitching positions for a well using a dedicated stitcher instance.
        """
        logger.info("Generating positions for well %s", well)
        # Get a dedicated stitcher instance for this operation
        stitcher_to_use = self.get_stitcher()
        positions_dir = Path(positions_dir)
        input_dir = Path(input_dir)
        self.fs_manager.ensure_directory(positions_dir)
        positions_file = positions_dir / Path(f"{well}.csv")

        # Get patterns and create reference pattern
        all_patterns = self._get_patterns_for_well(well, input_dir)
        if not all_patterns:
            raise ValueError(f"No patterns found for well {well}")

        ### currently only support 1 set of positions per well (more makes no sense)
        reference_pattern = self._get_reference_pattern(well, all_patterns[0])

        # Generate positions
        stitcher_to_use.generate_positions(
            input_dir,
            reference_pattern,
            positions_file,
            self.config.grid_size[0],
            self.config.grid_size[1],
        )

        return positions_file, reference_pattern

    def _create_output_filename(self, pattern):
        """
        Create an output filename for a stitched image based on a pattern.
        """
        parsable = pattern.replace('{iii}', '001')
        metadata = self.microscope_handler.parser.parse_filename(parsable)

        if not metadata:
            raise ValueError(f"Could not parse pattern: {pattern}")

        return self.microscope_handler.parser.construct_filename(
            well=metadata['well'],
            site=metadata['site'],
            channel=metadata['channel'],
            z_index=metadata.get('z_index', 1),
            extension='.tif',
            site_padding=DEFAULT_PADDING,
            z_padding=DEFAULT_PADDING
        )

    def stitch_images(self, well, input_dir, output_dir, positions_file):
        """
        Stitch images for a well using a dedicated stitcher instance.
        """
        logger.info("Stitching images for well %s", well)
        # Get a dedicated stitcher instance for this operation via the orchestrator's method
        stitcher_to_use = self.get_stitcher()
        output_dir = Path(output_dir)
        input_dir = Path(input_dir)

        # Use ImageLocator to find the actual image directory
        actual_input_dir = ImageLocator.find_image_directory(input_dir)
        logger.info("Using actual image directory: %s", actual_input_dir)

        # Ensure output directory exists
        self.fs_manager.ensure_directory(output_dir)
        logger.info("Ensured output directory exists: %s", output_dir)

        # Get patterns for this well
        all_patterns = self._get_patterns_for_well(well, actual_input_dir)
        if not all_patterns:
            raise ValueError(f"No patterns found for well {well} in {actual_input_dir}")

        # Process each pattern
        for pattern in all_patterns:
            # Find all matching files and skip if none found
            matching_files = self.microscope_handler.parser.path_list_from_pattern(
                actual_input_dir, pattern)
            if not matching_files:
                logger.warning("No files found for pattern %s, skipping", pattern)
                continue

            # Create output filename and path
            output_path = output_dir / self._create_output_filename(pattern)
            logger.info("Stitching pattern %s to %s", pattern, output_path)

            # Assemble the stitched image
            stitcher_to_use.assemble_image(
                positions_path=positions_file,
                images_dir=actual_input_dir,
                output_path=output_path,
                override_names=[str(actual_input_dir / f) for f in matching_files]
            )
