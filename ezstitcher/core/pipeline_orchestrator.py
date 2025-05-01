import logging
import os
import copy
import random
import time
import threading
import concurrent.futures
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple, Any, TYPE_CHECKING, Literal


from ezstitcher.core.microscope_interfaces import create_microscope_handler, MicroscopeHandler
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.io.filemanager import FileManager # Added
from ezstitcher.core.image_processor import ImageProcessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.opera_phenix_xml_parser import OperaPhenixXmlParser

# Import the pipeline architecture
from ezstitcher.core.pipeline import Step, Pipeline, StepExecutionPlan

from ezstitcher.core.pattern_resolver import get_patterns_for_well
from ezstitcher.io.storage_adapter import StorageAdapter, ZarrStorageAdapter

# Type hint for StorageAdapter without circular import issues at runtime
if TYPE_CHECKING:
    from ezstitcher.io.storage_adapter import StorageAdapter

logger = logging.getLogger(__name__)

DEFAULT_PADDING = 3

class PipelineOrchestrator:
    """Orchestrates the complete image processing and stitching pipeline.

    Note: Pattern resolution for wells is now handled by the
    `ezstitcher.core.pattern_resolver.get_patterns_for_well` function.
    """
    def __init__(self,
                 plate_path: Union[str, Path, None] = None,
                 workspace_path: Union[str, Path, None] = None,
                 config: Optional[PipelineConfig] = None,
                 root_dir: Optional[Union[str, Path]] = None,
                 backend: Optional[str] = None,
                 image_preprocessor: Optional[ImageProcessor] = None,
                 focus_analyzer: Optional[FocusAnalyzer] = None,
                 storage_mode: Literal["legacy", "memory", "zarr"] = "legacy", # Changed from storage_adapter
                 storage_root: Optional[Path] = None): # Added storage_root
        """
        Initialize the pipeline orchestrator with dependencies.

        Args:
            plate_path: Path to the source plate data.
            workspace_path: Path to the working directory. If None, derived from plate_path.
            config: Pipeline configuration object.
            root_dir: Root directory for file operations. If None, uses current directory.
            backend: String identifier for the storage backend type (e.g., "filesystem", "memory").
                     If None, defaults to "filesystem".
            image_preprocessor: Instance for image preprocessing steps.
            focus_analyzer: Instance for focus analysis.
            storage_mode: Mode for intermediate storage ('legacy', 'memory', 'zarr').
            storage_root: Root directory for disk-based storage modes (e.g., 'zarr').
        """
        self.config = config or PipelineConfig()
        self.plate_path = Path(plate_path) if plate_path else None
        self._init_fm = FileManager(backend='disk')
        self.storage_mode = storage_mode
        self.storage_adapter: Optional[StorageAdapter] = None
        self.storage_root = storage_root



        if workspace_path:
            self.workspace_path = workspace_path
        elif self.plate_path:
            # Example: Default workspace next to plate path
            self.workspace_path = self.plate_path.parent / f"{self.plate_path.name}_workspace"

        # --- Initialize core attributes ---
        self.microscope_handler = None
        self.input_dir: Optional[Path] = None
        self.stitcher = None

        if self.plate_path and self.workspace_path:
            try:
                # 1. Mirror plate_path to workspace_path first
                logger.info("Mirroring plate directory to workspace...")
                workspace_fm = FileManager(backend="disk")
                num_links = workspace_fm.mirror_directory_with_symlinks(
                    source_dir=self.plate_path,
                    target_dir=self.workspace_path,
                    recursive=True,
                    overwrite=True
                )
                logger.info("Created %d symlinks in workspace", num_links)

                # 2. Set input_dir to workspace_path for all subsequent operations
                self.input_dir = self.workspace_path

                # 3. Initialize file stuff
                self.initialize_storage_adapter()
                self.initialize_file_manager()

                # 4. Initialize microscope handler using workspace
                logger.info("Initializing microscope handler using workspace...")
                self.microscope_handler = create_microscope_handler(
                    microscope_type='auto',
                    plate_folder=self.workspace_path,
                    file_manager=self.file_manager
                )

                logger.info("Using microscope handler: %s", type(self.microscope_handler).__name__)



                # 5. Initialize stitcher
                self.stitcher = Stitcher(
                    config=self.config.stitcher,
                    filename_parser=self.microscope_handler.parser,
                    file_manager=self.file_manager
                )
            except Exception as e:
                logger.error("Failed during initialization: %s", e)
                raise
        else:
            logger.warning("Skipping workspace creation: plate_path or workspace_path not provided.")

        # --- Initialize other dependencies ---
        self.image_preprocessor = image_preprocessor or ImageProcessor()
        self.focus_analyzer = focus_analyzer or FocusAnalyzer()



        logger.info("PipelineOrchestrator initialized.")


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
            source_path_for_metadata = self.input_dir or self.plate_path
            if not source_path_for_metadata:
                 raise ValueError("Cannot determine grid size or pixel size without plate_path or input_dir.")

            # Create a context for metadata access
            from ezstitcher.core.pipeline import ProcessingContext
            temp_context = ProcessingContext(orchestrator=self)

            # Get grid dimensions with context and fallback to defaults if needed
            try:
                self.config.grid_size = self.microscope_handler.get_grid_dimensions(
                    source_path_for_metadata,
                    context=temp_context
                )
                logger.info("Grid size: %s", self.config.grid_size)
            except Exception as e:
                logger.warning("Failed to get grid dimensions: %s. Using default (4x4).", e)
                self.config.grid_size = (4, 4)  # Default dimensions

            # Get pixel size with context and fallback to defaults if needed
            try:
                self.config.pixel_size = self.microscope_handler.get_pixel_size(
                    source_path_for_metadata,
                    context=temp_context
                ) or self.config.stitcher.pixel_size
            except Exception as e:
                logger.warning("Failed to auto-detect pixel size from %s: %s. Using default.",
                              source_path_for_metadata, e)
                self.config.pixel_size = self.config.stitcher.pixel_size
            logger.info("Pixel size: %s", self.config.pixel_size)

            self.input_dir = self.microscope_handler.post_workspace(self.workspace_path)
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

            # --- Persist storage adapter if needed ---
            if self.storage_adapter and hasattr(self.storage_adapter, 'persist'):
                # Import here to avoid circular imports
                from ezstitcher.io.storage_adapter import resolve_persist_path

                # Check if there are any keys to persist
                keys = self.storage_adapter.list_keys()
                if not keys:
                    logger.warning("No keys found in storage adapter, nothing to persist")
                    return True

                # Log the keys that will be persisted
                logger.debug("Storage adapter contains %d keys to persist: %s",
                            len(keys), keys[:5] if len(keys) > 5 else keys)

                # Resolve the appropriate persist path
                persist_target_dir = resolve_persist_path(
                    storage_mode=self.storage_mode,
                    workspace_path=self.workspace_path,
                    storage_root=self.storage_root
                )

                # Only persist if we have a valid path (None means persist is not applicable)
                if persist_target_dir:
                    logger.info("Persisting %d keys using %s to %s...",
                               len(keys), type(self.storage_adapter).__name__, persist_target_dir)
                    try:
                        # Ensure the target directory exists before persisting
                        persist_target_dir.mkdir(parents=True, exist_ok=True)
                        self.storage_adapter.persist(persist_target_dir)
                        logger.info("Persistence complete to %s.", persist_target_dir)
                    except Exception as e:
                        logger.error("Storage adapter persist failed: %s", e, exc_info=True)
                        # Warn but don't fail the whole run
                else:
                    logger.debug("No persist path resolved for %s mode, skipping persist",
                                self.storage_mode)


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

        disk_fm = FileManager(backend='disk')
        try:
            # Use injected file_manager instance
            image_paths = disk_fm.list_image_files(input_dir, recursive=True)
            logger.info("Found %d image files. Extracting well info...", len(image_paths))

            if not self.microscope_handler:
                logger.error("Microscope handler not initialized, cannot parse filenames.")
                return []

            for img_path in image_paths:
                # Ensure img_path.name is passed if parser expects only filename string
                metadata = self.microscope_handler.parse_filename(img_path.name)
                if metadata and 'well' in metadata:
                    all_wells.add(metadata['well'])

        except Exception as e:
            logger.error("Error listing or parsing files in %s: %s", input_dir, e)
            return [] # Return empty on error

        sorted_wells = sorted(list(all_wells))
        logger.info("Found %d wells: %s in %.2fs",
                   len(sorted_wells), sorted_wells, time.time() - start_time)

        # Apply filtering logic (part of orchestrator's responsibility)
        if self.config.well_filter:
            # Ensure filter is a set for efficient lookup
            well_filter_set = set(self.config.well_filter)
            filtered_wells = [w for w in sorted_wells if w in well_filter_set]
            logger.info("Filtered to %d wells: %s", len(filtered_wells), filtered_wells)
            return filtered_wells

        return sorted_wells

    def prepare_pipeline_paths(self, pipeline, path_overrides=None):
        """
        Compute input/output directories for all steps in the pipeline.

        This method centralizes all path resolution logic. It applies the following rules:
        1. First step uses workspace_path as input_dir
        2. Subsequent steps use previous step's output_dir as input_dir
        3. Special steps (PositionGenerationStep, ImageStitchingStep) get special output directories
        4. Path overrides take precedence over default rules

        Args:
            pipeline: The pipeline to prepare paths for
            path_overrides: Optional dictionary of path overrides with keys like:
                            "{step_id}_input_dir" or "{step_id}_output_dir"

        Returns:
            dict: Dictionary mapping step IDs to StepExecutionPlan objects
        """
        from ezstitcher.core.steps import PositionGenerationStep, ImageStitchingStep

        path_overrides = path_overrides or {}
        step_plans = {}
        prev_output_dir = None

        for i, step in enumerate(pipeline.steps):
            step_id = id(step)
            step_name = step.name
            step_type = type(step).__name__

            # Apply input directory rules
            if i == 0:
                # First step's input_dir is workspace_path
                input_dir = self.input_dir
            else:
                # Subsequent steps use previous step's output_dir
                input_dir = prev_output_dir

            # Apply output directory rules based on step type
            if isinstance(step, PositionGenerationStep):
                # Position generation step
                plate_name = self.plate_path.name
                output_dir = self.plate_path.parent / f"{plate_name}{self.config.positions_dir_suffix}"
            elif isinstance(step, ImageStitchingStep):
                # Image stitching step
                plate_name = self.plate_path.name
                output_dir = self.plate_path.parent / f"{plate_name}{self.config.stitched_dir_suffix}"
            else:
                # Normal step
                if i == 0:
                    # First step gets a unique output directory
                    plate_name = self.plate_path.name
                    output_dir = self.plate_path.parent / f"{plate_name}{self.config.out_dir_suffix}"
                else:
                    # Subsequent normal steps use in-place processing by default
                    output_dir = input_dir

            # Note: Inline attribute overrides have been removed as part of the refactoring
            # to use StepExecutionPlan and context.get_step_input_dir()/context.get_step_output_dir()

            # Apply final overrides (highest priority)
            input_override_key = f"{step_id}_input_dir"
            output_override_key = f"{step_id}_output_dir"

            if input_override_key in path_overrides:
                input_dir = path_overrides[input_override_key]

            if output_override_key in path_overrides:
                output_dir = path_overrides[output_override_key]

            # Create execution plan
            plan = StepExecutionPlan(
                step_id=step_id,
                step_name=step_name,
                step_type=step_type,
                input_dir=input_dir,
                output_dir=output_dir
            )

            # Store the plan
            step_plans[step_id] = plan

            # Update for next iteration
            prev_output_dir = output_dir

        return step_plans

    def create_context(self, pipeline, well_filter=None, path_overrides=None):
        """
        Create a processing context for a pipeline with pre-computed paths.

        This method creates an immutable context with all paths resolved.
        Once created, the context's paths cannot be modified.

        Args:
            pipeline: The pipeline to create context for
            well_filter: Optional well filter
            path_overrides: Optional dictionary of path overrides

        Returns:
            ProcessingContext: The initialized context with immutable paths
        """
        from ezstitcher.core.pipeline import ProcessingContext

        # Compute paths for all steps
        step_plans = self.prepare_pipeline_paths(pipeline, path_overrides)

        # Create a fresh context
        context = ProcessingContext(
            well_filter=well_filter or self.config.well_filter,
            config=self.config,
            orchestrator=self
        )

        # Add execution plans to context
        for step in pipeline.steps:
            plan = step_plans.get(id(step))
            if plan:
                context.add_step_plan(step, plan)

        return context

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

                # Create context with pre-computed paths and path overrides
                context = self.create_context(
                    pipeline,
                    well_filter=[well],
                    path_overrides=getattr(pipeline, 'path_overrides', None)
                )

                # Run the pipeline with the context
                pipeline.run(context)

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
        # Pass file_manager to ensure the Stitcher has access to it
        return Stitcher(
            config=self.config.stitcher,
            filename_parser=self.microscope_handler.parser,
            file_manager=self.file_manager
        )


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
        # Use file_manager to ensure directory
        self.file_manager.ensure_directory(positions_dir)
        positions_file = positions_dir / Path(f"{well}.csv")

        # Get patterns and create reference pattern
        all_patterns = get_patterns_for_well(well, input_dir, self.microscope_handler)
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


        # Ensure output directory exists using file_manager
        self.file_manager.ensure_directory(output_dir)
        logger.info("Ensured output directory exists: %s", output_dir)

        # Get patterns for this well
        all_patterns = get_patterns_for_well(well, input_dir, self.microscope_handler)
        if not all_patterns:
            raise ValueError(f"No patterns found for well {well} in {input_dir}")

        # Process each pattern
        for pattern in all_patterns:
            # Find all matching files and skip if none found
            matching_files = self.microscope_handler.parser.path_list_from_pattern(
                input_dir, pattern)
            if not matching_files:
                logger.warning("No files found for pattern %s, skipping", pattern)
                continue

            # Create output filename and path
            output_path = output_dir / self._create_output_filename(pattern)
            logger.info("Stitching pattern %s to %s", pattern, output_path)

            # Assemble the stitched image
            stitcher_to_use.assemble_image(
                positions_path=positions_file,
                images_dir=input_dir,
                output_path=output_path,
                override_names=[str(input_dir / f) for f in matching_files]
            )

    def initialize_storage_adapter(self):
        """
        Initialize the storage adapter based on the storage_mode.

        This method creates a StorageAdapter instance based on the storage_mode:
        - "legacy": No adapter is created (None)
        - "memory": MemoryStorageAdapter is created
        - "zarr": ZarrStorageAdapter is created

        The storage_root is used as the root directory for the adapter.
        If no storage_root is provided, the input_dir is used.
        """
        if self.storage_mode != "legacy":
            # Determine the effective storage root
            effective_storage_root = self.storage_root or self.input_dir
            if not effective_storage_root:
                raise ValueError("Zarr/memory mode needs a valid root path.")

            # For zarr mode, create a specific subdirectory if no storage_root was provided
            if self.storage_mode == "zarr" and not self.storage_root:
                effective_storage_root = self.input_dir / "zarr_storage"
                logger.warning("Defaulting zarr root to %s", effective_storage_root)

            # Import here to avoid circular imports
            from ezstitcher.io.storage_adapter import select_storage

            # Create the storage adapter
            self.storage_adapter = select_storage(
                mode=self.storage_mode,
                storage_root=effective_storage_root
            )

            # Verify the adapter was created successfully
            if self.storage_adapter is None:
                raise RuntimeError(
                    f"Failed to create StorageAdapter for mode '{self.storage_mode}'"
                )

            logger.info("Initialized StorageAdapter: %s at %s",
                       type(self.storage_adapter).__name__, effective_storage_root)
        else:
            self.storage_adapter = None
            logger.info("No StorageAdapter: legacy mode.")

    def initialize_file_manager(self):
        """
        Initialize the FileManager with the appropriate backend.

        This method creates a FileManager instance with a backend based on the storage_mode:
        - "memory": Uses a memory backend for testing
        - All other modes: Uses a filesystem backend

        The input_dir is used as the root directory for the FileManager.
        """
        # Always use filesystem backend for FileManager unless explicitly testing memory
        if self.storage_mode == "memory":
            backend = "memory"
        else:
            backend = "filesystem"

        self.file_manager = FileManager(backend=backend, root_dir=self.input_dir)
        logger.info("Initialized FileManager with backend '%s'", backend)
