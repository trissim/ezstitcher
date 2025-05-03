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
from ezstitcher.io.overlay import OverlayMode
from ezstitcher.io.storage_config import StorageConfig
from ezstitcher.materialization.flag_engine import FlagInferenceEngine

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
                 storage_mode: Literal["legacy", "memory", "zarr"] = "legacy",
                 storage_root: Optional[Path] = None,
                 overlay_mode: Literal["disabled", "on_demand", "auto"] = "auto",
                 materialization_context: Optional[str] = None,
                 materialization_strategy: Optional[str] = None):
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
            overlay_mode: Mode for overlay disk writes ('disabled', 'on_demand', 'auto').
            materialization_context: Context type for materialization policy ('testing', 'benchmark', 'production').
            materialization_strategy: Materialization strategy ('lazy', 'eager', 'hybrid').
        """
        # Set basic attributes
        self.config = config or PipelineConfig()
        self.plate_path = Path(plate_path) if plate_path else None
        self.storage_mode = storage_mode
        self.storage_root = storage_root

        # Set root context based on workspace_path
        self.root_context = "workspace" if workspace_path else "plate"

        # Set default root context in VirtualPathFactory
        from ezstitcher.io.virtual_path_factory import VirtualPathFactory
        VirtualPathFactory.set_default_root_context(self.root_context)

        # Configure overlay mode
        self.overlay_mode = OverlayMode[overlay_mode.upper()]
        self.overlay_root = None

        # Set workspace_path if provided, otherwise it will be set in initialize()
        self.workspace_path = Path(workspace_path) if workspace_path else None

        # Initialize core attributes to None
        self.file_manager = None
        self.storage_adapter = None
        self.microscope_handler = None
        self.input_dir = None
        self.stitcher = None
        self.materialization_manager = None

        # Store materialization context
        self.materialization_context = materialization_context

        # Set other dependencies
        self.image_preprocessor = image_preprocessor or ImageProcessor()
        self.focus_analyzer = focus_analyzer or FocusAnalyzer()

        # Store backend parameters for later initialization
        self._backend = backend
        self._root_dir = root_dir

        # Set initialization flag
        self._initialized = False

        # Initialize materialization manager if not using legacy mode
        if storage_mode != "legacy":
            from ezstitcher.io.materialization import MaterializationManager, MaterializationPolicy
            policy = MaterializationPolicy.for_context(materialization_context) if materialization_context else None
            self.materialization_manager = MaterializationManager(None, policy)  # Will be updated with context later

        logger.info("PipelineOrchestrator constructed. Call initialize() to set up.")

    def initialize(self):
        """
        Initialize the orchestrator.

        This method sets up the workspace, file manager, storage adapter,
        microscope handler, and stitcher. It should be called after construction
        and before using the orchestrator.

        Returns:
            self: For method chaining
        """
        # Return early if already initialized
        if self._initialized:
            logger.info("Orchestrator already initialized")
            return self

        # Phase 1: Set up workspace_path
        self._initialize_workspace()

        # Phase 2: Initialize file manager
        self._initialize_file_manager()

        # Phase 3: Initialize storage adapter
        self._initialize_storage_adapter()

        # Phase 4: Initialize microscope handler
        self._initialize_microscope_handler()

        # Phase 5: Initialize stitcher
        self._initialize_stitcher()

        # Phase 6: Initialize materialization manager
        self._initialize_materialization_manager()

        # Set initialization flag
        self._initialized = True
        logger.info("PipelineOrchestrator fully initialized.")

        return self

    def _initialize_workspace(self):
        """Initialize workspace path and mirror plate directory if needed."""
        # Set up workspace_path if not provided in constructor
        if not self.workspace_path and self.plate_path:
            self.workspace_path = self.plate_path.parent / f"{self.plate_path.name}_workspace"

        # Mirror plate_path to workspace_path if both are provided
        if self.plate_path and self.workspace_path:
            try:
                # Mirror plate_path to workspace_path
                logger.info("Mirroring plate directory to workspace...")
                workspace_fm = FileManager(backend="disk")
                num_links = workspace_fm.mirror_directory_with_symlinks(
                    source_dir=self.plate_path,
                    target_dir=self.workspace_path,
                    recursive=True,
                    overwrite=True
                )
                logger.info("Created %d symlinks in workspace", num_links)

                # Set input_dir to workspace_path for all subsequent operations
                self.input_dir = self.workspace_path
            except Exception as e:
                logger.error("Failed during workspace initialization: %s", e)
                raise
        else:
            logger.warning("Skipping workspace creation: plate_path or workspace_path not provided.")

    def _initialize_file_manager(self):
        """Initialize the file manager with the appropriate backend."""
        # Skip if already initialized
        if self.file_manager is not None:
            logger.debug("File manager already initialized")
            return

        # Always use filesystem backend for FileManager unless explicitly testing memory
        if self.storage_mode == "memory":
            backend = "memory"
        else:
            backend = "filesystem"

        self.file_manager = FileManager(backend=backend, root_dir=self.input_dir)
        logger.info("Initialized FileManager with backend '%s'", backend)

    def _initialize_storage_adapter(self):
        """Initialize the storage adapter based on the storage mode."""
        # Skip if already initialized or using legacy mode
        if self.storage_adapter is not None or self.storage_mode == "legacy":
            logger.debug("Storage adapter already initialized or not needed (legacy mode)")
            return

        # Ensure file manager is initialized
        self._ensure_file_manager()

        # Normalize storage mode to lowercase
        normalized_storage_mode = self.storage_mode.lower() if isinstance(self.storage_mode, str) else str(self.storage_mode).lower()
        logger.debug(f"Normalized storage mode: {normalized_storage_mode} (from {self.storage_mode})")

        # Determine the effective storage root
        if self.storage_root:
            # If storage_root is explicitly provided, use it
            effective_storage_root = self.storage_root
        elif self.plate_path and self.input_dir:
            # Otherwise, use a suffixed version of input_dir
            plate_name = self.plate_path.name
            # Create a base output directory with the appropriate suffix
            base_output_dir = self.plate_path.parent / f"{plate_name}{self.config.out_dir_suffix}"

            # For zarr mode, create a specific subdirectory
            if normalized_storage_mode == "zarr":
                effective_storage_root = base_output_dir / "zarr_storage"
            # For memory mode, create a specific subdirectory
            elif normalized_storage_mode == "memory":
                effective_storage_root = base_output_dir / "memory_storage"
            else:
                # Fallback for any other mode
                effective_storage_root = base_output_dir / f"{normalized_storage_mode}_storage"

            logger.info(f"Using suffixed storage root: {effective_storage_root}")
        else:
            raise ValueError("Zarr/memory mode needs a valid root path. Either provide storage_root or plate_path.")

        # Ensure the storage root directory exists
        self.file_manager.ensure_directory(effective_storage_root)
        logger.info(f"Ensured storage root directory exists: {effective_storage_root}")

        # Import here to avoid circular imports
        from ezstitcher.io.storage_adapter import select_storage
        from ezstitcher.io.storage_config import StorageConfig

        # Create a StorageConfig instance using self attributes
        storage_config = StorageConfig(
            storage_mode=normalized_storage_mode,
            overlay_mode=self.overlay_mode,
            overlay_root=self.overlay_root
        )

        # Create the storage adapter with the storage_config
        self.storage_adapter = select_storage(
            mode=normalized_storage_mode,
            storage_config=storage_config,
            storage_root=effective_storage_root
        )

        # Verify the adapter was created successfully
        if self.storage_adapter is None:
            raise RuntimeError(
                f"Failed to create StorageAdapter for mode '{normalized_storage_mode}'"
            )

        logger.info("Initialized StorageAdapter: %s at %s",
                   type(self.storage_adapter).__name__, effective_storage_root)

        # Configure overlay for non-legacy storage modes
        if self.storage_adapter is not None:
            # Set overlay root to a subdirectory of the storage root
            self.overlay_root = effective_storage_root / "overlay"

            # Ensure the overlay root directory exists
            self.file_manager.ensure_directory(self.overlay_root)
            logger.info(f"Ensured overlay root directory exists: {self.overlay_root}")

            # Configure overlay in storage adapter
            self.storage_adapter.configure_overlay(self.overlay_mode, self.overlay_root)
            logger.info(f"Configured overlay for storage adapter: mode={self.overlay_mode.name}, root={self.overlay_root}")

    def _initialize_microscope_handler(self):
        """Initialize the microscope handler."""
        # Skip if already initialized
        if self.microscope_handler is not None:
            logger.debug("Microscope handler already initialized")
            return

        # Ensure file manager is initialized
        self._ensure_file_manager()

        # Check if we have a workspace path
        if not self.workspace_path:
            logger.warning("Cannot initialize microscope handler: workspace_path not set")
            return

        # Initialize microscope handler
        logger.info("Initializing microscope handler using workspace...")
        try:
            self.microscope_handler = create_microscope_handler(
                microscope_type='auto',
                plate_folder=self.workspace_path,
                file_manager=self.file_manager,
                pattern_format="ashlar"  # Default to Ashlar for backward compatibility
            )
            logger.info("Using microscope handler: %s", type(self.microscope_handler).__name__)
        except Exception as e:
            logger.error("Failed to initialize microscope handler: %s", e)
            raise

    def _initialize_stitcher(self):
        """Initialize the stitcher."""
        # Skip if already initialized
        if self.stitcher is not None:
            logger.debug("Stitcher already initialized")
            return

        # Ensure microscope handler is initialized
        microscope_handler = self._ensure_microscope_handler()
        file_manager = self._ensure_file_manager()

        # Initialize stitcher
        try:
            self.stitcher = Stitcher(
                config=self.config.stitcher,
                filename_parser=microscope_handler.parser,
                file_manager=file_manager,
                pattern_format="ashlar"  # Default to Ashlar for backward compatibility
            )
            logger.info("Initialized Stitcher")
        except Exception as e:
            logger.error("Failed to initialize stitcher: %s", e)
            raise

    def _initialize_materialization_manager(self):
        """Initialize the materialization manager."""
        # Skip if already initialized or using legacy mode
        if self.materialization_manager is not None or self.storage_mode == "legacy":
            logger.debug("Materialization manager already initialized or not needed (legacy mode)")
            return

        # Skip if storage adapter is not initialized
        if self.storage_adapter is None:
            logger.debug("Storage adapter not initialized, skipping materialization manager initialization")
            return

        # Import here to avoid circular imports
        try:
            from ezstitcher.io.materialization import MaterializationManager, MaterializationPolicy
            from ezstitcher.materialization.flag_engine import FlagInferenceEngine
            from ezstitcher.io.storage_config import StorageConfig

            # Create a policy based on the storage mode
            if self.storage_mode == "memory":
                policy = MaterializationPolicy(force_memory=True)
            elif self.storage_mode == "zarr":
                policy = MaterializationPolicy(respect_flags=True)
            else:
                policy = MaterializationPolicy()

            # Create a flag inference engine
            self.flag_inference_engine = FlagInferenceEngine()

            # Create a storage config
            storage_config = StorageConfig(
                storage_mode=self.storage_mode,
                overlay_mode=self.overlay_mode,
                overlay_root=self.overlay_root
            )

            # Create the materialization manager with keyword arguments only
            self.materialization_manager = MaterializationManager(
                context=self,  # Use self as the context instead of None
                storage_config=storage_config,
                policy=policy,
                engine=self.flag_inference_engine
            )
            logger.info("Initialized MaterializationManager with policy for %s mode", self.storage_mode)
        except ImportError as e:
            logger.warning("Failed to import MaterializationManager: %s", e)
        except Exception as e:
            logger.error("Failed to initialize materialization manager: %s", e)
            # Don't raise, just log the error and continue without materialization manager

    def needs_materialization(self, step, context=None, pipeline=None):
        """
        Check if a step requires materialization.

        Args:
            step: The step to check
            context: The processing context (optional)
            pipeline: The pipeline containing the step (optional)

        Returns:
            True if materialization is needed, False otherwise
        """
        # Check if materialization manager is available
        if self.materialization_manager:
            # Import here to avoid circular imports
            from ezstitcher.io.materialization_resolver import MaterializationResolver
            return MaterializationResolver.needs_materialization(
                step, self.materialization_manager, context, pipeline
            )

        # If no materialization manager is available, check storage mode and overlay mode
        if self.storage_mode == "legacy" or self.overlay_mode == OverlayMode.DISABLED:
            return False

        # Check if the step requires filesystem access
        return (getattr(step, 'requires_fs_input', False) or
                getattr(step, 'requires_fs_output', False) or
                getattr(step, 'force_disk_output', False) or
                getattr(step, 'requires_legacy_fs', False))

    def prepare_materialization(self, step, context):
        """
        Prepare materialization for a step.

        Args:
            step: The step to prepare materialization for
            context: The processing context

        Returns:
            Dictionary mapping original file paths to materialized paths
        """
        # Check if materialization manager is available
        if self.materialization_manager:
            # Update the materialization manager with the context
            self.materialization_manager.context = context

            # Get well from context
            well = context.well_filter[0] if context.well_filter else None
            if not well:
                logger.warning("No well filter found in context, skipping materialization")
                return {}

            # Get input directory for the step
            input_dir = context.get_step_input_dir(step)
            if not input_dir:
                logger.warning("No input directory found for step, skipping materialization")
                return {}

            # Prepare materialization for this step
            return self.materialization_manager.prepare_for_step(step, well, input_dir)

        # If no materialization manager is available, log a warning and return empty dict
        logger.warning("No materialization manager available for step %s", step.name)
        return {}


    def run(self, pipelines=None):
        """
        Process a plate through the complete pipeline.

        This method requires the orchestrator to be initialized first by calling initialize().

        Args:
            pipelines: List of pipelines to run for each well

        Returns:
            bool: True if successful, False otherwise

        Raises:
            RuntimeError: If the orchestrator has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator must be initialized before calling run(). Call initialize() first.")
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

        This method requires the orchestrator to be initialized first by calling initialize().

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

        Raises:
            RuntimeError: If the orchestrator has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator must be initialized before calling prepare_pipeline_paths(). Call initialize() first.")
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

        Args:
            pipeline: The pipeline to create context for
            well_filter: Optional well filter
            path_overrides: Optional dictionary of path overrides

        Returns:
            ProcessingContext: The initialized context with immutable paths

        Raises:
            RuntimeError: If the orchestrator has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator must be initialized before calling create_context(). Call initialize() first.")
        from ezstitcher.core.pipeline import ProcessingContext

        # Compute paths for all steps
        step_plans = self.prepare_pipeline_paths(pipeline, path_overrides)

        # Create a fresh context
        context = ProcessingContext(
            well_filter=well_filter or self.config.well_filter,
            config=self.config,
            orchestrator=self
        )

        # Set root context
        context.set_root_context(self.root_context)

        # Add execution plans to context
        for step in pipeline.steps:
            plan = step_plans.get(id(step))
            if plan:
                context.add_step_plan(step, plan)

        # Update materialization manager with context if available
        if hasattr(self, 'materialization_manager') and self.materialization_manager:
            self.materialization_manager.context = context

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

        This method requires the orchestrator to be initialized first by calling initialize().
        This ensures thread safety by creating a new instance on demand.

        Returns:
            Stitcher: A new Stitcher instance.

        Raises:
            RuntimeError: If the orchestrator has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator must be initialized before calling get_stitcher(). Call initialize() first.")

        # Ensure required components are available
        microscope_handler = self._ensure_microscope_handler()
        file_manager = self._ensure_file_manager()

        logger.debug("Creating new Stitcher instance for requestor.")
        # Create a new Stitcher instance
        return Stitcher(
            config=self.config.stitcher,
            filename_parser=microscope_handler.parser,
            file_manager=file_manager,
            pattern_format="ashlar"  # Default to Ashlar for backward compatibility
        )


    def _get_reference_pattern(self, well, sample_pattern):
        """
        Create a reference pattern for stitching.

        Requires the microscope handler to be initialized.

        Args:
            well: Well identifier
            sample_pattern: Sample filename pattern

        Returns:
            str: Reference pattern for stitching

        Raises:
            RuntimeError: If microscope handler is not initialized
            ValueError: If pattern is invalid
        """
        if not sample_pattern:
            raise ValueError(f"No pattern found for well {well}")

        # Ensure microscope handler is available
        microscope_handler = self._ensure_microscope_handler()

        metadata = microscope_handler.parser.parse_filename(sample_pattern)
        if not metadata:
            raise ValueError(f"Could not parse pattern: {sample_pattern}")

        return microscope_handler.parser.construct_filename(
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

        This method requires the orchestrator to be initialized first by calling initialize().

        Raises:
            RuntimeError: If the orchestrator has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator must be initialized before calling generate_positions(). Call initialize() first.")
        logger.info("Generating positions for well %s", well)

        # Check if we need to use overlay
        needs_overlay = (
            hasattr(self, 'storage_adapter') and
            self.storage_adapter is not None and
            self.storage_mode != "legacy" and
            self.overlay_mode != OverlayMode.DISABLED
        )

        # If using overlay, ensure all required images are written to disk
        if needs_overlay:
            logger.info("Using overlay for generate_positions")
            # Get patterns for this well
            all_patterns = get_patterns_for_well(well, input_dir, self.microscope_handler)
            if not all_patterns:
                raise ValueError(f"No patterns found for well {well}")

            # Get all matching files
            all_files = []
            for pattern in all_patterns:
                matching_files = self.microscope_handler.parser.path_list_from_pattern(input_dir, pattern)
                all_files.extend(matching_files)

            # Register overlay operations for all files
            overlay_paths = {}
            for file_path in all_files:
                # Construct a key based on the file path
                rel_path = Path(file_path).relative_to(input_dir) if Path(file_path).is_relative_to(input_dir) else Path(file_path).name
                key = f"overlay_{well}_{rel_path}"

                # Try to get the data from storage adapter
                try:
                    if self.storage_adapter.exists(key):
                        # Register for overlay
                        disk_path = self.storage_adapter.register_for_overlay(key, operation_type="read", cleanup=True)
                        if disk_path:
                            overlay_paths[file_path] = disk_path
                except Exception as e:
                    logger.warning(f"Error registering overlay for {file_path}: {e}")

            # Execute all overlay operations
            if overlay_paths:
                self.storage_adapter.execute_all_overlay_operations(self.file_manager)
                logger.info(f"Executed {len(overlay_paths)} overlay operations for generate_positions")

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

        # Clean up overlay operations if needed
        if needs_overlay:
            self.storage_adapter.cleanup_overlay_operations(self.file_manager)

        return positions_file, reference_pattern

    def _create_output_filename(self, pattern):
        """
        Create an output filename for a stitched image based on a pattern.

        Requires the microscope handler to be initialized.

        Args:
            pattern: Filename pattern

        Returns:
            str: Output filename

        Raises:
            RuntimeError: If microscope handler is not initialized
            ValueError: If pattern is invalid
        """
        # Ensure microscope handler is available
        microscope_handler = self._ensure_microscope_handler()

        parsable = pattern.replace('{iii}', '001')
        metadata = microscope_handler.parser.parse_filename(parsable)

        if not metadata:
            raise ValueError(f"Could not parse pattern: {pattern}")

        return microscope_handler.parser.construct_filename(
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

        This method requires the orchestrator to be initialized first by calling initialize().

        Raises:
            RuntimeError: If the orchestrator has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator must be initialized before calling stitch_images(). Call initialize() first.")
        logger.info("Stitching images for well %s", well)

        # Check if we need to use overlay
        needs_overlay = (
            hasattr(self, 'storage_adapter') and
            self.storage_adapter is not None and
            self.storage_mode != "legacy" and
            self.overlay_mode != OverlayMode.DISABLED
        )

        # If using overlay, ensure all required images are written to disk
        if needs_overlay:
            logger.info("Using overlay for stitch_images")
            # Get patterns for this well
            all_patterns = get_patterns_for_well(well, input_dir, self.microscope_handler)
            if not all_patterns:
                raise ValueError(f"No patterns found for well {well} in {input_dir}")

            # Get all matching files
            all_files = []
            for pattern in all_patterns:
                matching_files = self.microscope_handler.parser.path_list_from_pattern(input_dir, pattern)
                all_files.extend(matching_files)

            # Register overlay operations for all files
            overlay_paths = {}
            for file_path in all_files:
                # Construct a key based on the file path
                rel_path = Path(file_path).relative_to(input_dir) if Path(file_path).is_relative_to(input_dir) else Path(file_path).name
                key = f"overlay_{well}_{rel_path}"

                # Try to get the data from storage adapter
                try:
                    if self.storage_adapter.exists(key):
                        # Register for overlay
                        disk_path = self.storage_adapter.register_for_overlay(key, operation_type="read", cleanup=True)
                        if disk_path:
                            overlay_paths[file_path] = disk_path
                except Exception as e:
                    logger.warning(f"Error registering overlay for {file_path}: {e}")

            # Execute all overlay operations
            if overlay_paths:
                self.storage_adapter.execute_all_overlay_operations(self.file_manager)
                logger.info(f"Executed {len(overlay_paths)} overlay operations for stitch_images")

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

        # Clean up overlay operations if needed
        if needs_overlay:
            self.storage_adapter.cleanup_overlay_operations(self.file_manager)

    def _ensure_file_manager(self):
        """
        Ensure file manager is initialized.

        Returns:
            FileManager: The initialized file manager

        Raises:
            RuntimeError: If file manager is not initialized
        """
        if self.file_manager is None:
            raise RuntimeError(
                "File manager is not initialized. Call initialize() first, or "
                "ensure plate_path and workspace_path are properly set."
            )
        return self.file_manager

    def _ensure_storage_adapter(self):
        """
        Ensure storage adapter is initialized if not in legacy mode.

        Returns:
            StorageAdapter: The initialized storage adapter, or None if in legacy mode

        Raises:
            RuntimeError: If storage adapter is not initialized when required
        """
        if self.storage_mode != "legacy" and self.storage_adapter is None:
            raise RuntimeError(
                "Storage adapter is not initialized. Call initialize() first, or "
                "ensure storage_mode is properly set."
            )
        return self.storage_adapter

    def _ensure_microscope_handler(self):
        """
        Ensure microscope handler is initialized.

        Returns:
            MicroscopeHandler: The initialized microscope handler

        Raises:
            RuntimeError: If microscope handler is not initialized
        """
        if self.microscope_handler is None:
            raise RuntimeError(
                "Microscope handler is not initialized. Call initialize() first, or "
                "ensure plate_path and workspace_path are properly set."
            )
        return self.microscope_handler

    def _ensure_stitcher(self):
        """
        Ensure stitcher is initialized.

        Returns:
            Stitcher: The initialized stitcher

        Raises:
            RuntimeError: If stitcher is not initialized
        """
        if self.stitcher is None:
            raise RuntimeError(
                "Stitcher is not initialized. Call initialize() first, or "
                "ensure microscope_handler is properly initialized."
            )
        return self.stitcher

    def initialize_storage_adapter(self):
        """
        Initialize the storage adapter based on the storage_mode.

        This method creates a StorageAdapter instance based on the storage_mode:
        - "legacy": No adapter is created (None)
        - "memory": MemoryStorageAdapter is created
        - "zarr": ZarrStorageAdapter is created

        The storage_root is used as the root directory for the adapter.
        If no storage_root is provided, a suffixed version of input_dir is used
        to ensure output directories are never the same as input directories.

        Note: This method is deprecated. Use initialize() instead.
        """
        logger.warning("initialize_storage_adapter() is deprecated. Use initialize() instead.")
        self._initialize_storage_adapter()

    def initialize_file_manager(self):
        """
        Initialize the FileManager with the appropriate backend.

        This method creates a FileManager instance with a backend based on the storage_mode:
        - "memory": Uses a memory backend for testing
        - All other modes: Uses a filesystem backend

        The input_dir is used as the root directory for the FileManager.

        Note: This method is deprecated. Use initialize() instead.
        """
        logger.warning("initialize_file_manager() is deprecated. Use initialize() instead.")
        self._initialize_file_manager()


