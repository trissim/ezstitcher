import logging
import os
import copy
import random
import time
import threading
import concurrent.futures
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple, Any


from ezstitcher.core.microscope_interfaces import create_microscope_handler, MicroscopeHandler
from ezstitcher.core.stitcher import Stitcher
# Removed: from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.io.filemanager import FileManager # Added
from ezstitcher.core.image_processor import ImageProcessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.opera_phenix_xml_parser import OperaPhenixXmlParser

# Import the pipeline architecture
from ezstitcher.core.pipeline import Step, Pipeline, StepExecutionPlan

from ezstitcher.core.pattern_resolver import get_patterns_for_well
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
                 file_manager: Optional[FileManager] = None,
                 image_preprocessor: Optional[ImageProcessor] = None,
                 focus_analyzer: Optional[FocusAnalyzer] = None):
        """
        Initialize the pipeline orchestrator with dependencies.

        Args:
            plate_path: Path to the source plate data.
            workspace_path: Path to the working directory. If None, derived from plate_path.
            config: Pipeline configuration object.
            root_dir: Root directory for file operations. If None, uses current directory.
            backend: String identifier for the storage backend type (e.g., "filesystem", "memory").
                     If None, defaults to "filesystem".
            file_manager: FileManager instance. If None, a new one is created using root_dir and backend.
            image_preprocessor: Instance for image preprocessing steps.
            focus_analyzer: Instance for focus analysis.
        """
        self.config = config or PipelineConfig()
        self.plate_path = Path(plate_path) if plate_path else None

        # --- Use provided FileManager or create a new one ---
        if file_manager is not None:
            self.file_manager = file_manager
            logger.info("Using provided FileManager with backend: %s", type(self.file_manager.backend).__name__)
        else:
            # Create FileManager using root_dir and backend
            self.file_manager = FileManager(root_dir=root_dir, backend=backend)
            logger.info("Created new FileManager with backend: %s", type(self.file_manager.backend).__name__)

        # --- Workspace Path Determination ---
        # Logic to determine the workspace path, potentially using file_manager if needed
        # (e.g., checking existence or creating it via ensure_directory).
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
                workspace_fm = FileManager(root_dir=self.file_manager.root_dir, backend="disk")
                num_links = workspace_fm.mirror_directory_with_symlinks(
                    source_dir=self.plate_path,
                    target_dir=self.workspace_path,
                    recursive=True,
                    overwrite=True
                )
                logger.info("Created %d symlinks in workspace", num_links)

                # 2. Set input_dir to workspace_path for all subsequent operations
                self.input_dir = self.workspace_path

                # 3. Initialize microscope handler using workspace
                logger.info("Initializing microscope handler using workspace...")
                microscope_type = self._detect_microscope_handler(self.workspace_path)
                self.microscope_handler = create_microscope_handler(
                    microscope_type=microscope_type,
                    plate_folder=self.workspace_path,
                    file_manager=self.file_manager
                )

                logger.info("Using microscope handler: %s", type(self.microscope_handler).__name__)


                # 3. Post-workspace setup (e.g., renaming, flattenign)
                self.microscope_handler.post_workspace(self.workspace_path)

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

    # <<< New Workspace Creation Methods Start >>>

    def create_workspace(self) -> None:
        """
        Initializes the processing workspace. Creates symlinks from the input_dir
        to the workspace_path and performs necessary file remapping based on microscope type.

        This method always uses a disk-backed FileManager for workspace operations,
        regardless of the orchestrator's main FileManager backend type.
        """
        if not self.workspace_path or not self.input_dir or not self.microscope_handler:
            logger.error("Cannot create workspace: workspace_path, input_dir, or microscope_handler not set.")
            return

        logger.info("Creating workspace at: %s from input: %s", self.workspace_path, self.input_dir)
        start_time = time.time()

        # Create a dedicated disk-backed FileManager for workspace operations
        # Use the same root_dir as the main FileManager for consistency
        workspace_fm = FileManager(root_dir=self.file_manager.root_dir, backend="disk")
        logger.info("Created disk-backed FileManager for workspace operations")

        # 1. List initial files from the prepared input directory
        try:
            # List files from the potentially processed input_dir using the main FileManager
            # (this operation doesn't require disk-specific operations)
            raw_files = self.file_manager.list_files(self.input_dir, recursive=True)
            logger.info("Found %d files in input directory for workspace creation.", len(raw_files))
            if not raw_files:
                logger.warning("No files found in input directory. Workspace will be empty.")
                # Decide if this is an error or just an empty workspace
                return
        except Exception as e:
            logger.error("Failed to list files from input directory %s: %s", self.input_dir, e)
            raise RuntimeError(f"Failed to list input files for workspace creation: {e}") from e

        # 2. Create initial symlinks using the disk-backed FileManager
        try:
            # Pass the workspace_fm to the helper method
            initial_symlinks = self._create_initial_symlinks(raw_files, workspace_fm)
            logger.info("Created %d initial symlinks.", len(initial_symlinks))
        except Exception as e:
            logger.error("Failed during initial symlink creation: %s", e)
            raise RuntimeError(f"Failed to create initial symlinks: {e}") from e

        # 3. Handle Field Remapping (if required)
        final_symlink_paths = list(initial_symlinks.values()) # Default to initial if no remapping
        if self.microscope_handler.requires_field_remapping:
            logger.info("Field remapping required by handler, processing symlinks...")
            try:
                # Pass the workspace_fm to the helper method
                final_symlink_paths = self._remap_symlinks_using_metadata(initial_symlinks, workspace_fm)
                logger.info("Field remapping completed.")
            except Exception as e:
                logger.error("Failed during symlink remapping: %s. Workspace may have inconsistent names.", e)
                # Decide if this is fatal. For now, continue with potentially unmapped names.
                # Consider adding a config flag to control strictness.
        else:
            logger.info("Field remapping not required by handler.")

        # 4. Finalize (e.g., update context if needed, though context is created later)
        self._finalize_symlinks(final_symlink_paths)

        logger.info("Workspace created successfully in %.2f seconds.", time.time() - start_time)


    def _create_initial_symlinks(self, source_files: List[Union[str, Path]], file_manager: FileManager) -> Dict[str, Path]:
        """
        Creates the initial set of symlinks in the workspace_path directory, preserving the original directory structure.

        Args:
            source_files: List of source files to create symlinks for
            file_manager: FileManager instance to use for file operations (should be disk-backed)

        Returns:
            Dictionary mapping original filenames to symlink paths
        """
        if not self.workspace_path or not self.input_dir:
            raise ValueError("_create_initial_symlinks requires workspace_path and input_dir to be set.")

        symlink_map = {}
        logger.info("Creating symlinks in workspace: %s", self.workspace_path)

        # Ensure the workspace directory exists
        file_manager.ensure_directory(self.workspace_path)

        # Get the input directory as a Path object
        input_dir_path = Path(self.input_dir)

        # Log the input directory structure for debugging
        logger.debug("Input directory structure: %s", input_dir_path)

        # Check if the input directory exists
        if not input_dir_path.exists():
            logger.error("Input directory does not exist: %s", input_dir_path)
            return {}

        for file_path_str in source_files:
            try:
                source_path = Path(file_path_str)

                # Calculate relative path from the input_dir base
                try:
                    # Make sure we're using absolute paths for comparison
                    abs_source_path = source_path.absolute()
                    abs_input_dir = input_dir_path.absolute()

                    # Calculate the relative path
                    relative_path = abs_source_path.relative_to(abs_input_dir)
                    logger.debug("Relative path: %s (from %s to %s)",
                                relative_path, abs_source_path, abs_input_dir)

                    # Create the symlink path in the workspace, preserving the directory structure
                    symlink_path = self.workspace_path / relative_path

                    # Ensure parent directory exists within the workspace
                    file_manager.ensure_directory(symlink_path.parent)

                    # Create the symlink using the provided FileManager
                    file_manager.create_symlink(abs_source_path, symlink_path)

                    # Store original filename -> symlink path mapping
                    symlink_map[source_path.name] = symlink_path
                    logger.debug("Created symlink: %s -> %s", abs_source_path, symlink_path)
                except ValueError as e:
                    # This happens if source_path is not relative to input_dir
                    logger.warning("Source path %s is not relative to input_dir %s: %s. Skipping.",
                                  source_path, self.input_dir, e)
                    continue
            except Exception as e:
                logger.warning("Failed to create symlink for %s: %s. Skipping.", file_path_str, e)
                # Continue creating other symlinks

        logger.info("Created %d symlinks in workspace", len(symlink_map))
        return symlink_map


    def _remap_symlinks_using_metadata(self, initial_symlinks: Dict[str, Path], file_manager: FileManager) -> List[Path]:
        """
        Renames symlinks based on microscope metadata (specifically for Opera Phenix).

        Args:
            initial_symlinks: Dictionary mapping original filenames to symlink paths
            file_manager: FileManager instance to use for file operations (should be disk-backed)

        Returns:
            List of final symlink paths after remapping
        """
        if not self.input_dir:
            raise ValueError("_remap_symlinks_using_metadata requires input_dir to be set.")

        logger.info("Remapping symlinks using metadata...")
        final_symlink_paths = []

        # 1. Find the metadata file (e.g., Index.xml) in the original input directory
        metadata_file = self._find_metadata_file(self.input_dir)
        if not metadata_file:
            logger.warning("Could not find metadata file (e.g., Index.xml) in %s for remapping. Using original symlink names.", self.input_dir)
            return list(initial_symlinks.values()) # Return original paths

        # 2. Create the specific metadata parser (OperaPhenixXmlParser)
        try:
            # Assuming OperaPhenixXMLParser is the relevant class
            # Use the imported OperaPhenixXmlParser class
            logger.debug("Creating OperaPhenixXmlParser for: %s", metadata_file)
            xml_parser = OperaPhenixXmlParser(metadata_file)
            # Check if parser has the necessary remapping method (belt-and-suspenders)
            if not hasattr(self.microscope_handler.parser, 'remap_field_in_filename'):
                logger.error("Handler's parser is missing 'remap_field_in_filename' method needed for remapping.")
                return list(initial_symlinks.values()) # Cannot remap

        except ImportError:
            logger.error("Failed to import OperaPhenixXmlParser. Cannot perform remapping.")
            return list(initial_symlinks.values())
        except Exception as e:
            logger.error("Failed to create or use metadata parser for %s: %s", metadata_file, e)
            return list(initial_symlinks.values()) # Cannot remap if parser fails

        # 3. Iterate through initial symlinks and rename based on metadata
        rename_count = 0
        remap_start_time = time.time()
        for original_filename, symlink_path in initial_symlinks.items():
            try:
                # Calculate the new filename using the parser's remapping logic
                new_filename = self.microscope_handler.parser.remap_field_in_filename(original_filename, xml_parser)

                if new_filename and new_filename != original_filename:
                    new_symlink_path = symlink_path.with_name(new_filename)
                    logger.debug("Renaming symlink: %s -> %s", symlink_path, new_symlink_path)
                    # Use the provided FileManager to rename
                    file_manager.rename(symlink_path, new_symlink_path)
                    final_symlink_paths.append(new_symlink_path)
                    rename_count += 1
                else:
                    # Keep original if no remapping needed or failed for this file
                    final_symlink_paths.append(symlink_path)

            except Exception as e:
                logger.warning("Error remapping/renaming symlink for %s: %s. Keeping original.", original_filename, e)
                final_symlink_paths.append(symlink_path) # Keep original on error

        logger.info("Symlink remapping completed in %.2f seconds. Renamed %d symlinks.", time.time() - remap_start_time, rename_count)
        return final_symlink_paths


    def _find_metadata_file(self, search_dir: Path) -> Optional[Path]:
        """Finds the relevant metadata file (e.g., Index.xml) using FileManager."""
        # Logic to search for known metadata file patterns, specific to Opera Phenix here
        try:
            # Use file_manager to search recursively in the source directory
            metadata_file = self.file_manager.find_file_recursive(search_dir, "Index.xml")
            if metadata_file:
                logger.info("Found metadata file: %s", metadata_file)
                return Path(metadata_file) # Ensure it's a Path object

            logger.warning("Metadata file (Index.xml) not found in %s", search_dir)
            return None
        except Exception as e:
            logger.error("Error searching for metadata file in %s: %s", search_dir, e)
            return None


    def _calculate_remapped_name(self, original_filename: str, xml_parser: OperaPhenixXmlParser) -> Optional[str]:
        """Calculates the new filename based on metadata using the handler's parser."""
        # This helper is now effectively replaced by calling the parser method directly
        # in _remap_symlinks_using_metadata. Keeping as placeholder in case needed later.
        # try:
        #     if hasattr(self.handler.parser, 'remap_field_in_filename'):
        #         return self.handler.parser.remap_field_in_filename(original_filename, xml_parser)
        #     else:
        #         logger.warning("Parser lacks 'remap_field_in_filename' method.")
        #         return original_filename
        # except Exception as e:
        #     logger.warning("Error calculating remapped name for %s: %s", original_filename, e)
        #     return original_filename
        # No longer directly used, logic moved into _remap_symlinks_using_metadata
        return None


    def _finalize_symlinks(self, final_symlink_paths: List[Path]) -> None:
        """Perform any final actions after symlinks are created/renamed."""
        # This might involve updating internal state or context if needed later.
        # For now, just log completion.
        logger.info("Finalized %d symlinks in workspace.", len(final_symlink_paths))
        # Example: self.context.update_file_list(final_symlink_paths) # If context needs updating


    # <<< New Workspace Creation Methods End >>>


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
            # Workspace creation is now handled in __init__ after dependencies are ready.
            # Grid size and pixel size should be determined from the original plate_path or input_dir,
            # not the workspace_path which might contain remapped/symlinked data.
            # Let's assume these are needed *before* workspace creation or use input_dir.
            # If they rely on workspace structure, this needs rethinking.
            # For now, assume they use plate_path or input_dir.
            source_path_for_metadata = self.input_dir or self.plate_path
            if not source_path_for_metadata:
                 raise ValueError("Cannot determine grid size or pixel size without plate_path or input_dir.")

            self.config.grid_size = self.microscope_handler.get_grid_dimensions(source_path_for_metadata)
            logger.info("Grid size: %s", self.config.grid_size)
            try:
                # Use source_path_for_metadata
                self.config.pixel_size = self.microscope_handler.get_pixel_size(source_path_for_metadata) or self.config.stitcher.pixel_size
            except Exception as e:
                # Log error specific to pixel size detection
                logger.error(f"Failed to auto-detect pixel size from {source_path_for_metadata}: {e}")
                self.config.pixel_size = self.config.stitcher.pixel_size
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

        try:
            # Use injected file_manager instance
            image_paths = self.file_manager.list_image_files(input_dir, recursive=True)
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
                input_dir = self.workspace_path
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

    def prepare_images(self, plate_path: Path) -> Optional[Path]:
        """
        Prepares images using FileManager for operations like finding directories,
        renaming, and organizing Z-stacks.
        Orchestrator logic: Defines the sequence of preparation steps.
        FileManager role: Executes the individual file/directory operations.
        """
        logger.info("Preparing images for plate: %s", plate_path)
        start_time = time.time()

        if not self.microscope_handler:
            logger.error("Microscope handler not initialized, cannot prepare images.")
            return None

        try:
            # Find the image directory using FileManager
            image_dir = self.file_manager.find_image_directory(plate_path)
            logger.info("Found image directory: %s", image_dir)

            # Rename files using FileManager
            logger.info("Renaming files with consistent padding...")
            rename_start = time.time()
            # Ensure parser is correctly passed (might be self.microscope_handler itself)
            self.file_manager.rename_files_with_consistent_padding(
                directory=image_dir,
                parser=self.microscope_handler, # Assuming handler acts as parser
                width=DEFAULT_PADDING,
                force_suffixes=True
            )
            logger.info("Renamed files in %.2fs", time.time() - rename_start)

            # Detect and organize Z-stacks using FileManager
            zstack_start = time.time()
            has_zstack, z_folders = self.file_manager.detect_zstack_folders(image_dir)
            if has_zstack:
                logger.info("Found %d Z-stack folders. Organizing...", len(z_folders))
                # Ensure filename_parser is correctly passed
                self.file_manager.organize_zstack_folders(
                    plate_folder=image_dir,
                    filename_parser=self.microscope_handler # Assuming handler acts as parser
                )
                logger.info("Organized Z-stack folders in %.2fs", time.time() - zstack_start)
            else:
                logger.info("No Z-stack folders detected or organization not needed.")

            logger.info("Prepared images in %.2fs", time.time() - start_time)
            # Return the potentially modified image_dir
            # Re-finding might be necessary if organize_zstack changes structure significantly,
            # but for now, assume image_dir remains the primary input root.
            # return self.file_manager.find_image_directory(plate_path)
            return image_dir

        except Exception as e:
            logger.error("Error preparing images for %s: %s", plate_path, e)
            return None # Return None on error

    def _detect_microscope_handler(self, plate_path: Path) -> str:
        """
        Auto-detect the microscope type by testing parsers against sample files.

        Args:
            plate_path: Path to the plate directory

        Returns:
            str: The detected microscope type identifier

        Raises:
            ValueError: If no suitable parser can be found
        """
        logger.info("Detecting microscope type...")

        # Check if a specific parser is forced in the configuration
        if self.config.force_parser:
            forced_parser = self.config.force_parser
            logger.info("Using forced parser from configuration: %s", forced_parser)
            return forced_parser.lower()  # Normalize to lowercase for consistency

        # First try to detect based on characteristic files
        try:
            if self.file_manager.find_file_recursive(plate_path, "Index.xml"):
                logger.info("Auto-detected Opera Phenix microscope type based on Index.xml file.")
                return 'operaphenix'

            if self.file_manager.list_image_files(plate_path, extensions={'.htd'}, recursive=False):
                logger.info("Auto-detected ImageXpress microscope type based on HTD file.")
                return 'imagexpress'
        except Exception as e:
            logger.warning("Error during file-based detection: %s. Falling back to filename-based detection.", e)

        # If no characteristic files found, try filename-based detection
        try:
            # Get sample files using FileManager
            sample_files = self.file_manager.list_image_files(plate_path, recursive=True)

            # Limit to a reasonable sample size
            max_sample_size = 50
            if len(sample_files) > max_sample_size:
                sample_files = random.sample(sample_files, max_sample_size)

            if not sample_files:
                logger.error("No image files found in %s", plate_path)
                raise ValueError(f"No image files found in {plate_path}")

            # Discover all available handlers
            handlers = MicroscopeHandler._discover_handlers()

            # Test each parser against the sample files
            matches = {}
            for name, (parser_class, _) in handlers.items():
                matches[name] = 0
                for f in sample_files:
                    if parser_class.can_parse(f.name):
                        matches[name] += 1

            # Log the match counts for each parser
            for name, count in matches.items():
                match_percent = count / len(sample_files) * 100
                logger.info("Parser '%s' matched %d/%d files (%.1f%%)",
                           name, count, len(sample_files), match_percent)

            # Find the best match
            if not matches:
                logger.error("No parsers available")
                raise ValueError("No parsers available")

            best_match = max(matches.items(), key=lambda x: x[1])

            # Only accept if we have at least one match
            if best_match[1] > 0:
                logger.info("Selected parser '%s' with %d/%d matches",
                           best_match[0], best_match[1], len(sample_files))
                return best_match[0]

            logger.error("No parser could match any of the %d sample files", len(sample_files))
            raise ValueError(f"No parser could match any of the {len(sample_files)} sample files")

        except Exception as e:
            logger.error("Error during microscope type detection: %s", e)
            raise ValueError(f"Failed to detect microscope type: {e}") from e

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

        # Find the actual image directory using file_manager
        actual_input_dir = self.file_manager.find_image_directory(input_dir)
        logger.info("Using actual image directory: %s", actual_input_dir)

        # Ensure output directory exists using file_manager
        self.file_manager.ensure_directory(output_dir)
        logger.info("Ensured output directory exists: %s", output_dir)

        # Get patterns for this well
        all_patterns = get_patterns_for_well(well, actual_input_dir, self.microscope_handler)
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
