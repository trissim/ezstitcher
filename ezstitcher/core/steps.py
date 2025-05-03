"""
Step classes for the pipeline architecture.

This module contains the Step class and all specialized step implementations for
different types of processing operations, including:

1. Base Step class for general-purpose processing
2. Step factories (ZFlatStep, FocusStep, CompositeStep) for common operations
3. Specialized steps (PositionGenerationStep, ImageStitchingStep) for specific tasks

For conceptual explanation, see the documentation at:
https://ezstitcher.readthedocs.io/en/latest/concepts/step.html
"""

from typing import Dict, List, Union, Callable, Any, TypeVar, Optional, Sequence, Tuple, TYPE_CHECKING
import logging
from pathlib import Path
import numpy as np

# Import core components
from ezstitcher.io.filemanager import FileManager # Added
from ezstitcher.core.utils import prepare_patterns_and_functions
from ezstitcher.core.abstract_step import AbstractStep
from ezstitcher.core.image_processor import ImageProcessor as IP
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.io.overlay import OverlayMode
# Removed adapt_func_to_stack import

# Import StepResult for type hints
if TYPE_CHECKING:
    from ezstitcher.core.pipeline import StepResult, ProcessingContext


# Type definitions
# Note: All functions in ProcessingFunc are now expected to accept List[np.ndarray]
# and return List[np.ndarray]. Use utils.stack() to wrap single-image functions.
FunctionType = Callable[[List[np.ndarray], ...], List[np.ndarray]]
# A function can be a callable or a tuple of (callable, kwargs)
FunctionWithArgs = Union[FunctionType, Tuple[FunctionType, Dict[str, Any]]]
ProcessingFunc = Union[FunctionWithArgs, Dict[str, FunctionWithArgs], List[FunctionWithArgs]]
VariableComponents = List[str]
GroupBy = Optional[str]
WellFilter = Optional[List[str]]
T = TypeVar('T')  # For generic return types

# Configure logging
logger = logging.getLogger(__name__)


class Step(AbstractStep):
    """
    A processing step in a pipeline.

    A Step encapsulates a processing operation that can be applied to images.
    It mirrors the functionality of process_patterns_with_variable_components
    while providing a more object-oriented interface.

    Step Architecture Notes:
    - Steps must be stateless and should NOT modify the context directly
    - Steps must return a StepResult object containing:
      - Normal processing results
      - Requested context updates
      - Requested storage operations
    - Pipeline.run() is responsible for applying these changes

    Attributes:
        func: The processing function(s) to apply
        variable_components: Components that vary across files (e.g., 'z_index', 'channel')
        group_by: How to group files for processing (e.g., 'channel', 'site')
        name: Human-readable name for the step
        requires_fs_input: Whether this step requires input files on disk
        requires_fs_output: Whether this step writes output files directly to disk
        force_disk_output: Whether to force disk materialization of outputs
        requires_legacy_fs: Whether this step requires legacy file system access
    """

    # Class-level flags for file system requirements - IMMUTABLE
    requires_fs_input = False
    requires_fs_output = False
    force_disk_output = False
    requires_legacy_fs = False

    def __init__(
        self,
        func: ProcessingFunc,
        variable_components: VariableComponents = ['site'],
        #group_by: GroupBy = None,
        group_by: GroupBy = 'channel',
        name: str = None,
        requires_legacy_fs: bool = False,
        **kwargs
    ):
        """
        Initialize a processing step.

        Args:
            func: The processing function(s) to apply. Can be:
                - A single callable function
                - A tuple of (function, kwargs)
                - A list of functions or (function, kwargs) tuples
                - A dictionary mapping component values to functions or tuples
            variable_components: Components that vary across files
            group_by: How to group files for processing
            name: Human-readable name for the step
            requires_legacy_fs: Whether this step requires legacy file system access
            **kwargs: Additional keyword arguments, including input_dir and output_dir
                      which will be extracted by the Pipeline
        """
        # Store input_dir and output_dir in a temporary attribute if present
        self._ephemeral_init_kwargs = {}
        if 'input_dir' in kwargs:
            self._ephemeral_init_kwargs['input_dir'] = kwargs.pop('input_dir')
        if 'output_dir' in kwargs:
            self._ephemeral_init_kwargs['output_dir'] = kwargs.pop('output_dir')

        # Initialize the step
        self.func = func
        self.variable_components = variable_components
        self.group_by = group_by
        self._name = name or self._generate_name()

        # Handle requires_legacy_fs flag
        if requires_legacy_fs:
            # Override the class-level attribute with an instance-level attribute
            self.requires_legacy_fs = True
            logger.debug(f"Step '{self._name}' requires legacy filesystem access")

    def _generate_name(self) -> str:
        """
        Generate a descriptive name based on the function.

        Returns:
            A human-readable name for the step
        """
        # Helper function to get name from function or function tuple
        def get_func_name(f):
            if isinstance(f, tuple) and len(f) == 2 and callable(f[0]):
                return getattr(f[0], '__name__', str(f[0]))
            if callable(f):
                return getattr(f, '__name__', str(f))
            return str(f)

        # Dictionary of functions
        if isinstance(self.func, dict):
            funcs = ", ".join(f"{k}:{get_func_name(f)}" for k, f in self.func.items())
            return f"ChannelMappedStep({funcs})"

        # List of functions
        if isinstance(self.func, list):
            funcs = ", ".join(get_func_name(f) for f in self.func)
            return f"MultiStep({funcs})"

        # Single function or function tuple
        return f"Step({get_func_name(self.func)})"

    def needs_materialization(self) -> bool:
        """
        Return whether this step requires materialization based on its flags.

        This is a declarative method that simply returns the step's materialization
        requirements based on its flags. The actual decision of whether to perform
        materialization is made by the orchestration layer.

        Returns:
            True if the step declares it needs materialization, False otherwise
        """
        return (self.requires_fs_input or
                self.requires_fs_output or
                self.force_disk_output or
                self.requires_legacy_fs)

    def process(self, context: 'ProcessingContext') -> 'StepResult':
        """
        Process the step with the given context.

        Args:
            context: The processing context containing pre-computed paths and other state (read-only)

        Returns:
            StepResult object containing processing results, context updates, and storage operations.
            Steps should NOT modify the context directly.
        """
        logger.info("Processing step: %s", self.name)

        # Create a result object
        result = self.create_result()

        # Get directories and components from context (read-only)
        input_dir = context.get_step_input_dir(self)
        output_dir = context.get_step_output_dir(self)
        in_place = input_dir == output_dir
        if in_place:
            logger.warning("Input and output directories are the same for step %s. Working in-place.", self.name)

        well_filter = context.well_filter
        try:
            orchestrator = context.orchestrator
            if not orchestrator:
                raise ValueError("Orchestrator missing from context.")
            # Access FileManager via the orchestrator in the context
            file_manager = orchestrator.file_manager
            microscope_handler = orchestrator.microscope_handler
            if not microscope_handler:
                 raise ValueError("Microscope handler missing from orchestrator.")
            if not well_filter:
                logger.warning("No wells specified in context's well_filter. Skipping step.")
                return result

        except (AttributeError, ValueError) as e:
            logger.error(f"Context validation failed for step {self.name}: {e}")
            raise RuntimeError(f"Invalid context for step {self.name}") from e

        # --- Core Processing Logic ---
        logger.debug(f"Step '{self.name}': Input='{input_dir}', Output='{output_dir}'")

        # Get patterns with variable components
        patterns_by_well = microscope_handler.auto_detect_patterns(
            input_dir,
            well_filter=well_filter,
            variable_components=self.variable_components
        )

        # Process each well
        step_results = {}
        for well, patterns in patterns_by_well.items():
            if well_filter and well not in well_filter:
                continue

            logger.info("Processing well: %s", well)
            well_results = {}

            # Prepare patterns, functions, and args
            grouped_patterns, component_to_funcs, component_to_args = prepare_patterns_and_functions(
                patterns, self.func, component=self.group_by
            )

            # Process each component
            for component_value, component_patterns in grouped_patterns.items():
                component_func = component_to_funcs[component_value]
                component_args = component_to_args[component_value]
                output_files = []

                # Process each pattern
                for pattern in component_patterns:
                    # Find matching files
                    matching_files = microscope_handler.parser.path_list_from_pattern(input_dir, pattern)

                    # Load images using FileManager
                    image_paths_to_load = [Path(input_dir) / filename for filename in matching_files]
                    try:
                        # Use file_manager instance from context
                        images = [file_manager.load_image(p) for p in image_paths_to_load]
                        # Filter out None results (failed loads)
                        loaded_images = [img for img in images if img is not None]
                    except Exception as e:
                        logger.error(f"Error loading images for pattern '{pattern}': {e}", exc_info=True)
                        loaded_images = [] # Ensure it's empty on error

                    if not loaded_images:
                        logger.warning(f"No images successfully loaded for pattern '{pattern}'. Skipping.")
                        continue  # Skip if no valid images found
                    if len(loaded_images) != len(matching_files):
                         logger.warning(f"Loaded {len(loaded_images)} out of {len(matching_files)} files for pattern '{pattern}'.")

                    if in_place:
                        for p in image_paths_to_load:
                            if file_manager.exists(p):
                                file_manager.remove(p)

                    # Process the loaded images with component-specific args
                    try:
                        # Pass well and component to _apply_processing for better debugging
                        processed_images = self._apply_processing(
                            loaded_images,
                            context=context,
                            func=component_func,
                            well=well,
                            component=component_value
                        )
                        if not processed_images: # Check if processing failed or returned empty
                             logger.warning(f"Processing returned no images for pattern '{pattern}'. Skipping save.")
                             continue
                    except Exception as e:
                        logger.error(f"Error applying processing function for pattern '{pattern}': {e}", exc_info=True)
                        continue # Skip saving if processing failed

                    # Prepare save operations
                    original_filenames_loaded = [mf for img, mf in zip(images, matching_files) if img is not None]
                    save_operations = self._prepare_save_operations(
                        context=context,
                        input_dir=Path(input_dir),
                        output_dir=Path(output_dir),
                        images=processed_images,
                        filenames=original_filenames_loaded
                    )

                    # Create save result
                    save_result = self._create_save_result(
                        operations=save_operations,
                        file_manager=file_manager
                    )

                    # Merge the save result into our main result
                    result.merge(save_result)

                    # Get saved files for backward compatibility
                    if "saved_files" in save_result.results:
                        output_files.extend(save_result.results["saved_files"])

                # Store results for this component
                if output_files:
                    well_results[component_value] = output_files

            # Store results for this well
            step_results[well] = well_results

        # Add results to the StepResult
        result.add_result("results", step_results)

        # Handle storage of numpy arrays
        if hasattr(context, 'orchestrator') and context.orchestrator:
            storage_mode = getattr(context.orchestrator, 'storage_mode', "legacy")
            if storage_mode != "legacy":
                # Import here to avoid circular imports
                from ezstitcher.io.storage_adapter import generate_storage_key

                # Log the number of results
                logger.debug("Step '%s' has %d well results to potentially store",
                            self.name, len(step_results))

                # Iterate through results and store numpy arrays
                for well, well_data in step_results.items():
                    for component, component_data in well_data.items():
                        # Log the type of component_data to help with debugging
                        logger.debug(
                            "Result data for well '%s', component '%s' is type: %s",
                            well, component, type(component_data).__name__
                        )

                        # Handle different data types in results
                        if isinstance(component_data, np.ndarray):
                            # Direct numpy array case
                            data_to_store = component_data
                            key = generate_storage_key(self.name, well, component)
                            # Add storage operation
                            result.store(key, data_to_store)
                        elif isinstance(component_data, list) and component_data:
                            # List of file paths case - try to load the first image
                            try:
                                # Get the file manager from context
                                file_manager = context.orchestrator.file_manager
                                # Try to load the first file in the list
                                first_file = component_data[0]
                                logger.debug("Attempting to load image from path: %s", first_file)
                                data_to_store = file_manager.load_image(first_file)
                                if data_to_store is None:
                                    logger.warning(
                                        "Could not load image from path: %s", first_file
                                    )
                                    continue
                                key = generate_storage_key(self.name, well, component)
                                # Add storage operation
                                result.store(key, data_to_store)
                            except Exception as e:
                                logger.error(
                                    "Error loading image for storage: %s", e, exc_info=True
                                )
                                continue
                        else:
                            # Unsupported data type
                            logger.debug(
                                "Skipping unsupported data type for well '%s', component '%s'",
                                well, component
                            )
                            continue

        return result

    def _extract_function_and_args(
        self,
        func_item: FunctionWithArgs
    ) -> Tuple[Callable, Dict[str, Any]]:
        """Extract function and arguments from a function item.

        A function item can be either a callable or a tuple of (callable, kwargs).

        Args:
            func_item: Function item to extract from

        Returns:
            Tuple of (function, kwargs)
        """
        if isinstance(func_item, tuple) and len(func_item) == 2 and callable(func_item[0]):
            # It's a (function, kwargs) tuple
            return func_item[0], func_item[1]
        if callable(func_item):
            # It's just a function, use default args
            return func_item, {}

        # Invalid function item
        logger.warning(
            "Invalid function item: %s. Expected callable or (callable, kwargs) tuple.",
            str(func_item)
        )
        # Return a dummy function that returns the input unchanged
        return lambda x, **kwargs: x, {}

    def _apply_function_list(
        self,
        images: List[np.ndarray],
        function_list: List[FunctionWithArgs]
    ) -> List[np.ndarray]:
        """Apply a list of functions sequentially to images.

        Args:
            images: List of images to process
            function_list: List of functions to apply (can include tuples of (function, kwargs))

        Returns:
            List of processed images
        """
        processed_images = images

        for func_item in function_list:
            # Extract function and args
            func, func_args = self._extract_function_and_args(func_item)

            # Apply the function
            result = self._apply_single_function(processed_images, func, func_args)
            processed_images = [img for img in result]

        return processed_images



    def _apply_single_function(
        self,
        images: List[np.ndarray],
        func: Callable,
        args: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Apply a single processing function with specific args.

        Args:
            images: List of images to process
            func: Processing function to apply
            args: Arguments to pass to the function

        Returns:
            List of processed images
        """
        try:
            result = func(images, **args)

            # Handle different return types
            if isinstance(result, list):
                return result
            if isinstance(result, np.ndarray):
                func_name = getattr(func, '__name__', 'unknown')

                # Check if this is a 3D array (stack of images)
                if result.ndim >= 3:
                    # Convert 3D+ array to list of 2D arrays
                    logger.debug(
                        "Function %s returned a 3D array. Converting to list of 2D arrays.",
                        func_name
                    )
                    return [result[i] for i in range(result.shape[0])]

                # It's a single 2D image
                logger.warning(
                    "Function %s returned a single image instead of a list. Wrapping it.",
                    func_name
                )
                return [result]

            # Unexpected return type
            func_name = getattr(func, '__name__', 'unknown')
            result_type = type(result).__name__
            logger.error(
                "Function %s returned an unexpected type (%s). Returning original images.",
                func_name,
                result_type
            )
            return images

        except Exception as e:
            func_name = getattr(func, '__name__', str(func))
            logger.exception(
                "Error applying processing function %s: %s",
                func_name,
                e
            )
            return images

    def _apply_processing(
        self,
        images: List[np.ndarray],
        context: 'ProcessingContext',
        func: Optional[ProcessingFunc] = None,
        well: Optional[str] = None,
        component: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Apply processing function(s) to a stack (list) of images.

        Note: This method only handles single functions or lists of functions.
        Dictionary mapping of functions to component values is handled by
        prepare_patterns_and_functions before this method is called.

        Args:
            images: List of images (numpy arrays) to process.
            context: Processing context (read-only)
            func: Processing function(s) to apply. Defaults to self.func.
            well: The well being processed (for debugging)
            component: The component being processed (for debugging)

        Returns:
            List of processed images, or the original list if an error occurs.
        """
        logger.debug("Step._apply_processing called: step=%s, well=%s, component=%s, images=%s",
                    self.name, well, component,
                    f"{len(images)} images" if images else "None")

        # Handle empty input
        if not images:
            logger.debug("No images to process, returning empty list")
            return []

        # Get processing function
        processing_func = func if func is not None else self.func

        # Log the processing function
        if isinstance(processing_func, tuple) and callable(processing_func[0]):
            func_name = getattr(processing_func[0], '__name__', str(processing_func[0]))
        else:
            func_name = getattr(processing_func, '__name__', str(processing_func))
        logger.debug("Using processing function: %s", func_name)

        try:
            # Case 1: List of functions or function tuples
            if isinstance(processing_func, list):
                logger.debug("Applying list of %d functions", len(processing_func))
                result = self._apply_function_list(images, processing_func)
                logger.debug("Function list processing complete, returned %d images", len(result))
                return result

            # Case 2: Single function or function tuple
            is_callable = callable(processing_func)
            is_func_tuple = isinstance(processing_func, tuple) and len(processing_func) == 2

            if is_callable or is_func_tuple:
                func, args = self._extract_function_and_args(processing_func)
                logger.debug("Applying single function %s with args %s",
                            getattr(func, '__name__', str(func)), args)
                result = self._apply_single_function(images, func, args)
                logger.debug("Single function processing complete, returned %d images", len(result))
                return result

            # Case 3: Invalid function
            logger.warning("No valid processing function provided. Returning original images.")
            return images

        except Exception as e:
            # Try to get function name, but handle the case where processing_func might be a tuple
            if isinstance(processing_func, tuple) and callable(processing_func[0]):
                func_name = getattr(processing_func[0], '__name__', str(processing_func[0]))
            else:
                func_name = getattr(processing_func, '__name__', str(processing_func))

            logger.exception("Error applying processing function %s: %s", func_name, e)
            return images

    def _prepare_save_operations(
        self,
        context: 'ProcessingContext',
        input_dir: Path,
        output_dir: Path,
        images: List[np.ndarray],
        filenames: List[str]
    ) -> List[dict]:
        """
        Prepare file save operations by resolving paths and generating storage keys.

        Args:
            context: Processing context (read-only)
            input_dir: The logical input directory of the step
            output_dir: The directory where images should be saved
            images: List of processed images (numpy arrays)
            filenames: List of original filenames corresponding to the images

        Returns:
            List of operation dictionaries, each containing:
            - image: The numpy array to save
            - output_path: The file path to save to
            - storage_key: The key to use for storage adapter
        """
        logger.debug("Preparing save operations: step=%s, input_dir=%s, output_dir=%s, images=%s, filenames=%s",
                    self.name, input_dir, output_dir,
                    f"{len(images)} images" if images else "None",
                    f"{len(filenames)} filenames" if filenames else "None")

        # Handle case where images is a single numpy array (not in a list)
        if isinstance(images, np.ndarray):
            logger.debug("Converting single numpy array to list for processing")
            images = [images]
            # If we don't have filenames, create a default one
            if not filenames:
                default_fname = f"result_{self.name.lower().replace(' ', '_')}.tiff"
                filenames = [default_fname]
                logger.debug("Created default filename: %s", default_fname)

        if not images or not filenames:
            logger.warning("No images or filenames to save, returning empty list")
            return []

        # Get well from context for key generation
        well = self.get_well(context)
        logger.debug("Using well from context: %s", well)

        # Import here to avoid circular imports
        from ezstitcher.io.storage_adapter import generate_storage_key

        # Prepare operations
        operations = []

        for img, fname in zip(images, filenames):
            if img is None:
                logger.debug("Skipping None image for filename: %s", fname)
                continue

            # Construct output path
            try:
                rel_path = Path(fname).relative_to(input_dir) if Path(fname).is_relative_to(input_dir) else Path(fname).name
                output_path = output_dir / rel_path
                logger.debug("Output path for %s: %s", fname, output_path)
            except Exception as e:
                logger.error("Error constructing output path for %s: %s", fname, e)
                output_path = output_dir / Path(fname).name
                logger.debug("Falling back to simple output path: %s", output_path)

            try:
                # Create a key using the step name, well, and filename
                component = Path(fname).stem
                storage_key = generate_storage_key(self.name, well, component)

                # Force key to include test_step for test compatibility
                if "test step" in self.name.lower() and "test_step" not in storage_key:
                    logger.warning("Key %s doesn't contain 'test_step' despite step name %s",
                                 storage_key, self.name)
                    # This is a safety check - generate_storage_key should already handle this

                # Log the key being used
                logger.debug("Generated storage key '%s' for file '%s'", storage_key, fname)

                # Add operation to list
                operations.append({
                    "image": img,
                    "output_path": output_path,
                    "storage_key": storage_key,
                    "filename": fname
                })
            except Exception as e:
                logger.error("Error preparing save operation for %s: %s", fname, e, exc_info=True)

        logger.debug("Prepared %d save operations", len(operations))
        return operations

    def _create_save_result(
        self,
        operations: List[dict],
        file_manager: FileManager
    ) -> 'StepResult':
        """
        Create a StepResult with storage operations and file paths.

        Args:
            operations: List of operation dictionaries from _prepare_save_operations
            file_manager: FileManager instance for file operations

        Returns:
            StepResult with storage operations and saved file paths
        """
        logger.debug("Creating StepResult for %d save operations", len(operations))

        # Create result object
        result = self.create_result()

        # Prepare saved_files list for backward compatibility
        saved_files = []

        # Process operations
        for op in operations:
            image = op["image"]
            output_path = op["output_path"]
            storage_key = op["storage_key"]

            # Add storage operation to result
            result.store(storage_key, image)

            # Add output path to saved_files for backward compatibility
            saved_files.append(str(output_path))

            # For disk-based operations, ensure the directory exists
            # This is needed even though we're not writing files here,
            # as the pipeline will need valid paths for fallback operations
            try:
                file_manager.ensure_directory(output_path.parent)
            except Exception as e:
                logger.error("Error ensuring directory %s: %s", output_path.parent, e)

        # Add saved files to results for backward compatibility
        result.add_result("saved_files", saved_files)

        # Log summary
        logger.debug("Created StepResult with %d storage operations and %d saved file paths",
                    len(result.storage_operations), len(saved_files))

        return result

    @property
    def name(self) -> str:
        """The name of this step."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set the name of this step."""
        self._name = value

    @staticmethod
    def create_result():
        """
        Create a new StepResult object.

        Returns:
            Empty StepResult object
        """
        from ezstitcher.core.pipeline import StepResult
        return StepResult()

    @staticmethod
    def get_file_manager(context):
        """
        Get the file manager from the context.

        Args:
            context: Processing context

        Returns:
            FileManager instance or None if not available
        """
        if context and hasattr(context, 'orchestrator') and context.orchestrator:
            return getattr(context.orchestrator, 'file_manager', None)
        return None

    @staticmethod
    def get_well(context):
        """
        Get the current well from the context.

        Args:
            context: Processing context

        Returns:
            Well identifier or None if not available
        """
        if context and hasattr(context, 'well_filter') and context.well_filter:
            return context.well_filter[0]
        return None

    def __repr__(self) -> str:
        """
        String representation of the step.

        Returns:
            A human-readable representation of the step
        """
        components = ", ".join(self.variable_components)
        return f"{self.name} [components={components}, group_by={self.group_by}]"




class PositionGenerationStep(Step):
    """
    A specialized Step for generating positions.

    This step takes processed reference images and generates position files
    for stitching. It stores the positions file in the context for later use.

    This step supports overlay operations for non-legacy storage modes,
    allowing it to work with tools that require filesystem access.
    """

    # Class-level flags for file system requirements - IMMUTABLE
    requires_fs_input = True   # Needs input files on disk for pattern detection
    requires_fs_output = True  # Writes position files directly to disk
    force_disk_output = False  # No need to force disk output beyond normal requirements

    def __init__(
        self,
        name: str = "Position Generation",
        **kwargs
    ):
        """
        Initialize a position generation step.

        Args:
            name: Name of the step
            **kwargs: Additional keyword arguments, including input_dir and output_dir
        """
        super().__init__(
            func=None,  # No processing function needed
            name=name,
            **kwargs
        )

    def process(self, context: 'ProcessingContext') -> 'StepResult':
        """
        Process the step with the given context.

        Args:
            context: The processing context

        Returns:
            StepResult object containing processing results
        """
        logger.info("Processing step: %s", self.name)

        # Create a result object
        result = self.create_result()

        # Get directories from context
        input_dir = context.get_step_input_dir(self)
        output_dir = context.get_step_output_dir(self)
        positions_dir = context.positions_dir

        # Get well from context
        well = context.well_filter[0] if context.well_filter else None
        if not well:
            logger.error("No well filter found in context")
            return result

        # Get orchestrator from context
        orchestrator = context.orchestrator
        if not orchestrator:
            logger.error("No orchestrator found in context")
            return result

        # Ensure output directory exists
        orchestrator.file_manager.ensure_directory(output_dir)
        orchestrator.file_manager.ensure_directory(positions_dir)

        # Generate positions file
        try:
            positions_file, reference_pattern = orchestrator.generate_positions(
                well=well,
                input_dir=input_dir,
                output_dir=positions_dir
            )

            # Store positions file in result
            if positions_file:
                result.add_result("positions_file", str(positions_file))
                result.add_result("reference_pattern", reference_pattern)
                result.add_result("positions_dir", str(positions_dir))

                # Request context updates
                result.update_context("positions_dir", str(positions_dir))
                result.update_context("reference_pattern", reference_pattern)

                logger.info("Generated positions file: %s", positions_file)
            else:
                logger.error("Failed to generate positions file")
        except Exception as e:
            logger.error("Error in position generation step: %s", e, exc_info=True)

        return result


class ImageStitchingStep(Step):
    """
    A step that stitches images using position files.

    This step uses the positions_path from the context to stitch images.

    This step supports overlay operations for non-legacy storage modes,
    allowing it to work with tools that require filesystem access.
    """

    # Class-level flags for file system requirements - IMMUTABLE
    requires_fs_input = True   # Needs input files on disk for stitching
    requires_fs_output = True  # Writes stitched images directly to disk
    force_disk_output = False  # No need to force disk output beyond normal requirements

    def __init__(self, name=None, image_source_dir: Optional[Union[str, Path]] = None, **kwargs):
        """
        Initialize an ImageStitchingStep.

        Args:
            name (str, optional): Name of the step
            image_source_dir (Union[str, Path], optional): Explicit directory containing images to stitch.
                                                        If None, defaults will be attempted later.
            **kwargs: Additional arguments for the step, including input_dir/output_dir overrides.
        """
        # Store image_source_dir before calling super().__init__ which might pop kwargs
        self.image_source_dir = Path(image_source_dir) if image_source_dir else None

        super().__init__(
            func=None,  # ImageStitchingStep doesn't use the standard func mechanism
            name=name or "Image Stitching",
            variable_components=[],  # Empty list for variable_components
            **kwargs
        )

    def process(self, context: 'ProcessingContext') -> 'StepResult':
        """
        Process the step with the given context.

        Args:
            context: The processing context

        Returns:
            StepResult object containing processing results
        """
        logger.info("Processing step: %s", self.name)

        # Create a result object
        result = self.create_result()

        # Get directories from context
        input_dir = context.get_step_input_dir(self)
        output_dir = context.get_step_output_dir(self)

        # Get well from context
        well = context.well_filter[0] if context.well_filter else None
        if not well:
            logger.error("No well filter found in context")
            return result

        # Get orchestrator from context
        orchestrator = context.orchestrator
        if not orchestrator:
            logger.error("No orchestrator found in context")
            return result

        # Determine the directory containing the actual images to stitch
        images_to_stitch_dir = self.image_source_dir if self.image_source_dir else input_dir
        logger.info("ImageStitchingStep will use images from: %s", images_to_stitch_dir)

        # Get positions directory from context
        positions_dir = getattr(context, 'positions_dir', None)

        # If not found in context, try to find at parent level of plate
        if not positions_dir and orchestrator:
            # Access FileManager via orchestrator in context
            file_manager = orchestrator.file_manager
            plate_name = orchestrator.plate_path.name
            parent_positions_dir = orchestrator.plate_path.parent / f"{plate_name}_positions"
            # Use file_manager.exists for consistency and testability
            if file_manager.exists(parent_positions_dir):
                positions_dir = parent_positions_dir
                logger.info("Using positions directory at parent level: %s", positions_dir)
            else:
                # Fallback: Search recursively for a directory containing "positions"
                logger.warning("Positions directory not found relative to plate path. Searching recursively...")
                try:
                    # List all directories recursively
                    all_items = file_manager.list_files(Path(input_dir).parent, recursive=True)
                    # Filter for directories containing "positions" (case-insensitive)
                    possible_dirs = [p for p in all_items if p.is_dir() and "positions" in p.name.lower()]
                    if possible_dirs:
                        positions_dir = possible_dirs[0] # Take the first match
                        logger.info("Found positions directory via recursive search: %s", positions_dir)
                    else:
                        positions_dir = None # Explicitly set to None if not found
                except Exception as find_err:
                    logger.error("Error during recursive search for positions directory: %s", find_err)
                    positions_dir = None

        # If still not found, raise an error
        if not positions_dir:
            logger.error("No positions directory found for well %s", well)
            return result

        # Ensure output directory exists
        orchestrator.file_manager.ensure_directory(output_dir)

        # Stitch images
        try:
            stitched_files = orchestrator.stitch_images(
                well=well,
                input_dir=images_to_stitch_dir,
                output_dir=output_dir,
                positions_file=Path(positions_dir) / f"{well}.csv"
            )

            # Store stitched files in result
            if stitched_files:
                result.add_result("stitched_files", stitched_files)
                result.add_result("well", well)
                result.add_result("output_dir", str(output_dir))
                logger.info("Stitched %d files", len(stitched_files))
            else:
                logger.error("Failed to stitch images")
        except Exception as e:
            logger.error("Error in image stitching step: %s", e, exc_info=True)

        return result


class ZFlatStep(Step):
    """
    Specialized step for Z-stack flattening.

    This step performs Z-stack flattening using the specified method.
    It pre-configures variable_components=['z_index'] and group_by=None.
    """

    PROJECTION_METHODS = {
        "max": "max_projection",
        "mean": "mean_projection",
        "median": "median_projection",
        "min": "min_projection",
        "std": "std_projection",
        "sum": "sum_projection"
    }

    def __init__(
        self,
        method: str = "max",
        variable_components: VariableComponents = ['z_index'],
        group_by: GroupBy = None,
        name: str = None,
        **kwargs
    ):
        """
        Initialize a Z-stack flattening step.

        Args:
            method: Projection method. Options: "max", "mean", "median", "min", "std", "sum"
            variable_components: Components that vary across files (default: ['z_index'])
            group_by: How to group files for processing (default: None)
            name: Human-readable name for the step
            **kwargs: Additional keyword arguments, including input_dir and output_dir
        """
        # Validate method
        if method not in self.PROJECTION_METHODS and method not in self.PROJECTION_METHODS.values():
            raise ValueError(f"Unknown projection method: {method}. "
                            f"Options are: {', '.join(self.PROJECTION_METHODS.keys())}")

        # Get the full method name if a shorthand was provided
        self.method = method
        full_method = self.PROJECTION_METHODS.get(method, method)

        # Initialize the Step with pre-configured parameters
        super().__init__(
            func=(IP.create_projection, {'method': full_method}),
            variable_components=variable_components,
            group_by=group_by,
            name=name or f"{method.capitalize()} Projection",
            **kwargs
        )


class FocusStep(Step):
    """
    Specialized step for focus-based Z-stack processing.

    This step finds the best focus plane in a Z-stack using FocusAnalyzer.
    It pre-configures variable_components=['z_index'] and group_by=None.
    """

    def __init__(
        self,
        focus_options: Optional[Dict[str, Any]] = None,
        variable_components: VariableComponents = ['z_index'],
        group_by: GroupBy = None,
        name: str = None,
        **kwargs
    ):
        """
        Initialize a focus step.

        Args:
            focus_options: Dictionary of focus analyzer options:
                - metric: Focus metric. Options: "combined", "normalized_variance",
                         "laplacian", "tenengrad", "fft" (default: "combined")
            variable_components: Components that vary across files (default: ['z_index'])
            group_by: How to group files for processing (default: None)
            name: Human-readable name for the step
            **kwargs: Additional keyword arguments, including input_dir and output_dir
        """
        # Initialize focus options
        focus_options = focus_options or {'metric': 'combined'}
        metric = focus_options.get('metric', 'combined')

        def process_func(images):
            best_image, _, _ = FocusAnalyzer.select_best_focus(images, metric=metric)
            return best_image

        # Initialize the Step with pre-configured parameters
        super().__init__(
            func=(process_func, {}),
            variable_components=variable_components,
            group_by=group_by,
            name=name or f"Best Focus ({metric})",
            **kwargs
        )


class CompositeStep(Step):
    """
    Specialized step for creating composite images from multiple channels.

    This step creates composite images from multiple channels with specified weights.
    It pre-configures variable_components=['channel'] and group_by=None.
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        variable_components: VariableComponents = ['channel'],
        group_by: GroupBy = None,
        name: str = "Channel Composite",
        **kwargs
    ):
        """
        Initialize a channel compositing step.

        Args:
            weights: List of weights for each channel. If None, equal weights are used.
            variable_components: Components that vary across files (default: ['channel'])
            group_by: How to group files for processing (default: None)
            name: Human-readable name for the step
            **kwargs: Additional keyword arguments, including input_dir and output_dir
        """
        # Initialize the Step with pre-configured parameters
        super().__init__(
            func=(IP.create_composite, {'weights': weights}),
            variable_components=variable_components,
            group_by=group_by,
            name=name,
            **kwargs
        )


class NormStep(Step):
    """
    Specialized step for image normalization.

    This step performs percentile-based normalization on images.
    It pre-configures func=IP.stack_percentile_normalize with customizable percentile parameters.
    """

    def __init__(
        self,
        low_percentile: float = 0.1,
        high_percentile: float = 99.9,
        variable_components: VariableComponents = ['site'],
        group_by: GroupBy = None,
        name: str = "Percentile Normalization",
        **kwargs
    ):
        """
        Initialize a normalization step.

        Args:
            low_percentile: Low percentile for normalization (0-100)
            high_percentile: High percentile for normalization (0-100)
            variable_components: Components that vary across files (default: ['site'])
            group_by: How to group files for processing (default: None)
            name: Human-readable name for the step
            **kwargs: Additional keyword arguments, including input_dir and output_dir
        """
        # Initialize the Step with pre-configured parameters
        super().__init__(
            func=(IP.stack_percentile_normalize, {
                'low_percentile': low_percentile,
                'high_percentile': high_percentile
            }),
            variable_components=variable_components,
            group_by=group_by,
            name=name,
            **kwargs
        )
