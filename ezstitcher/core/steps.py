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

from typing import Dict, List, Union, Callable, Any, TypeVar, Optional, Sequence, Tuple
import logging
from pathlib import Path
import numpy as np

# Import core components
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.utils import prepare_patterns_and_functions
from ezstitcher.core.abstract_step import AbstractStep
from ezstitcher.core.image_processor import ImageProcessor as IP
from ezstitcher.core.focus_analyzer import FocusAnalyzer
# Removed adapt_func_to_stack import


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

    Attributes:
        func: The processing function(s) to apply
        variable_components: Components that vary across files (e.g., 'z_index', 'channel')
        group_by: How to group files for processing (e.g., 'channel', 'site')
        name: Human-readable name for the step
    """

    def __init__(
        self,
        func: ProcessingFunc,
        variable_components: VariableComponents = ['site'],
        group_by: GroupBy = None,
        name: str = None,
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

    def process(self, context: 'ProcessingContext') -> 'ProcessingContext':
        """
        Process the step with the given context.

        Args:
            context: The processing context containing pre-computed paths and other state

        Returns:
            The updated processing context with processing results
        """
        logger.info("Processing step: %s", self.name)

        # Get directories and microscope handler from context
        input_dir = context.get_step_input_dir(self)
        output_dir = context.get_step_output_dir(self)
        well_filter = context.well_filter
        orchestrator = context.orchestrator  # Required, will raise AttributeError if missing
        microscope_handler = orchestrator.microscope_handler

        # Find the actual directory containing images
        # This works whether input_dir is a plate folder or a subfolder
        actual_input_dir = FileSystemManager.find_image_directory(Path(input_dir))
        logger.debug("Using actual image directory: %s", actual_input_dir)

        # Get patterns with variable components
        patterns_by_well = microscope_handler.auto_detect_patterns(
            actual_input_dir,
            well_filter=well_filter,
            variable_components=self.variable_components
        )

        # Process each well
        results = {}
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
                    matching_files = microscope_handler.parser.path_list_from_pattern(actual_input_dir, pattern)

                    # Load images
                    try:
                        images = [FileSystemManager.load_image(str(Path(actual_input_dir) / filename)) for filename in matching_files]
                        images = [img for img in images if img is not None]
                    except Exception as e:
                        logger.error("Error loading images: %s", str(e))
                        images = []

                    if not images:
                        continue  # Skip if no valid images found

                    # Process the images with component-specific args
                    try:
                        images = self._apply_processing(images, func=component_func)
                    except Exception as e:
                        logger.error("Error applying processing function: %s", str(e))
                        continue

                    # Save images and get output files
                    pattern_files = self._save_images(actual_input_dir, output_dir, images, matching_files)
                    if pattern_files:
                        output_files.extend(pattern_files)

                # Store results for this component
                if output_files:
                    well_results[component_value] = output_files

            # Store results for this well
            results[well] = well_results

        # Store results in context
        context.results = results
        return context



    def _ensure_2d(self, img):
        """Ensure an image is 2D by reducing dimensions if needed."""
        if not isinstance(img, np.ndarray) or img.ndim <= 2:
            return img

        # Try to squeeze out singleton dimensions first
        squeezed = np.squeeze(img)
        if squeezed.ndim <= 2:
            return squeezed

        # If still not 2D, take first slice until it is
        result = img
        while result.ndim > 2:
            result = result[0]

        logger.warning("Reduced image dimensions from %dD to 2D", img.ndim)
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
            processed_images = [self._ensure_2d(img) for img in result]

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
        func: Optional[ProcessingFunc] = None
    ) -> List[np.ndarray]:
        """Apply processing function(s) to a stack (list) of images.

        Note: This method only handles single functions or lists of functions.
        Dictionary mapping of functions to component values is handled by
        prepare_patterns_and_functions before this method is called.

        Functions can be specified in several ways:
        - A single callable function
        - A tuple of (function, kwargs)
        - A list of functions or (function, kwargs) tuples

        Args:
            images: List of images (numpy arrays) to process.
            func: Processing function(s) to apply. Defaults to self.func.

        Returns:
            List of processed images, or the original list if an error occurs.
        """
        # Handle empty input
        if not images:
            return []

        # Get processing function
        processing_func = func if func is not None else self.func

        try:
            # Case 1: List of functions or function tuples
            if isinstance(processing_func, list):
                return self._apply_function_list(images, processing_func)

            # Case 2: Single function or function tuple
            is_callable = callable(processing_func)
            is_func_tuple = isinstance(processing_func, tuple) and len(processing_func) == 2

            if is_callable or is_func_tuple:
                func, args = self._extract_function_and_args(processing_func)
                return self._apply_single_function(images, func, args)

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

    def _save_images(self, input_dir, output_dir, images, filenames):
        """Save processed images.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            images: Images to save
            filenames: Filenames to use

        Returns:
            list: Paths to saved images
        """
        if not output_dir or not images or not filenames:
            return []

        try:
            # Ensure output directory exists
            FileSystemManager.ensure_directory(output_dir)

            # Clean up old files if working in place
            if input_dir == output_dir:
                for filename in filenames:
                    FileSystemManager.delete_file(Path(output_dir) / filename)

            # Initialize output files list
            output_files = []

            # Convert to list if it's a single image
            if isinstance(images, np.ndarray):
                images = [images]
                filenames = [filenames[0]]

            # Save each image
            for i, img in enumerate(images):
                if i < len(filenames):
                    output_path = Path(output_dir) / filenames[i]
                    FileSystemManager.save_image(str(output_path), img)
                    output_files.append(str(output_path))

            return output_files

        except Exception as e:
            logger.error("Error saving images: %s", str(e))
            return []

    @property
    def name(self) -> str:
        """The name of this step."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set the name of this step."""
        self._name = value



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
    """

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

    def process(self, context: 'ProcessingContext') -> 'ProcessingContext':
        """
        Generate positions for stitching and store them in the context.

        Args:
            context: The processing context containing pre-computed paths and other state

        Returns:
            The updated processing context with positions information
        """
        logger.info("Processing step: %s", self.name)

        # Get required objects from context
        well = context.well_filter[0] if context.well_filter else None
        orchestrator = context.orchestrator  # Required, will raise AttributeError if missing
        input_dir = context.get_step_input_dir(self)
        output_dir = context.get_step_output_dir(self)

        # Call the generate_positions method
        positions_file, reference_pattern = orchestrator.generate_positions(well, input_dir, output_dir)

        # Store in context
        context.positions_dir = output_dir
        context.reference_pattern = reference_pattern
        return context


class ImageStitchingStep(Step):
    """
    A step that stitches images using position files.

    This step uses the positions_path from the context to stitch images.
    """

    def __init__(self, name=None, **kwargs):
        """
        Initialize an ImageStitchingStep.

        Args:
            name (str, optional): Name of the step
            **kwargs: Additional arguments for the step
        """
        super().__init__(
            func=None,  # ImageStitchingStep doesn't use the standard func mechanism
            name=name or "Image Stitching",
            variable_components=[],  # Empty list for variable_components
            **kwargs
        )

    def process(self, context: 'ProcessingContext') -> 'ProcessingContext':
        """
        Stitch images using the positions file.

        Args:
            context: The processing context containing pre-computed paths and other state

        Returns:
            The updated processing context
        """
        logger.info("Processing step: %s", self.name)

        # Get orchestrator from context
        orchestrator = context.orchestrator
        if not orchestrator:
            raise ValueError("ImageStitchingStep requires an orchestrator in the context")

        # Get well from context
        well = context.well_filter[0] if context.well_filter else None
        if not well:
            raise ValueError("ImageStitchingStep requires a well filter in the context")

        # Get directories from context
        input_dir = context.get_step_input_dir(self)
        output_dir = context.get_step_output_dir(self)

        # Get positions directory from context or find it
        positions_dir = getattr(context, 'positions_dir', None)

        # If not found in context, try to find at parent level of plate
        if not positions_dir and orchestrator:
            plate_name = orchestrator.plate_path.name
            parent_positions_dir = orchestrator.plate_path.parent / f"{plate_name}_positions"
            if parent_positions_dir.exists():
                positions_dir = parent_positions_dir
                logger.info(f"Using positions directory at parent level: {positions_dir}")
            else:
                # Fallback to existing logic if no positions directory is found
                positions_dir = FileSystemManager.find_directory_substring_recursive(
                    Path(input_dir).parent, "positions")

        # If still not found, raise an error
        if not positions_dir:
            raise ValueError(f"No positions directory found for well {well}")

        # Call the stitch_images method
        orchestrator.stitch_images(
            well=well,
            input_dir=input_dir,
            output_dir=output_dir,
            positions_file=Path(positions_dir) / f"{well}.csv"
        )

        return context




def group_patterns_by(patterns, component, microscope_handler=None):
    """
    Group patterns by the specified component.

    Args:
        patterns (list): Patterns to group
    Returns:
        dict: Dictionary mapping component values to lists of patterns
    """
    grouped_patterns = {}
    for pattern in patterns:
        # Extract the component value from the pattern
        component_value = microscope_handler.parser.parse_filename(pattern)[component]
        if component_value not in grouped_patterns:
            grouped_patterns[component_value] = []
        grouped_patterns[component_value].append(pattern)
    return grouped_patterns


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
