"""
Step classes for the pipeline architecture.

This module contains the Step class and its specialized subclasses for
different types of processing operations.
"""

from typing import Dict, List, Union, Callable, Any, TypeVar, Optional
import logging
from pathlib import Path
import numpy as np

# Import core components
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.utils import prepare_patterns_and_functions
# Removed adapt_func_to_stack import


# Type definitions
# Note: All functions in ProcessingFunc are now expected to accept List[np.ndarray]
# and return List[np.ndarray]. Use utils.stack() to wrap single-image functions.
ProcessingFunc = Union[Callable, Dict[str, Callable], List[Callable]]
VariableComponents = List[str]
GroupBy = Optional[str]
WellFilter = Optional[List[str]]
T = TypeVar('T')  # For generic return types

# Configure logging
logger = logging.getLogger(__name__)


class Step:
    """
    A processing step in a pipeline.

    A Step encapsulates a processing operation that can be applied to images.
    It mirrors the functionality of process_patterns_with_variable_components
    while providing a more object-oriented interface.

    Attributes:
        func: The processing function(s) to apply
        variable_components: Components that vary across files (e.g., 'z_index', 'channel')
        group_by: How to group files for processing (e.g., 'channel', 'site')
        input_dir: The input directory
        output_dir: The output directory
        well_filter: Wells to process
        processing_args: Additional arguments to pass to the processing function
        name: Human-readable name for the step
    """

    def __init__(
        self,
        func: ProcessingFunc,
        variable_components: VariableComponents = None,
        group_by: GroupBy = None,
        input_dir: str = None,
        output_dir: str = None,
        well_filter: WellFilter = None,
        processing_args: Dict[str, Any] = None,
        name: str = None
    ):
        """
        Initialize a processing step.

        Args:
            func: The processing function(s) to apply
            variable_components: Components that vary across files
            group_by: How to group files for processing
            input_dir: The input directory
            output_dir: The output directory
            well_filter: Wells to process
            processing_args: Additional arguments to pass to the processing function
            name: Human-readable name for the step
        """
        self.func = func
        self.variable_components = variable_components or []
        self.group_by = group_by
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.well_filter = well_filter
        self.processing_args = processing_args or {}
        self.name = name or self._generate_name()

    def _generate_name(self) -> str:
        """
        Generate a descriptive name based on the function.

        Returns:
            A human-readable name for the step
        """
        if isinstance(self.func, dict):
            funcs = ", ".join(f"{k}:{f.__name__}" for k, f in self.func.items())
            return f"ChannelMappedStep({funcs})"
        elif isinstance(self.func, list):
            funcs = ", ".join(f.__name__ for f in self.func)
            return f"MultiStep({funcs})"
        else:
            return f"Step({self.func.__name__})"

    def process(self, context: 'ProcessingContext') -> 'ProcessingContext':
        """
        Process the step with the given context.

        This method applies the step's processing function to the input files
        and saves the results to the output directory.

        Args:
            context: The processing context

        Returns:
            The updated processing context
        """
        logger.info("Processing step: %s", self.name)

        # Get directories and microscope handler
        input_dir = self.input_dir
        output_dir = self.output_dir
        well_filter = self.well_filter or context.well_filter
        orchestrator = context.orchestrator  # Required, will raise AttributeError if missing = context.microscope_handler
        microscope_handler = orchestrator.microscope_handler

        if not input_dir:
            raise ValueError("Input directory must be specified")

        # Get patterns with variable components
        patterns_by_well = microscope_handler.auto_detect_patterns(
            input_dir,
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

            # Prepare patterns and functions
            grouped_patterns, component_to_funcs = prepare_patterns_and_functions(
                patterns, self.func, component=self.group_by
            )

            # Process each component
            for component_value, component_patterns in grouped_patterns.items():
                component_func = component_to_funcs[component_value]
                output_files = []

                # Process each pattern
                for pattern in component_patterns:
                    # Find matching files
                    matching_files = microscope_handler.parser.path_list_from_pattern(input_dir, pattern)

                    # Load images
                    try:
                        images = [FileSystemManager.load_image(str(Path(input_dir) / filename)) for filename in matching_files]
                        images = [img for img in images if img is not None]
                    except Exception as e:
                        logger.error("Error loading images: %s", str(e))
                        images = []

                    if not images:
                        continue  # Skip if no valid images found

                    # Process the images
                    images = self._apply_processing(images, func=component_func)

                    # Save images and get output files
                    pattern_files = self._save_images(input_dir, output_dir, images, matching_files)
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

    def _apply_processing(self, images: List[np.ndarray], func: Optional[ProcessingFunc] = None) -> List[np.ndarray]:
        """Apply processing function(s) to a stack (list) of images.

        Note: This method only handles single functions or lists of functions.
        Dictionary mapping of functions to component values is handled by
        prepare_patterns_and_functions before this method is called.

        Args:
            images: List of images (numpy arrays) to process.
            func: Stack-aware processing function or list of functions. Defaults to self.func.

        Returns:
            List of processed images, or the original list if an error occurs.
        """
        if not images:
            return []

        processing_func = func if func is not None else self.func
        args_to_pass = self.processing_args or {}

        try:
            # Case 1: List of functions - apply sequentially
            if isinstance(processing_func, list):
                processed_images = images
                for f in processing_func:
                    processed_images = self._apply_processing(processed_images, func=f)
                    # Ensure all images are 2D after each function application
                    processed_images = [self._ensure_2d(img) for img in processed_images]
                return processed_images

            # Case 2: Single callable function
            elif callable(processing_func):
                result = processing_func(images, **args_to_pass)

                # Handle different return types
                if isinstance(result, list):
                    return [self._ensure_2d(img) for img in result]
                if isinstance(result, np.ndarray):
                    logger.warning("Function %s returned a single image instead of a list. Wrapping it.",
                                  getattr(processing_func, '__name__', 'unknown'))
                    return [self._ensure_2d(result)]

                # Unexpected return type
                logger.error("Function %s returned an unexpected type (%s). Returning original images.",
                            getattr(processing_func, '__name__', 'unknown'), type(result).__name__)
                return images

            # Case 3: Invalid function
            else:
                logger.warning("No valid processing function provided. Returning original images.")
                return images

        except Exception as e:
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
            if input_dir is output_dir:
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

    def __repr__(self) -> str:
        """
        String representation of the step.

        Returns:
            A human-readable representation of the step
        """
        components = ", ".join(self.variable_components)
        output_dir_str = f"→ {str(self.output_dir)}" if self.output_dir else ""
        return f"{self.name} [components={components}, group_by={self.group_by}] {output_dir_str}"

    def _context_wrapper(self, images, **kwargs):
        """
        Common wrapper for context-aware steps.

        Args:
            images: Images to process
            **kwargs: Additional arguments

        Returns:
            Processed images
        """
        context = kwargs.get('context')
        if not context:
            return images

        # Extract common context attributes
        well = context.well
        dirs = context.dirs
        stitcher = getattr(context, 'stitcher', None)

        # Call the step-specific implementation
        self._process_with_context(context, well, dirs, stitcher)

        return images

    def _process_with_context(self, context, well, dirs, stitcher):
        """
        Process using context information.

        This method should be implemented by subclasses that need context access.

        Args:
            context: Processing context
            well: Well identifier
            dirs: Directory dictionary
            stitcher: Stitcher instance
        """
        # Default implementation does nothing


class PositionGenerationStep(Step):
    """
    A specialized Step for generating positions.

    This step takes processed reference images and generates position files
    for stitching. It stores the positions file in the context for later use.
    """

    def __init__(
        self,
        name: str = "Position Generation",
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,  # Output directory for positions files
        processing_args: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a position generation step.

        Args:
            name: Name of the step
            input_dir: Input directory
            output_dir: Output directory (for positions files)
            processing_args: Additional arguments for the processing function
        """
        super().__init__(
            func=None,  # No processing function needed
            name=name,
            input_dir=input_dir,
            output_dir=output_dir,
            processing_args=processing_args
        )

    def process(self, context):
        """
        Generate positions for stitching and store them in the context.
        """
        logger.info("Processing step: %s", self.name)

        if self.output_dir is self.input_dir:
            self.output_dir = self.input_dir.parent / f"{self.input_dir.name}_positions"
            logger.info(f"Input and output directories are the same, using default positions directory: {self.output_dir}")

        # Get required objects from context
        well = context.well_filter[0] if context.well_filter else None
        orchestrator = context.orchestrator  # Required, will raise AttributeError if missing
        input_dir = self.input_dir or context.input_dir
        positions_dir = self.output_dir or context.output_dir

        # Call the generate_positions method
        positions_dir, reference_pattern = orchestrator.generate_positions(well, input_dir, positions_dir)

        # Store in context
        context.positions_dir = positions_dir
        context.reference_pattern = reference_pattern
        return context


class ImageStitchingStep(Step):
    """
    A specialized Step for stitching images.

    This step takes processed images and stitches them using the positions file
    generated by a previous PositionGenerationStep.
    """

    def __init__(
        self,
        name: str = "Image Stitching",
        input_dir: Optional[Path] = None,
        positions_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        processing_args: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an image stitching step.

        Args:
            name: Name of the step
            input_dir: Input directory
            output_dir: Output directory
            processing_args: Additional arguments for the processing function
        """
        super().__init__(
            func=None,  # No processing function needed
            name=name,
            input_dir=input_dir,
            output_dir=output_dir, ### stitched images folder
            processing_args=processing_args
        )
        self.positions_dir = positions_dir

    def process(self, context):
        """
        Stitch images using the positions file from the context.
        """
        logger.info("Processing step: %s", self.name)

        if not self.positions_dir:
            self.positions_dir = FileSystemManager.find_directory_substring_recursive(self.input_dir.parent, "positions")
            if self.positions_dir is None:
                raise ValueError("No positions directory found")
            else:
                logger.info(f"positions_dir not provided, using detected positoins directory in parent @: {self.positions_dir}")


        if self.output_dir is self.input_dir:
            self.output_dir = self.input_dir.parent / f"{self.input_dir.name}_stitched"
            logger.info(f"Input and output directories are the same, using default positions directory: {self.output_dir}")

        # Get required objects from context
        well = context.well_filter[0] if context.well_filter else None
        orchestrator = context.orchestrator  # Required, will raise AttributeError if missing
        positions_dir = getattr(context, '', None)  # Oppositions_dirtional, check if exists
        input_dir = self.input_dir or context.input_dir
        output_dir = self.output_dir or context.output_dir
        positions_path = FileSystemManager.find_file_recursive(self.positions_dir, f"{well}.csv")

        # Call the stitch_images method
        orchestrator.stitch_images(well, input_dir, output_dir, positions_path)

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
