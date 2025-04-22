"""
Core implementation of the Flexible Pipeline Architecture.

This module provides a flexible, declarative API for defining image processing
pipelines in EZStitcher. It builds on the strengths of the current
process_patterns_with_variable_components method while adding an object-oriented
core with a functional interface.
"""

from typing import Dict, List, Union, Callable, Any, TypeVar, Optional
import logging
from pathlib import Path
import copy
import numpy as np

# Import core components
from ezstitcher.core.file_system_manager import FileSystemManager

# Type definitions
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
        logger.info(f"Processing step: {self.name}")

        # Get directories and microscope handler
        input_dir = self.input_dir
        output_dir = self.output_dir
        well_filter = self.well_filter or context.well_filter
        # Get microscope handler from context - this will raise an error if not present
        microscope_handler = context.microscope_handler

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

            logger.info(f"Processing well: {well}")
            well_results = {}

            # Prepare patterns and functions
            grouped_patterns, component_to_funcs = _prepare_patterns_and_functions(
                patterns, self.func, component=self.group_by
            )

            for component_value in grouped_patterns.keys():
                component_patterns = grouped_patterns[component_value]
                component_func = component_to_funcs[component_value]

                # Load all images for this group
                for pattern in component_patterns:
                    matching_files = microscope_handler.parser.path_list_from_pattern(input_dir, pattern)
                    images = [FileSystemManager.load_image(str(Path(input_dir) / filename)) for filename in matching_files]
                    images = [img for img in images if img is not None]

                    if not images:
                        continue  # Skip if no valid images found

                    # Store original number of images to detect flattening
                    original_image_count = len(images)
                    images = self._apply_processing(images, func=component_func)

                    # Save images and get output files
                    output_files = self._save_images(input_dir, output_dir, images, matching_files)
                    well_results[component_value] = output_files

            # Store results for this well
            results[well] = well_results

        context.results = results
        return context



    def _apply_processing(self, images, func=None):
        """Apply processing function(s) to images.

        Args:
            images: Images to process
            func: Processing function to use (defaults to self.func)

        Returns:
            Processed images, always as a list even if only one image is returned
        """
        if not images:
            return images

        # Use provided function or fall back to the step's function
        processing_func = func if func is not None else self.func

        if isinstance(processing_func, list):
            processed = images
            for f in processing_func:
                processed = f(processed, **(self.processing_args or {}))
        elif callable(processing_func):
            processed = processing_func(images, **(self.processing_args or {}))
        else:
            processed = images

        # Ensure the result is always a list
        if isinstance(processed, np.ndarray) and processed.ndim >= 2:
            # Single image was returned, wrap it in a list
            return [processed]
        elif isinstance(processed, list):
            # Already a list, return as is
            return processed
        else:
            # Unknown format, return original images
            return images

    def _save_images(self, input_dir, output_dir, images, filenames):
        """Save processed images."""
        if not output_dir or not images or not filenames:
            return []

        FileSystemManager.ensure_directory(output_dir)

        # Clean up old files if working in place
        if input_dir is output_dir:
            for filename in filenames:
                FileSystemManager.delete_file(Path(output_dir) / filename)

        # Initialize output files list
        output_files = []

        # Handle single image result (e.g., from flattening)
        if isinstance(images, np.ndarray):
            output_path = Path(output_dir) / filenames[0]
            FileSystemManager.save_image(str(output_path), images)
            output_files.append(str(output_path))
        # Handle list with a single image
        elif isinstance(images, list) and len(images) == 1:
            output_path = Path(output_dir) / filenames[0]
            FileSystemManager.save_image(str(output_path), images[0])
            output_files.append(str(output_path))
        # Handle multiple images
        elif isinstance(images, list) and images:
            for i, img in enumerate(images):
                if i < len(filenames):
                    output_path = Path(output_dir) / filenames[i]
                    FileSystemManager.save_image(str(output_path), img)
                    output_files.append(str(output_path))

        return output_files

    def __repr__(self) -> str:
        """
        String representation of the step.

        Returns:
            A human-readable representation of the step
        """
        components = ", ".join(self.variable_components)
        output_dir_str = f"→ {str(self.output_dir)}" if self.output_dir else ""
        return f"{self.name} [components={components}, group_by={self.group_by}] {output_dir_str}"


class Pipeline:
    """
    A sequence of processing steps.

    A Pipeline is a sequence of processing steps that are executed in order.
    Each step takes input from the previous step's output and produces new output.

    Attributes:
        steps: The sequence of processing steps
        input_dir: The input directory
        output_dir: The output directory
        well_filter: Wells to process
        name: Human-readable name for the pipeline
        _config: Configuration parameters
    """

    def __init__(
        self,
        steps: List[Step] = None,
        input_dir: str = None,
        output_dir: str = None,
        well_filter: WellFilter = None,
        name: str = None
    ):
        """
        Initialize a pipeline.

        Args:
            steps: The sequence of processing steps
            input_dir: The input directory
            output_dir: The output directory
            well_filter: Wells to process
            name: Human-readable name for the pipeline
        """
        self.steps = []
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.well_filter = well_filter
        self.name = name or f"Pipeline({len(steps or [])} steps)"
        self._config = {}

        # Add steps if provided
        if steps:
            for step in steps:
                self.add_step(step)

    def add_step(self, step: Step, output_dir: str = None) -> 'Pipeline':
        """
        Add a step to the pipeline.

        This method automatically sets input and output directories for the step
        if they are not already specified:
        - If input_dir is not specified, it uses the output_dir of the previous step,
          or the pipeline's input_dir if this is the first step.
        - If output_dir is not specified, it uses the step's input_dir.

        Args:
            step: The step to add
            output_dir: Optional output directory for the step

        Returns:
            Self, for method chaining
        """
        # Override output_dir if provided
        if output_dir:
            step.output_dir = output_dir

        # Set input_dir if not specified
        if not step.input_dir:
            if self.steps:  # Not the first step
                # Use the output_dir of the previous step
                step.input_dir = self.steps[-1].output_dir
            else:  # First step
                # Use the pipeline's input_dir
                step.input_dir = self.input_dir

        # Set output_dir if not specified
        if not step.output_dir:
            # Use the pipeline's output_dir if available, otherwise use the same directory as input_dir
            step.output_dir = self.output_dir or step.input_dir

        # If this is the first step and pipeline's input_dir is not set, use step's input_dir
        if not self.steps and not self.input_dir and step.input_dir:
            self.input_dir = step.input_dir

        # If pipeline's output_dir is not set, use the last step's output_dir
        if not self.output_dir and step.output_dir:
            self.output_dir = step.output_dir

        self.steps.append(step)
        return self

    def add_steps(self, *steps: Step) -> 'Pipeline':
        """
        Add multiple steps to the pipeline.

        Args:
            *steps: The steps to add

        Returns:
            Self, for method chaining
        """
        for step in steps:
            self.add_step(step)
        return self

    def set_input(self, input_dir: str) -> 'Pipeline':
        """
        Set the input directory.

        Args:
            input_dir: The input directory

        Returns:
            Self, for method chaining
        """
        self.input_dir = input_dir
        return self

    def set_output(self, output_dir: str) -> 'Pipeline':
        """
        Set the output directory.

        Args:
            output_dir: The output directory

        Returns:
            Self, for method chaining
        """
        self.output_dir = output_dir
        return self

    def set_well_filter(self, well_filter: WellFilter) -> 'Pipeline':
        """
        Set the well filter.

        Args:
            well_filter: Wells to process

        Returns:
            Self, for method chaining
        """
        self.well_filter = well_filter
        return self

    def set_config(self, **kwargs) -> 'Pipeline':
        """
        Set configuration parameters.

        Args:
            **kwargs: Configuration parameters

        Returns:
            Self, for method chaining
        """
        self._config.update(kwargs)
        return self

    def clone(self) -> 'Pipeline':
        """
        Create a copy of this pipeline.

        Returns:
            A new Pipeline instance with the same steps and configuration
        """
        new_pipeline = Pipeline(
            steps=copy.deepcopy(self.steps),
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            well_filter=self.well_filter,
            name=f"Clone of {self.name}"
        )
        new_pipeline._config = copy.deepcopy(self._config)
        return new_pipeline

    def run(
        self,
        input_dir: str = None,
        output_dir: str = None,
        well_filter: WellFilter = None,
        microscope_handler = None
    ) -> Dict[str, Any]:
        """
        Execute the pipeline.

        Args:
            input_dir: Optional input directory override
            output_dir: Optional output directory override
            well_filter: Optional well filter override
            microscope_handler: Optional microscope handler override

        Returns:
            The results of the pipeline execution

        Raises:
            ValueError: If no input directory is specified
        """
        logger.info(f"Running pipeline: {self.name}")

        # Use provided values or fall back to instance values
        effective_input = input_dir or self.input_dir
        effective_output = output_dir or self.output_dir
        effective_well_filter = well_filter or self.well_filter

        # If input_dir is still not set, try to get it from the first step
        if not effective_input and self.steps:
            effective_input = self.steps[0].input_dir

        if not effective_input:
            raise ValueError("Input directory must be specified")

        logger.info(f"Input directory: {effective_input}")
        logger.info(f"Output directory: {effective_output}")
        logger.info(f"Well filter: {effective_well_filter}")

        # Initialize context
        context = ProcessingContext(
            input_dir=effective_input,
            output_dir=effective_output,
            well_filter=effective_well_filter,
            microscope_handler=microscope_handler,
            config=self._config
        )

        # Execute each step
        for i, step in enumerate(self.steps):
            logger.info(f"Executing step {i+1}/{len(self.steps)}: {step}")
            context = step.process(context)

        logger.info(f"Pipeline completed: {self.name}")
        return context.results

    def __repr__(self) -> str:
        """
        String representation of the pipeline.

        Returns:
            A human-readable representation of the pipeline
        """
        steps_repr = "\n  ".join(repr(step) for step in self.steps)
        input_dir_str = str(self.input_dir) if self.input_dir else "None"
        output_dir_str = str(self.output_dir) if self.output_dir else "None"
        return (f"{self.name}\n"
                f"  Input: {input_dir_str}\n"
                f"  Output: {output_dir_str}\n"
                f"  Well filter: {self.well_filter}\n"
                f"  Steps:\n  {steps_repr}")


class ProcessingContext:
    """
    Maintains state during pipeline execution.

    The ProcessingContext holds input/output directories, well filter, configuration,
    and results during pipeline execution.

    Attributes:
        input_dir: The input directory
        output_dir: The output directory
        well_filter: Wells to process
        config: Configuration parameters
        results: Processing results
    """

    def __init__(
        self,
        input_dir: str = None,
        output_dir: str = None,
        well_filter: WellFilter = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize the processing context.

        Args:
            input_dir: The input directory
            output_dir: The output directory
            well_filter: Wells to process
            config: Configuration parameters
            **kwargs: Additional context attributes
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.well_filter = well_filter
        self.config = config or {}
        self.results = {}

        # Add any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


def _prepare_patterns_and_functions(patterns, processing_funcs, component='default'):
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
        # Get functions for this component
        if isinstance(processing_funcs, dict) and comp_value in processing_funcs:
            # Direct mapping for this component
            component_to_funcs[comp_value] = processing_funcs[comp_value]
        elif isinstance(processing_funcs, dict) and component == 'channel':
            # For channel grouping, use the channel-specific function if available
            component_to_funcs[comp_value] = processing_funcs.get(comp_value, processing_funcs)
        else:
            # Use the same function for all components
            component_to_funcs[comp_value] = processing_funcs

    return grouped_patterns, component_to_funcs


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

