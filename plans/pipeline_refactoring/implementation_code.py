"""
Implementation code for the Flexible Pipeline Architecture.

This module provides a flexible, declarative API for defining image processing
pipelines in EZStitcher. It builds on the strengths of the current
process_patterns_with_variable_components method while adding an object-oriented
core with a functional interface.
"""

from typing import Dict, List, Union, Callable, Optional, Any, TypeVar
import logging
from pathlib import Path
import copy

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

        This method mirrors the logic in process_patterns_with_variable_components
        while providing a more object-oriented interface.

        Args:
            context: The processing context

        Returns:
            The updated processing context
        """
        logger.info(f"Processing step: {self.name}")

        # Use provided values or fall back to context values
        input_dir = self.input_dir or context.input_dir
        output_dir = self.output_dir or context.output_dir
        well_filter = self.well_filter or context.well_filter

        if not input_dir:
            raise ValueError("Input directory must be specified")

        # Get file patterns with variable components
        patterns = get_file_patterns(
            input_dir,
            self.variable_components,
            well_filter
        )

        # Process each pattern
        results = {}

        for well, well_patterns in patterns.items():
            if well_filter and well not in well_filter:
                continue

            logger.info(f"Processing well: {well}")

            well_results = process_patterns(
                well_patterns,
                self.func,
                self.group_by,
                self.processing_args
            )

            results[well] = well_results

            # Save results if output directory is specified
            if output_dir:
                save_results(well_results, output_dir, well)

        # Update context with results
        context.results = results

        return context

    def __repr__(self) -> str:
        """
        String representation of the step.

        Returns:
            A human-readable representation of the step
        """
        components = ", ".join(self.variable_components)
        return (f"{self.name} [components={components}, group_by={self.group_by}] "
                f"{'→ ' + self.output_dir if self.output_dir else ''}")


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
        self.steps = list(steps) if steps else []
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.well_filter = well_filter
        self.name = name or f"Pipeline({len(self.steps)} steps)"
        self._config = {}

    def add_step(self, step: Step, output_dir: str = None) -> 'Pipeline':
        """
        Add a step to the pipeline.

        Args:
            step: The step to add
            output_dir: Optional output directory for the step

        Returns:
            Self, for method chaining
        """
        if output_dir:
            step.output_dir = output_dir
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
        self.steps.extend(steps)
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
        well_filter: WellFilter = None
    ) -> Dict[str, Any]:
        """
        Execute the pipeline.

        Args:
            input_dir: Optional input directory override
            output_dir: Optional output directory override
            well_filter: Optional well filter override

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
        return (f"{self.name}\n"
                f"  Input: {self.input_dir}\n"
                f"  Output: {self.output_dir}\n"
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


# Utility functions

def get_file_patterns(
    input_dir: str,
    variable_components: VariableComponents,
    well_filter: WellFilter = None
) -> Dict[str, List[str]]:
    """
    Get file patterns with variable components.

    This function finds files in the input directory and groups them by well.
    It then extracts patterns with the specified variable components.

    Args:
        input_dir: The input directory
        variable_components: Components that vary across files
        well_filter: Wells to process

    Returns:
        A dictionary mapping wells to lists of file patterns
    """
    # This is a simplified implementation - in practice, we would use
    # the microscope handler to find files and extract patterns

    # Find all files in the input directory
    input_path = Path(input_dir)
    all_files = list(input_path.glob("**/*.tif*"))

    # Group files by well
    well_files = {}

    for file_path in all_files:
        # Extract well from filename
        import re
        well_match = re.search(r'([A-Za-z]\d+)', file_path.name)
        well = well_match.group(1) if well_match else "A01"

        if well_filter and well not in well_filter:
            continue

        if well not in well_files:
            well_files[well] = []

        well_files[well].append(str(file_path))

    # Extract patterns with variable components
    patterns = {}

    for well, files in well_files.items():
        # This is a simplified implementation - in practice, we would use
        # the microscope handler to extract patterns
        patterns[well] = files

    return patterns


def prepare_functions(
    patterns: Union[List[str], Dict[str, List[str]]],
    func: ProcessingFunc,
    group_by: GroupBy = None
) -> Tuple[Dict[str, List[str]], Dict[str, ProcessingFunc]]:
    """
    Prepare patterns and processing functions for processing.

    This function handles two main tasks:
    1. Ensuring patterns are in a component-keyed dictionary format
    2. Determining which processing functions to use for each component

    Args:
        patterns: Patterns to process, either as a flat list or grouped by component
        func: Processing functions to apply (callable, list, dict)
        group_by: Component name for grouping

    Returns:
        tuple: (grouped_patterns, component_to_funcs)
    """
    # Fast path: If both patterns and func are dictionaries with matching keys,
    # they're already properly structured, so return them as is
    if (isinstance(patterns, dict) and isinstance(func, dict) and
            set(patterns.keys()).issubset(set(func.keys()))):
        return patterns, func

    # Ensure patterns are in a dictionary format
    # If already a dict, use as is; otherwise wrap the list in a dictionary
    component = group_by or 'default'
    grouped_patterns = patterns if isinstance(patterns, dict) else {component: patterns}

    # Determine which processing functions to use for each component
    component_to_funcs = {}

    for comp_value in grouped_patterns.keys():
        # Get functions for this component
        component_to_funcs[comp_value] = get_processing_function(func, comp_value)

    return grouped_patterns, component_to_funcs


def get_processing_function(
    func: ProcessingFunc,
    component: str = None
) -> Optional[ProcessingFunc]:
    """
    Get processing function for a component.

    Args:
        func: Processing functions (callable, list, or dict)
        component: Optional component to get specific function for

    Returns:
        Processing function or None if no function is defined
    """
    if func is None:
        return None

    if callable(func) or isinstance(func, list):
        # If func is a callable or list of functions, apply to all components
        return func
    elif isinstance(func, dict) and component is not None and component in func:
        # If func is a dict, get function for the specified component
        return func[component]
    else:
        return None


def process_patterns(
    patterns: List[str],
    func: ProcessingFunc,
    group_by: GroupBy = None,
    processing_args: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process patterns with the given function.

    This function applies the processing function to the patterns,
    optionally grouping them by the specified dimension.

    Args:
        patterns: The file patterns to process
        func: The processing function(s) to apply
        group_by: How to group patterns for processing
        processing_args: Additional arguments to pass to the processing function

    Returns:
        The processing results
    """
    processing_args = processing_args or {}

    # Prepare patterns and functions
    grouped_patterns, component_to_funcs = prepare_functions(patterns, func, group_by)

    # Process each group
    results = {}

    for group_key, group_patterns in grouped_patterns.items():
        # Handle different function types
        if isinstance(func, dict):
            # Function mapping (e.g., by channel)
            if group_by == 'channel' and group_key in func:
                # Direct mapping for this channel
                group_func = func[group_key]
                if isinstance(group_func, list):
                    # Apply multiple functions in sequence
                    result = group_patterns
                    for f in group_func:
                        result = f(result, **processing_args)
                    results[group_key] = result
                else:
                    # Apply single function
                    results[group_key] = group_func(group_patterns, **processing_args)
            else:
                # Apply different functions to different channels within this group
                group_results = {}
                for channel, channel_func in func.items():
                    channel_patterns = [p for p in group_patterns if channel in p]
                    if channel_patterns:
                        if isinstance(channel_func, list):
                            # Apply multiple functions in sequence
                            result = channel_patterns
                            for f in channel_func:
                                result = f(result, **processing_args)
                            group_results[channel] = result
                        else:
                            # Apply single function
                            group_results[channel] = channel_func(channel_patterns, **processing_args)
                results[group_key] = group_results
        elif isinstance(func, list):
            # Apply multiple functions in sequence
            result = group_patterns
            for f in func:
                result = f(result, **processing_args)
            results[group_key] = result
        else:
            # Apply single function
            results[group_key] = func(group_patterns, **processing_args)

    return results


def group_patterns_by(patterns: List[str], group_by: str) -> Dict[str, List[str]]:
    """
    Group patterns by the specified dimension.

    Args:
        patterns: The file patterns to group
        group_by: The dimension to group by

    Returns:
        A dictionary mapping group keys to lists of patterns
    """
    # This is a simplified implementation - in practice, we would use
    # the microscope handler to group patterns

    grouped_patterns = {}

    for pattern in patterns:
        # Extract group key from pattern
        import re

        if group_by == 'channel':
            # Example: extract channel from filename
            group_match = re.search(r'ch(\d+)', pattern, re.IGNORECASE)
            group_key = group_match.group(1) if group_match else "1"
        elif group_by == 'z_index':
            # Example: extract z-index from filename
            group_match = re.search(r'z(\d+)', pattern, re.IGNORECASE)
            group_key = group_match.group(1) if group_match else "0"
        elif group_by == 'site':
            # Example: extract site from filename
            group_match = re.search(r'site(\d+)', pattern, re.IGNORECASE)
            group_key = group_match.group(1) if group_match else "0"
        else:
            # Default: use the whole pattern as the key
            group_key = pattern

        if group_key not in grouped_patterns:
            grouped_patterns[group_key] = []

        grouped_patterns[group_key].append(pattern)

    return grouped_patterns


def save_results(results: Dict[str, Any], output_dir: str, well: str) -> None:
    """
    Save processing results to the output directory.

    Args:
        results: The processing results
        output_dir: The output directory
        well: The well being processed
    """
    # This is a simplified implementation - in practice, we would use
    # the file system manager to save results

    logger.info(f"Saving results for well {well} to {output_dir}")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results (this is just a placeholder)
    # In practice, we would save the actual processed images
    pass


# Functional API

def step(
    func: ProcessingFunc,
    variable_components: VariableComponents = None,
    group_by: GroupBy = None,
    **kwargs
) -> Step:
    """
    Create a processing step.

    This is a convenience function that creates a Step instance.

    Args:
        func: The processing function(s) to apply
        variable_components: Components that vary across files
        group_by: How to group files for processing
        **kwargs: Additional step parameters

    Returns:
        A new Step instance
    """
    return Step(func, variable_components, group_by, **kwargs)


def pipeline(*steps: Step, **kwargs) -> Pipeline:
    """
    Create a processing pipeline.

    This is a convenience function that creates a Pipeline instance.

    Args:
        *steps: The processing steps to include
        **kwargs: Additional pipeline parameters

    Returns:
        A new Pipeline instance
    """
    if len(steps) == 1 and isinstance(steps[0], list):
        # Handle the case where steps are passed as a list
        return Pipeline(steps[0], **kwargs)
    return Pipeline(steps, **kwargs)


# Example usage

if __name__ == "__main__":
    # This is just an example - in practice, we would use the actual
    # ImagePreprocessor and Stitcher classes

    class ImagePreprocessor:
        @staticmethod
        def create_projection(images, **kwargs):
            print(f"Creating projection from {len(images)} images with {kwargs}")
            return images

        @staticmethod
        def create_composite(images, **kwargs):
            print(f"Creating composite from {len(images)} images with {kwargs}")
            return images

    class Stitcher:
        @staticmethod
        def generate_positions(images, **kwargs):
            print(f"Generating positions from {len(images)} images with {kwargs}")
            return images

        @staticmethod
        def assemble_image(images, **kwargs):
            print(f"Assembling image from {len(images)} images with {kwargs}")
            return images

    # Define processing functions
    def dapi_process(images, **kwargs):
        print(f"Processing DAPI channel images: {len(images)} images with {kwargs}")
        return images

    def calcein_process(images, **kwargs):
        print(f"Processing Calcein channel images: {len(images)} images with {kwargs}")
        return images

    # Define paths
    workspace_folder = "/path/to/workspace"
    process_folder = "/path/to/processed"
    positions_folder = "/path/to/positions"
    stitched_folder = "/path/to/stitched"

    # Create reference pipeline
    reference_pipeline = (
        pipeline(
            # Flatten Z-stacks
            step(
                func=ImagePreprocessor.create_projection,
                variable_components=['z_index'],
                processing_args={'method': 'max_projection'},
                name="Z-Stack Flattening"
            ),

            # Process channels
            step(
                func={"1": dapi_process, "2": calcein_process},
                variable_components=['site'],
                group_by='channel',
                name="Channel Processing"
            ),

            # Create composites
            step(
                func=ImagePreprocessor.create_composite,
                variable_components=['channel'],
                group_by='site',
                processing_args={'weights': {"1": 0.7, "2": 0.3}},
                name="Composite Creation"
            )
        )
        .set_input(workspace_folder)
        .set_output(process_folder)
        .set_well_filter(["A01", "B02"])
    )

    # Print the pipeline
    print(reference_pipeline)
