"""
Core implementation of the Flexible Pipeline Architecture.

This module provides a flexible, declarative API for defining image processing
pipelines in EZStitcher. It builds on the strengths of the current
process_patterns_with_variable_components method while adding an object-oriented
core with a functional interface.
"""

from typing import Dict, List, Any
import logging

# Import Step classes from steps module
from ezstitcher.core.steps import Step, WellFilter
from ezstitcher.core.utils import prepare_patterns_and_functions

# Configure logging
logger = logging.getLogger(__name__)


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

    def run(
        self,
        input_dir: str = None,
        output_dir: str = None,
        well_filter: WellFilter = None,
        microscope_handler = None,
        orchestrator = None,
        positions_file = None
    ) -> Dict[str, Any]:
        """
        Execute the pipeline.

        Args:
            input_dir: Optional input directory override
            output_dir: Optional output directory override
            well_filter: Optional well filter override
            microscope_handler: Optional microscope handler override
            orchestrator: Optional PipelineOrchestrator instance
            positions_file: Optional positions file to use for stitching

        Returns:
            The results of the pipeline execution

        Raises:
            ValueError: If no input directory is specified
        """
        logger.info("Running pipeline: %s", self.name)

        self.orchestrator = orchestrator
        self.microscope_handler = self.orchestrator.microscope_handler
        if orchestrator is None:
            raise ValueError("orchestrator must be specified")
        effective_input = input_dir or self.input_dir
        effective_output = output_dir or self.output_dir
        effective_well_filter = well_filter or self.well_filter

        # If input_dir is still not set, try to get it from the first step
        if not effective_input and self.steps:
            effective_input = self.steps[0].input_dir

        if not effective_input:
            raise ValueError("Input directory must be specified")

        logger.info("Input directory: %s", effective_input)
        logger.info("Output directory: %s", effective_output)
        logger.info("Well filter: %s", effective_well_filter)

        # Initialize context
        context = ProcessingContext(
            input_dir=effective_input,
            output_dir=effective_output,
            well_filter=effective_well_filter,
            orchestrator=orchestrator,
        )

        # Execute each step
        for i, step in enumerate(self.steps):
            logger.info("Executing step %d/%d: %s", i+1, len(self.steps), step)
            context = step.process(context)

        logger.info("Pipeline completed: %s", self.name)
        return context.results

    def collect_unique_dirs(self) -> set:
        """
        Collects all unique directory paths from all steps in the pipeline.

        Iterates through each step's attributes and collects values for attributes
        with "dir" in their name.

        Returns:
            A set of unique directory paths.
        """
        unique_dirs = set()
        for step in self.steps:
            for attr_name, attr_value in step.__dict__.items():
                if "dir" in attr_name.lower() and attr_value:
                    unique_dirs.add(attr_value)
        return unique_dirs

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
