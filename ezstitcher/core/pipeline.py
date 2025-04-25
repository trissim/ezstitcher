"""
Core implementation of the Flexible Pipeline Architecture.

This module provides a flexible, declarative API for defining image processing
pipelines in EZStitcher. It builds on the strengths of the current
process_patterns_with_variable_components method while adding an object-oriented
core with a functional interface.
"""

from typing import Dict, List, Any
import logging
from pathlib import Path

# Import Step classes from steps module
from ezstitcher.core.steps import ImageStitchingStep
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
        Add a step to the pipeline with improved directory resolution.
        
        Directory resolution follows these rules:
        1. Input directory is resolved first based on previous step
        2. Output directory is set based on input directory
        3. Explicit output_dir overrides automatic resolution
        """
        # First ensure input directory is coherent
        self._ensure_coherent_input_directory(step)
        
        # Set output directory if not explicitly provided
        if not output_dir and not step.output_dir:
            self._set_step_output_directory(step)
        elif output_dir:
            step.output_dir = output_dir
        
        # Add step and update pipeline directories
        self.steps.append(step)
        self._update_pipeline_directories(step)
        return self

    def _ensure_coherent_input_directory(self, step: Step):
        """Ensure step's input directory is coherent with pipeline flow."""
        if not self.steps:  # First step
            if not step.input_dir:
                step.input_dir = self.input_dir
            return

        prev_step = self.steps[-1]
        
        # If no input specified, use previous step's output
        if not step.input_dir:
            step.input_dir = prev_step.output_dir or prev_step.input_dir

    def _check_directory_conflicts(self, step: Step, proposed_dir: Path) -> bool:
        """
        Check for directory conflicts in pipeline.
        
        Args:
            step: Step being configured
            proposed_dir: Proposed output directory

        Returns:
            bool: True if conflict exists
        """
        proposed_dir = Path(proposed_dir)
        last_processing = next((s for s in reversed(self.steps) 
                              if s.__class__.__name__ != "PositionGenerationStep"), None)
        return (last_processing and Path(last_processing.output_dir) == proposed_dir) or \
               (step.input_dir and Path(step.input_dir) == proposed_dir)

    def _set_stitching_step_output_directory(self, step):
        """Set output directory for ImageStitchingStep."""
        stitched_suffix = getattr(self.orchestrator.config, 'stitched_dir_suffix', '_stitched') if hasattr(self, 'orchestrator') else '_stitched'
        
        last_processing = next((s for s in reversed(self.steps) 
                              if s.__class__.__name__ != "PositionGenerationStep"), None)
        base_dir = Path(last_processing.output_dir if last_processing else self.input_dir)
        
        step.output_dir = base_dir.parent / f"{base_dir.name}{stitched_suffix}"
        
        if self._check_directory_conflicts(step, step.output_dir):
            step.output_dir = base_dir.parent / f"{base_dir.name}{stitched_suffix}_final"

    def _set_step_output_directory(self, step: Step):
        """Set the step's output directory if not already specified."""
        if step.output_dir:
            return  # Output directory already specified

        # Get directory suffixes from orchestrator's config if available
        out_suffix = "_out"  # Default suffix for all processing steps
        positions_suffix = "_positions"  # Default suffix for position generation steps

        # Try to get suffixes from orchestrator config
        if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'config'):
            config = self.orchestrator.config
            out_suffix = config.out_dir_suffix
            positions_suffix = config.positions_dir_suffix

        # Check if this is a stitching step
        is_stitching = step.__class__.__name__ == "ImageStitchingStep"

        # Check if this is a position generation step
        is_position_generation = step.__class__.__name__ == "PositionGenerationStep"

        # Special handling for ImageStitchingStep
        if is_stitching:
            self._set_stitching_step_output_directory(step)
            return

        # Special handling for PositionGenerationStep
        if is_position_generation:
            input_path = Path(step.input_dir)
            step.output_dir = input_path.parent / f"{input_path.name}{positions_suffix}"
            logger.info("PositionGenerationStep using default directory: %s", step.output_dir)
            return

        # For all other processing steps
        if not self.output_dir:
            input_path = Path(step.input_dir)
            step.output_dir = input_path.parent / f"{input_path.name}{out_suffix}"
            logger.info("Processing step using default directory: %s", step.output_dir)
        else:
            step.output_dir = self.output_dir

    def _update_pipeline_directories(self, step: Step):
        """Update pipeline directories based on the step if needed."""
        # If this is the first step and pipeline's input_dir is not set, use step's input_dir
        if not self.steps and not self.input_dir and step.input_dir:
            self.input_dir = step.input_dir

        # If pipeline's output_dir is not set, use the step's output_dir
        if not self.output_dir and step.output_dir:
            self.output_dir = step.output_dir

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
