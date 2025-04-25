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

        This method ensures a coherent data flow through the pipeline:
        - Each step's output directory must match the next step's input directory
        - If a step specifies an input directory, the previous step's output is set to match
        - If a step specifies an output directory, it's used for that step only
        - The last step uses the pipeline's output directory (if specified)

        Args:
            step: The step to add
            output_dir: Optional output directory for the step

        Returns:
            Self, for method chaining
        """
        # Override step's output_dir if explicitly provided to this method
        if output_dir:
            step.output_dir = output_dir

        # Handle input directory and ensure coherent flow with previous step
        self._ensure_coherent_input_directory(step)

        # Set output directory if not specified
        self._set_step_output_directory(step)

        # Update pipeline directories if needed
        self._update_pipeline_directories(step)

        # Add the step to the pipeline
        self.steps.append(step)
        return self

    def _ensure_coherent_input_directory(self, step: Step):
        """
        Ensure the step's input directory is coherent with the appropriate previous step's output.
        For ImageStitchingStep, use the pipeline's input directory by default.
        """
        logger.debug(f"Resolving input directory for step: {step.name}")

        if not self.steps:  # First step
            # If no input directory specified, use pipeline's input directory
            if not step.input_dir:
                step.input_dir = self.input_dir
                logger.debug(f"First step using pipeline input_dir: {step.input_dir}")
            return

        # No special handling for ImageStitchingStep - it will use the normal directory resolution logic

        # Normal directory resolution for other steps
        prev_step = self.steps[-1]

        if step.input_dir:
            # This step has a specified input directory
            # Update previous step's output to match this step's input
            prev_step.output_dir = step.input_dir
            logger.debug(f"Updated previous step output_dir to match current step input_dir: {step.input_dir}")
        else:
            # No input directory specified, use previous step's output
            if not prev_step.output_dir:
                # If previous step has no output directory, use a default
                prev_step.output_dir = self.output_dir or prev_step.input_dir
                logger.debug("Set previous step output_dir to default: %s", prev_step.output_dir)

            # Set this step's input to previous step's output
            step.input_dir = prev_step.output_dir
            logger.debug("Set current step input_dir to previous step output_dir: %s", step.input_dir)

    def _find_last_processing_step(self):
        """Find the last processing step in the pipeline (excluding PositionGenerationStep)."""
        for s in reversed(self.steps):
            # Skip PositionGenerationStep as it's not an image processing step
            if s.__class__.__name__ != "PositionGenerationStep":
                return s
        return None

    def _check_directory_conflicts(self, step, proposed_dir):
        """
        Check if the proposed directory conflicts with existing directories.

        Args:
            step: The step being configured
            proposed_dir: The proposed output directory

        Returns:
            bool: True if there's a conflict, False otherwise
        """
        # Check for conflict with last processing step
        last_processing_step = self._find_last_processing_step()

        if last_processing_step and last_processing_step.output_dir == proposed_dir:
            logger.info("Avoiding output directory conflict with last processing step.")
            return True

        # Check for conflict with input directory
        if step.input_dir == proposed_dir:
            logger.info("Avoiding output directory conflict with input directory.")
            return True

        # Check for conflicts with previous steps
        for prev_step in self.steps:
            if prev_step.output_dir and Path(prev_step.output_dir) == proposed_dir:
                logger.info("Avoiding output directory conflict with previous step.")
                return True

        return False

    def _set_stitching_step_output_directory(self, step):
        """Set output directory for ImageStitchingStep."""
        # Get the stitched_dir_suffix from the orchestrator's config if available
        stitched_suffix = "_stitched"  # Default suffix
        if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'config'):
            stitched_suffix = self.orchestrator.config.stitched_dir_suffix

        if self.output_dir:
            # Use pipeline's output_dir if specified
            step.output_dir = self.output_dir

            # Check for conflicts
            if self._check_directory_conflicts(step, step.output_dir):
                # If there's a conflict, create a unique directory with the configured stitched suffix
                workspace_dir = Path(self.input_dir)
                step.output_dir = workspace_dir.parent / f"{workspace_dir.name}{stitched_suffix}"
                logger.info("Using alternative directory: %s", step.output_dir)
        else:
            # Find the appropriate base directory
            if self.steps:
                # Get the workspace directory (parent of the first step's input)
                workspace_dir = Path(self.input_dir)
                step.output_dir = workspace_dir.parent / f"{workspace_dir.name}{stitched_suffix}"
            else:
                # For standalone stitching step, use input directory as base
                input_path = Path(step.input_dir)
                step.output_dir = input_path.parent / f"{input_path.name}{stitched_suffix}"

            # Check for conflicts
            if self._check_directory_conflicts(step, step.output_dir):
                # If there's a conflict, create a unique directory with _stitched_final suffix
                workspace_dir = Path(self.input_dir)
                step.output_dir = workspace_dir.parent / f"{workspace_dir.name}{stitched_suffix}_final"
                logger.info("Using alternative directory: %s", step.output_dir)

            logger.info("ImageStitchingStep has no output directory specified. "
                       "Using default stitched directory: %s", step.output_dir)

    def _set_step_output_directory(self, step: Step):
        """Set the step's output directory if not already specified."""
        if step.output_dir:
            return  # Output directory already specified

        # Get directory suffixes from orchestrator's config if available
        out_suffix = "_out"  # Default suffix for regular steps
        processed_suffix = "_processed"  # Default suffix for intermediate steps
        positions_suffix = "_positions"  # Default suffix for position generation steps

        # Try to get suffixes from orchestrator config
        if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'config'):
            config = self.orchestrator.config
            out_suffix = config.out_dir_suffix
            processed_suffix = config.processed_dir_suffix
            positions_suffix = config.positions_dir_suffix

        # Check if this is a stitching step
        is_stitching = step.__class__.__name__ == "ImageStitchingStep"

        # Check if this is a position generation step
        is_position_generation = step.__class__.__name__ == "PositionGenerationStep"

        # Check if pipeline has a stitching step (including this one)
        has_stitching_step = is_stitching or any(
            s.__class__.__name__ == "ImageStitchingStep" for s in self.steps)

        # Special handling for ImageStitchingStep
        if is_stitching:
            self._set_stitching_step_output_directory(step)
            return

        # Special handling for PositionGenerationStep
        if is_position_generation:
            # Use positions suffix for position generation steps
            input_path = Path(step.input_dir)
            step.output_dir = input_path.parent / f"{input_path.name}{positions_suffix}"
            logger.info("PositionGenerationStep using default directory: %s", step.output_dir)
            return

        # Get input path for non-stitching steps
        input_path = Path(step.input_dir)

        # Determine default output directory based on step type and pipeline context
        if self.output_dir and has_stitching_step:
            # For processing steps in a pipeline with stitching and output_dir,
            # create intermediate directories
            suffix = processed_suffix
            log_msg = "Processing step in pipeline with stitching step."
        elif not self.steps and self.output_dir:
            # For first step with pipeline output specified
            step.output_dir = self.output_dir
            return
        elif not self.steps:
            # For first step with no output specified
            suffix = out_suffix
            log_msg = "First step has no output directory specified."
        else:
            # For other steps, use pipeline's output_dir or step's input_dir
            step.output_dir = self.output_dir or step.input_dir
            return

        # Create default directory without checking for conflicts
        # This aligns with the default behavior of in-place processing
        default_dir = input_path.parent / f"{input_path.name}{suffix}"

        # Set the output directory
        step.output_dir = default_dir

        # Log the decision
        logger.info("%s Using default directory: %s", log_msg, step.output_dir)

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
