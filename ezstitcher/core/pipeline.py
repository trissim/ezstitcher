"""
Core implementation of the Flexible Pipeline Architecture.

This module provides a flexible, declarative API for defining image processing
pipelines in EZStitcher. It builds on the strengths of the current
process_patterns_with_variable_components method while adding an object-oriented
core with a functional interface.
"""

from typing import Dict, List, Any, Optional, Union, Set, Callable, TypeVar, Generic
import logging
from pathlib import Path

# Import base interface
from .pipeline_base import PipelineInterface

# Import Step classes from steps module
from ezstitcher.core.steps import ImageStitchingStep
from ezstitcher.core.steps import Step, WellFilter
from ezstitcher.core.utils import prepare_patterns_and_functions

# Configure logging
logger = logging.getLogger(__name__)


class StepExecutionPlan:
    """
    Contains execution information for a pipeline step.

    This class holds the input/output directories and other execution
    parameters for a step, allowing for immutable path resolution.

    Attributes:
        step_id: Unique identifier for the step (from id())
        step_name: Human-readable name of the step
        step_type: Type of the step (class name)
        input_dir: Input directory for the step
        output_dir: Output directory for the step
    """

    def __init__(
        self,
        step_id: int,
        step_name: str,
        step_type: str,
        input_dir: Path,
        output_dir: Path
    ):
        """
        Initialize the execution plan.

        Args:
            step_id: Unique identifier for the step
            step_name: Human-readable name of the step
            step_type: Type of the step (class name)
            input_dir: Input directory for the step
            output_dir: Output directory for the step
        """
        self.step_id = step_id
        self.step_name = step_name
        self.step_type = step_type
        self.input_dir = input_dir
        self.output_dir = output_dir


class Pipeline(PipelineInterface):
    """
    A sequence of processing steps.

    A Pipeline is a sequence of processing steps that are executed in order.
    Each step takes input from the previous step's output and produces new output.

    Attributes:
        steps: The sequence of processing steps
        name: Human-readable name for the pipeline
        _config: Configuration parameters
        path_overrides: Dictionary mapping step IDs to input/output directory overrides
    """

    def __init__(
        self,
        steps: List[Step] = None,
        name: str = None,
        input_dir: Union[str, Path, None] = None,
        output_dir: Union[str, Path, None] = None
    ):
        """
        Initialize a pipeline.

        Args:
            steps: The sequence of processing steps
            name: Human-readable name for the pipeline
            input_dir: Input directory for the first step (if not already overridden)
            output_dir: Output directory for the last step (if not already overridden)
        """
        self.steps = []
        self.name = name or f"Pipeline({len(steps or [])} steps)"
        self._config = {}
        self.path_overrides = {}  # Dictionary to store path overrides

        # Add steps if provided
        if steps:
            for step in steps:
                if step is not None:  # Skip None values in steps list
                    self.add_step(step)

        # Apply input_dir to first step and output_dir to last step if they don't have overrides
        if steps and (input_dir is not None or output_dir is not None):
            # Apply input_dir to first step if it doesn't have an override
            if input_dir is not None and self.steps:
                first_step = self.steps[0]
                first_step_id = id(first_step)
                input_key = f"{first_step_id}_input_dir"

                # Convert to Path if it's a string
                if isinstance(input_dir, str):
                    input_dir = Path(input_dir)

                # Apply only if the step doesn't already have an input_dir override
                if input_key not in self.path_overrides:
                    self.path_overrides[input_key] = input_dir

            # Apply output_dir to last step if it doesn't have an override
            if output_dir is not None and self.steps:
                last_step = self.steps[-1]
                last_step_id = id(last_step)
                output_key = f"{last_step_id}_output_dir"

                # Convert to Path if it's a string
                if isinstance(output_dir, str):
                    output_dir = Path(output_dir)

                # Apply only if the step doesn't already have an output_dir override
                if output_key not in self.path_overrides:
                    self.path_overrides[output_key] = output_dir


    def add_step(self, step: Step) -> 'Pipeline':
        """
        Add a step to the pipeline.

        If the step has _ephemeral_init_kwargs containing input_dir or output_dir,
        they will be extracted and stored in the pipeline's path_overrides dictionary,
        then removed from the step instance to avoid polluting it.

        Args:
            step: Step to add

        Returns:
            Self for method chaining
        """
        step_id = id(step)

        # Priority 1: ephemeral init kwargs
        for source in ("_ephemeral_init_kwargs", "__dict__"):
            kw = getattr(step, source, {})
            for key in ("input_dir", "output_dir"):
                if key in kw:
                    value = kw[key]
                    if isinstance(value, str):
                        value = Path(value)
                    override_key = f"{step_id}_{key}"
                    if override_key not in self.path_overrides:
                        self.path_overrides[override_key] = value
                    # Remove from step state to enforce statelessness
                    if hasattr(step, key):
                        delattr(step, key)
            if source == "_ephemeral_init_kwargs" and hasattr(step, "_ephemeral_init_kwargs"):
                delattr(step, "_ephemeral_init_kwargs")

        self.steps.append(step)
        return self

    def run(
        self,
        context: 'ProcessingContext'
    ) -> 'ProcessingContext':
        """
        Execute the pipeline.

        Args:
            context: The processing context containing pre-computed paths and other state

        Returns:
            The updated processing context with all results

        Raises:
            ValueError: If context is not properly initialized
        """
        logger.info("Running pipeline: %s", self.name)

        if not context.orchestrator:
            raise ValueError("context.orchestrator must be specified")

        # Log the resolved paths
        for step in self.steps:
            input_dir = context.get_step_input_dir(step)
            output_dir = context.get_step_output_dir(step)
            if not input_dir or not output_dir:
                raise ValueError(f"No paths resolved for step: {step.name}")
            logger.info(f"Step '{step.name}' paths: input_dir={input_dir}, output_dir={output_dir}")

        logger.info("Well filter: %s", context.well_filter)

        # Execute each step
        for i, step in enumerate(self.steps):
            logger.info("Executing step %d/%d: %s", i+1, len(self.steps), step)
            context = step.process(context)

        logger.info("Pipeline completed: %s", self.name)
        return context



    def collect_unique_dirs(self) -> set:
        """
        Collects all unique directory paths from all steps in the pipeline.

        Iterates through each step's attributes and collects values for attributes
        with "dir" in their name. Also includes paths from path_overrides.

        Returns:
            A set of unique directory paths.
        """
        unique_dirs = set()

        # Collect paths from step attributes (legacy support)
        for step in self.steps:
            for attr_name, attr_value in step.__dict__.items():
                if "dir" in attr_name.lower() and attr_value:
                    unique_dirs.add(attr_value)

        # Collect paths from path_overrides
        for key, value in self.path_overrides.items():
            if "dir" in key.lower() and value:
                unique_dirs.add(value)

        return unique_dirs

    def __repr__(self) -> str:
        """
        String representation of the pipeline.

        Returns:
            A human-readable representation of the pipeline
        """
        steps_repr = "\n  ".join(repr(step) for step in self.steps)
        return (f"{self.name}\n"
                f"  Steps:\n  {steps_repr}")


class ProcessingContext:
    """
    Maintains state during pipeline execution.

    The ProcessingContext is the canonical owner of all state during pipeline execution.
    Steps should use only context attributes and must not modify context fields
    except for accumulating results.

    Attributes:
        well_filter: Wells to process (should not be modified by steps)
        orchestrator: Reference to the pipeline orchestrator
        config: Configuration parameters
        results: Processing results
        step_plans: Dictionary mapping step IDs to StepExecutionPlan objects
    """

    def __init__(
        self,
        well_filter: WellFilter = None,
        config: Dict[str, Any] = None,
        orchestrator = None,
        **kwargs
    ):
        """
        Initialize the processing context.

        Args:
            well_filter: Wells to process
            config: Configuration parameters
            orchestrator: Reference to the pipeline orchestrator
            **kwargs: Additional context attributes
        """
        self.well_filter = well_filter
        self.config = config or {}
        self.orchestrator = orchestrator
        self.results = {}

        # Dictionary mapping step IDs to StepExecutionPlan objects
        self.step_plans = {}

        # Add any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_step_plan(self, step, plan):
        """
        Add an execution plan for a step.

        Args:
            step: The step to add a plan for
            plan: The StepExecutionPlan for the step
        """
        self.step_plans[id(step)] = plan

    def get_step_input_dir(self, step):
        """
        Get input directory for a step.

        Args:
            step: The step to get the input directory for

        Returns:
            Path: The input directory for the step
        """
        plan = self.step_plans.get(id(step))
        if plan:
            return plan.input_dir
        return None

    def get_step_output_dir(self, step):
        """
        Get output directory for a step.

        Args:
            step: The step to get the output directory for

        Returns:
            Path: The output directory for the step
        """
        plan = self.step_plans.get(id(step))
        if plan:
            return plan.output_dir
        return None







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
