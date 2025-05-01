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
import numpy as np # Added for type checking

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
            step_id = id(step) # Get step ID for context/logging
            logger.info("Executing step %d/%d: %s (ID: %s)", i+1, len(self.steps), step.name, step_id)

            # Assuming step.process returns a dictionary of its outputs
            step_outputs = step.process(context)

            if not isinstance(step_outputs, dict):
                 logger.warning(f"Step '{step.name}' process method did not return a dictionary. Output type: {type(step_outputs)}. Skipping results merge and storage adapter write for this step.")
                 # Decide how to handle this - potentially update context based on return type?
                 # For now, we assume the step might have modified context directly if it didn't return dict.
                 # context = step_outputs if isinstance(step_outputs, ProcessingContext) else context # Example handling
                 continue # Skip merge and storage write if output is not a dict

            # Merge step outputs into context results
            context.results.update(step_outputs)
            logger.debug(f"Step '{step.name}' produced outputs: {list(step_outputs.keys())}")

            # --- Write outputs to Storage Adapter ---
            # Import here to avoid circular imports
            from ezstitcher.io.storage_adapter import write_result

            pipeline_id = self.name # Use pipeline name as part of the key
            step_name_safe = step.name.replace(" ", "_").lower() # Make step name safe for keys

            for output_name, output_data in step_outputs.items():
                if isinstance(output_data, np.ndarray): # Only store numpy arrays
                    # Construct a unique key
                    storage_key = f"{pipeline_id}_{step_name_safe}_{i}_{output_name}"

                    # Use the helper function to write the result
                    success = write_result(
                        context=context,
                        key=storage_key,
                        data=output_data,
                        fallback_path=None  # No fallback path for pipeline outputs
                    )

                    if success:
                        logger.debug("Successfully stored output '%s' with key '%s'",
                                    output_name, storage_key)
                else:
                    logger.debug("Skipping non-numpy output '%s' from step '%s'. Type: %s",
                                output_name, step.name, type(output_data).__name__)
            # -----------------------------------------

        logger.info("Pipeline completed: %s", self.name)
        return context

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

    @property
    def storage_mode(self) -> str:
        """
        Get the storage mode from the orchestrator.

        Returns:
            The storage mode ("legacy", "memory", "zarr")
        """
        if not hasattr(self, 'orchestrator') or self.orchestrator is None:
            logger.debug("No orchestrator available, defaulting to 'legacy' storage mode")
            return "legacy"

        mode = getattr(self.orchestrator, 'storage_mode', "legacy")
        logger.debug("Resolved storage_mode from context: %s", mode)
        return mode

    def is_legacy_mode(self) -> bool:
        """
        Check if the storage mode is legacy.

        Returns:
            True if storage_mode is "legacy", False otherwise
        """
        return self.storage_mode == "legacy"

    def store_array(self, key: str, data: np.ndarray, fallback_path: Optional[Path] = None) -> bool:
        """
        Store a numpy array using the storage adapter if available.

        Args:
            key: The storage key
            data: The numpy array to store
            fallback_path: Optional path to save to if storage adapter is not available

        Returns:
            True if stored successfully, False if no storage adapter is available or on error
        """
        # Import here to avoid circular imports
        from ezstitcher.io.storage_adapter import write_result

        # Use the helper function to write the result
        return write_result(
            context=self,
            key=key,
            data=data,
            fallback_path=fallback_path
        )

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
