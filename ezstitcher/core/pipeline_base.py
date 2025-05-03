"""
Core interfaces and types for the EZStitcher pipeline system.

This module provides the foundational abstractions that define the pipeline architecture.
It contains no implementation details, only interfaces and type definitions.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Any, TYPE_CHECKING
from pathlib import Path

# Forward reference for StepResult
if TYPE_CHECKING:
    from .pipeline import StepResult

# Use string literal for type annotations to avoid circular imports
# The actual ProcessingContext class is defined in pipeline.py

class StepInterface(ABC):
    """Base interface for all pipeline steps."""

    @abstractmethod
    def __init__(
        self,
        name: Optional[str] = None,
        variable_components: Optional[List[str]] = None
    ) -> None:
        """
        Initialize a pipeline step.

        Args:
            name: Human-readable name for the step
            variable_components: List of components that vary in processing
        """
        pass

    @abstractmethod
    def process(self, context: 'ProcessingContext') -> 'StepResult':
        """
        Process data according to the step's functionality.

        Args:
            context: Processing context containing input data and metadata (read-only)

        Returns:
            StepResult object containing processing results, context updates, and storage operations.
            Steps should NOT modify the context directly.
        """
        pass

class PipelineInterface(ABC):
    """Base interface for pipeline implementations."""

    @abstractmethod
    def __init__(
        self,
        steps: Optional[List[StepInterface]] = None,
        name: Optional[str] = None,
        well_filter: Optional[List[str]] = None
    ) -> None:
        """
        Initialize a pipeline.

        Args:
            steps: List of processing steps
            name: Human-readable name
            well_filter: List of wells to process
        """
        pass

    @abstractmethod
    def add_step(self, step: StepInterface) -> 'PipelineInterface':
        """
        Add a processing step to the pipeline.

        Args:
            step: Step to add

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
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
        """
        pass

class PipelineFactoryInterface(ABC):
    """Base interface for pipeline factories."""

    @abstractmethod
    def __init__(
        self,
        normalize: bool = True,
        normalization_params: Optional[Dict[str, Any]] = None,
        preprocessing_steps: Optional[List[StepInterface]] = None,
        well_filter: Optional[List[str]] = None,
        path_overrides: Optional[Dict[str, Union[str, Path]]] = None
    ) -> None:
        """
        Initialize a pipeline factory.

        Args:
            normalize: Whether to apply normalization
            normalization_params: Parameters for normalization
            preprocessing_steps: Steps to add before main processing
            well_filter: Wells to process
            path_overrides: Optional dictionary of path overrides
        """
        pass

    @abstractmethod
    def create_pipelines(self) -> List[PipelineInterface]:
        """
        Create configured pipelines.

        Returns:
            List of configured pipeline instances
        """
        pass

# Common enums and constants
class ProcessingMode:
    """Available processing modes."""
    BASIC = "basic"
    MULTICHANNEL = "multichannel"
    ZSTACK = "zstack"
    FOCUS = "focus"

class StepType:
    """Common step types."""
    PREPROCESSING = "preprocessing"
    POSITION_GENERATION = "position_generation"
    STITCHING = "stitching"
    COMPOSITE = "composite"
