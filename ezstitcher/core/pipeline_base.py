"""
Core interfaces and types for the EZStitcher pipeline system.

This module provides the foundational abstractions that define the pipeline architecture.
It contains no implementation details, only interfaces and type definitions.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Any, TYPE_CHECKING
from pathlib import Path

# Use string literal for type annotations to avoid circular imports
# The actual ProcessingContext class is defined in pipeline.py

class StepInterface(ABC):
    """Base interface for all pipeline steps."""

    @abstractmethod
    def __init__(
        self,
        name: Optional[str] = None,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        variable_components: Optional[List[str]] = None
    ) -> None:
        """
        Initialize a pipeline step.

        Args:
            name: Human-readable name for the step
            input_dir: Input directory for this step
            output_dir: Output directory for this step
            variable_components: List of components that vary in processing
        """
        pass

    @abstractmethod
    def process(self, context: 'ProcessingContext') -> 'ProcessingContext':
        """
        Process data according to the step's functionality.

        Args:
            context: Processing context containing input data and metadata

        Returns:
            Updated context with processing results
        """
        pass

class PipelineInterface(ABC):
    """Base interface for pipeline implementations."""

    @abstractmethod
    def __init__(
        self,
        steps: Optional[List[StepInterface]] = None,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
        well_filter: Optional[List[str]] = None
    ) -> None:
        """
        Initialize a pipeline.

        Args:
            steps: List of processing steps
            input_dir: Pipeline input directory
            output_dir: Pipeline output directory
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
        orchestrator: Optional[Any] = None,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        well_filter: Optional[List[str]] = None
    ) -> 'ProcessingContext':
        """
        Execute the pipeline.

        Args:
            orchestrator: Pipeline orchestrator instance
            input_dir: Override input directory
            output_dir: Override output directory
            well_filter: Override well filter

        Returns:
            Processing results in context
        """
        pass

class PipelineFactoryInterface(ABC):
    """Base interface for pipeline factories."""

    @abstractmethod
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        normalize: bool = True,
        normalization_params: Optional[Dict[str, Any]] = None,
        preprocessing_steps: Optional[List[StepInterface]] = None,
        well_filter: Optional[List[str]] = None
    ) -> None:
        """
        Initialize a pipeline factory.

        Args:
            input_dir: Input directory for created pipelines
            output_dir: Output directory for created pipelines
            normalize: Whether to apply normalization
            normalization_params: Parameters for normalization
            preprocessing_steps: Steps to add before main processing
            well_filter: Wells to process
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
