"""
Abstract base class for steps in the EZStitcher pipeline architecture.

This module defines the AbstractStep interface that all steps must implement,
ensuring a consistent interface across different step implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, TYPE_CHECKING
from .pipeline_base import StepInterface

# ProcessingContext and StepResult are defined in pipeline.py
# Using string literal for type annotations to avoid circular imports
if TYPE_CHECKING:
    from .pipeline import ProcessingContext, StepResult


class AbstractStep(StepInterface):
    """
    Abstract base class defining the interface for all steps in EZStitcher.

    This class defines the minimal contract that all step implementations must follow,
    ensuring consistent behavior across different step types.

    Step Architecture Notes:
    - Steps must be stateless and should NOT modify the context directly
    - Steps must return a StepResult object containing:
      - Normal processing results
      - Requested context updates
      - Requested storage operations
    - Pipeline.run() is responsible for applying these changes
    """

    @abstractmethod
    def process(self, context: 'ProcessingContext') -> 'StepResult':
        """
        Process data according to the step's functionality.

        This is the core method that all steps must implement. It processes
        data according to the step's functionality and returns a structured result.

        Args:
            context: Processing context containing input data and metadata (read-only)

        Returns:
            StepResult object containing processing results, context updates, and storage operations.
            Steps should NOT modify the context directly.
        """
        pass
