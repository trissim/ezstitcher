"""
Abstract base class for steps in the EZStitcher pipeline architecture.

This module defines the AbstractStep interface that all steps must implement,
ensuring a consistent interface across different step implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict
from .pipeline_base import StepInterface

# ProcessingContext is defined in pipeline.py
# Using string literal for type annotations to avoid circular imports


class AbstractStep(StepInterface):
    """
    Abstract base class defining the interface for all steps in EZStitcher.

    This class defines the minimal contract that all step implementations must follow,
    ensuring consistent behavior across different step types.
    """

    @abstractmethod
    def process(self, group: List[Any], context: Optional['ProcessingContext'] = None) -> Any:
        """
        Process a group of images.

        This is the core method that all steps must implement. It takes a group
        of images and processes them according to the step's functionality.

        Args:
            group: Group of images to process
            context: Pipeline context for sharing data between steps

        Returns:
            Processed result (typically an image or list of images)
        """
        pass
