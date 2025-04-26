"""
Abstract base class for steps in the EZStitcher pipeline architecture.

This module defines the AbstractStep interface that all steps must implement,
ensuring a consistent interface across different step implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AbstractStep(ABC):
    """
    Abstract base class defining the interface for all steps in EZStitcher.

    This class defines the minimal contract that all step implementations must follow,
    ensuring consistent behavior across different step types.
    """

    @abstractmethod
    def process(self, group: List[Any], context: Optional[Dict[str, Any]] = None) -> Any:
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

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this step."""
        pass

    @property
    @abstractmethod
    def input_dir(self) -> str:
        """The input directory for this step."""
        pass

    @property
    @abstractmethod
    def output_dir(self) -> str:
        """The output directory for this step."""
        pass
