"""
Materialization resolver for EZStitcher.

This module provides a resolver for determining if a step needs materialization,
separating the decision logic from the step class to maintain the stateless
contract of steps.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MaterializationResolver:
    """
    Resolver for materialization decisions.

    This class is responsible for determining if a step needs materialization
    based on the step's flags and the orchestration context, keeping the
    decision logic separate from the step class.
    """

    @staticmethod
    def needs_materialization(
        step: Any,
        manager: Any,
        context: Optional[Any] = None,
        pipeline: Optional[Any] = None
    ) -> bool:
        """
        Determine if a step needs materialization.

        Args:
            step: The step to check
            manager: The materialization manager
            context: The processing context (optional)
            pipeline: The pipeline containing the step (optional)

        Returns:
            True if materialization is needed, False otherwise
        """
        # If no manager is provided, materialization is not needed
        if not manager:
            return False

        # Update the context in the materialization manager if provided
        if context:
            manager.context = context

        # Delegate to the materialization manager to determine if materialization is needed
        return manager.needs_materialization(step, pipeline=pipeline)
