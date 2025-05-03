"""
Materialization manager for EZStitcher.

This module provides a centralized manager for materialization operations,
ensuring that data is materialized to disk only when needed and in a consistent way.
"""

import logging
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Literal, Any

import numpy as np

from ezstitcher.io.overlay import OverlayMode, OverlayOperation
from ezstitcher.io.storage_config import StorageConfig
from ezstitcher.core.pattern_resolver import get_patterns_for_well
from ezstitcher.materialization.flag_engine import FlagInferenceEngine

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Modes for handling materialization failures."""
    FAIL_FAST = auto()  # Raise an exception on the first failure
    LOG_AND_CONTINUE = auto()  # Log errors and continue
    FALLBACK_TO_DISK = auto()  # Try to find files on disk if materialization fails


class MaterializationPolicy:
    """
    Configuration for materialization behavior.

    This class defines how materialization should behave in different contexts.
    """

    def __init__(
        self,
        respect_flags: bool = True,
        force_memory: bool = False,
        force_disk: bool = False,
        lazy_cleanup: bool = False,
        failure_mode: FailureMode = FailureMode.LOG_AND_CONTINUE
    ):
        """
        Initialize a materialization policy.

        Args:
            respect_flags: Whether to respect the requires_fs_* flags
            force_memory: Whether to force in-memory operation (overrides flags)
            force_disk: Whether to force disk materialization (overrides flags)
            lazy_cleanup: Whether to delay cleanup until the end of the pipeline
            failure_mode: How to handle materialization failures
        """
        self.respect_flags = respect_flags
        self.force_memory = force_memory
        self.force_disk = force_disk
        self.lazy_cleanup = lazy_cleanup
        self.failure_mode = failure_mode

    @classmethod
    def for_context(cls, context_type: str) -> 'MaterializationPolicy':
        """
        Create a policy for a specific context type.

        Args:
            context_type: The type of context (e.g., "testing", "benchmark", "production")

        Returns:
            A MaterializationPolicy instance
        """
        if context_type == "testing":
            return cls(respect_flags=False, force_memory=True)
        elif context_type == "benchmark":
            return cls(respect_flags=True, force_disk=True)
        elif context_type == "production":
            return cls(respect_flags=True, lazy_cleanup=True)
        else:
            return cls()  # Default policy


class MaterializationError(Exception):
    """Exception raised when materialization fails."""
    pass


class MaterializationManager:
    """
    Centralized manager for materialization operations.

    This class is responsible for:
    1. Determining if materialization is needed for a step
    2. Registering files for materialization
    3. Executing materialization operations
    4. Handling materialization failures
    """

    def __init__(
        self,
        context,
        *,
        storage_config: 'StorageConfig',
        flag_engine: 'FlagInferenceEngine',
        policy: Optional[MaterializationPolicy] = None,
    ):
        """
        Initialize the materialization manager.

        Args:
            context: The processing context
            storage_config: Storage configuration
            flag_engine: Flag inference engine to use
            policy: Materialization policy to use
        """
        self.context = context
        self.policy = policy or MaterializationPolicy()
        self.pending_operations = {}  # Dict[str, OverlayOperation]
        self.storage_config = storage_config
        self.storage_mode = storage_config.storage_mode
        self.overlay_mode = storage_config.overlay_mode
        self.overlay_root = storage_config.overlay_root
        
        # Require a flag inference engine
        if flag_engine is None:
            raise ValueError("MaterializationManager requires a FlagInferenceEngine")
        self.flag_inference_engine = flag_engine

    @property
    def orchestrator(self):
        """Get the orchestrator from the context."""
        return getattr(self.context, 'orchestrator', None)

    @property
    def storage_adapter(self):
        """Get the storage adapter from the orchestrator."""
        if not self.orchestrator:
            return None
        return getattr(self.orchestrator, 'storage_adapter', None)

    @property
    def file_manager(self):
        """Get the file manager from the orchestrator."""
        if not self.orchestrator:
            return None
        return getattr(self.orchestrator, 'file_manager', None)

    @property
    def orchestrator_storage_mode(self):
        """Get the storage mode from the orchestrator."""
        if not self.orchestrator:
            return "legacy"
        return getattr(self.orchestrator, 'storage_mode', "legacy")

    @property
    def orchestrator_overlay_mode(self):
        """Get the overlay mode from the orchestrator."""
        if not self.orchestrator:
            return OverlayMode.DISABLED
        return getattr(self.orchestrator, 'overlay_mode', OverlayMode.DISABLED)

    def needs_materialization(
        self,
        step,
        pipeline: Optional[Any] = None
    ) -> bool:
        """
        Determine if a step needs materialization.

        Args:
            step: The step to check
            pipeline: The pipeline containing the step (optional)

        Returns:
            True if materialization is needed, False otherwise
        """
        # Skip if using legacy storage mode
        if self.storage_mode == "legacy":
            return False

        # Skip if overlay is disabled
        if self.overlay_mode == OverlayMode.DISABLED:
            return False

        # Check policy overrides
        if self.policy.force_memory:
            return False

        if self.policy.force_disk:
            return True

        # Infer flags if pipeline is provided
        if pipeline and self.flag_inference_engine:
            inferred_flags = self.flag_inference_engine.infer_flags(step, pipeline, self.context)

            # Check inferred flags
            if (inferred_flags.get('inferred_requires_fs_input', False) or
                inferred_flags.get('inferred_requires_fs_output', False)):
                logger.debug("Step %s needs materialization based on inferred flags",
                            step.__class__.__name__)
                return True

        # Special case for ImageStitchingStep: check if input_positions needs materialization
        if self.context:
            # Import here to avoid circular imports
            from ezstitcher.core.steps import ImageStitchingStep
            if isinstance(step, ImageStitchingStep):
                # Check if positions_dir is in context
                positions_dir = getattr(self.context, 'positions_dir', None)
                if positions_dir:
                    # Check if positions_dir is a non-disk VirtualPath
                    from ezstitcher.io.virtual_path import VirtualPath, PhysicalPath
                    if isinstance(positions_dir, VirtualPath) and not isinstance(positions_dir, PhysicalPath):
                        # Positions directory is a non-disk VirtualPath, needs materialization
                        logger.debug("ImageStitchingStep needs materialization for positions_dir")
                        return True

                    # Check if positions file exists and is a non-disk VirtualPath
                    well = self.context.well_filter[0] if hasattr(self.context, 'well_filter') and self.context.well_filter else None
                    if well:
                        from pathlib import Path
                        positions_file = positions_dir / f"{well}.csv"
                        if isinstance(positions_file, VirtualPath) and not isinstance(positions_file, PhysicalPath):
                            # Positions file is a non-disk VirtualPath, needs materialization
                            logger.debug("ImageStitchingStep needs materialization for positions file: %s",
                                        positions_file)
                            return True

        # Check if the step requires filesystem access
        if self.policy.respect_flags:
            # Use the step's declarative needs_materialization method
            return step.needs_materialization()

        # Default to no materialization if flags are not respected
        return False

    def _handle_failure(self, message: str, exception: Exception) -> None:
        """
        Handle a materialization failure.

        Args:
            message: Error message
            exception: The exception that occurred

        Raises:
            MaterializationError: If failure_mode is FAIL_FAST
        """
        if self.policy.failure_mode == FailureMode.FAIL_FAST:
            raise MaterializationError(f"{message}: {exception}")
        else:
            logger.error(f"{message}: {exception}")

    def _construct_key(
        self,
        well: str,
        file_path: Union[str, Path],
        input_dir: Union[str, Path]
    ) -> str:
        """
        Construct a storage key for materialization.

        Args:
            well: Well identifier
            file_path: File path
            input_dir: Input directory

        Returns:
            Storage key in the format "overlay_{well}_{relative_path}"
        """
        path = Path(file_path)
        input_path = Path(input_dir)

        # Get the relative path of the file from input_dir
        try:
            rel_path = path.relative_to(input_path) if path.is_relative_to(input_path) else path.name
        except Exception:
            # Fallback to using the filename if relative_to fails
            rel_path = path.name

        # Return a key in the format "overlay_{well}_{relative_path}"
        return f"overlay_{well}_{rel_path}"

    def register_file(
        self,
        file_path: Union[str, Path],
        well: str,
        input_dir: Union[str, Path],
        operation_type: Literal["read", "write", "both"] = "read",
        cleanup: bool = True
    ) -> Optional[Path]:
        """
        Register a file for materialization.

        Args:
            file_path: Path of the file to register
            well: Well identifier
            input_dir: Input directory
            operation_type: Type of operation ("read", "write", "both")
            cleanup: Whether to clean up the file after use

        Returns:
            Path where the file will be materialized, or None if registration failed
        """
        if not self.storage_adapter:
            return None

        # Construct a key based on the file path
        key = self._construct_key(well, file_path, input_dir)

        # Try to get the data from storage adapter
        try:
            if self.storage_adapter.exists(key):
                # Register for overlay
                disk_path = self.storage_adapter.register_for_overlay(
                    key,
                    operation_type=operation_type,
                    cleanup=cleanup
                )
                if disk_path:
                    # Store the operation in our pending operations
                    if key in self.storage_adapter.overlay_operations:
                        self.pending_operations[key] = self.storage_adapter.overlay_operations[key]
                    logger.debug(f"Registered materialization for {file_path} -> {disk_path}")
                    return disk_path
                else:
                    logger.debug(f"Failed to register materialization for {file_path}")
            else:
                logger.debug(f"Key not found in storage adapter: {key}")
        except Exception as e:
            self._handle_failure(f"Error registering materialization for {file_path}", e)

        return None

    def register_files(
        self,
        files: List[Union[str, Path]],
        well: str,
        input_dir: Union[str, Path],
        operation_type: Literal["read", "write", "both"] = "read",
        cleanup: bool = True
    ) -> Dict[str, Path]:
        """
        Register multiple files for materialization.

        Args:
            files: List of file paths to register
            well: Well identifier
            input_dir: Input directory
            operation_type: Type of operation ("read", "write", "both")
            cleanup: Whether to clean up the files after use

        Returns:
            Dictionary mapping original file paths to materialized paths
        """
        materialized_paths = {}

        for file_path in files:
            disk_path = self.register_file(file_path, well, input_dir, operation_type, cleanup)
            if disk_path:
                materialized_paths[str(file_path)] = disk_path

        return materialized_paths

    def register_patterns(
        self,
        patterns: List[str],
        well: str,
        input_dir: Union[str, Path],
        operation_type: Literal["read", "write", "both"] = "read",
        cleanup: bool = True
    ) -> Dict[str, Path]:
        """
        Register all files matching patterns for materialization.

        Args:
            patterns: List of patterns to match
            well: Well identifier
            input_dir: Input directory
            operation_type: Type of operation ("read", "write", "both")
            cleanup: Whether to clean up the files after use

        Returns:
            Dictionary mapping original file paths to materialized paths
        """
        if not patterns or not self.orchestrator:
            return {}

        # Get all matching files
        all_files = []
        for pattern in patterns:
            try:
                matching_files = self.orchestrator.microscope_handler.parser.path_list_from_pattern(
                    input_dir, pattern
                )
                all_files.extend(matching_files)
            except Exception as e:
                self._handle_failure(f"Error getting files for pattern {pattern}", e)

        # Register files for materialization
        return self.register_files(all_files, well, input_dir, operation_type, cleanup)

    def prepare_for_step(
        self,
        step,
        well: str,
        input_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Prepare materialization for a specific step.

        This method registers files for materialization but does not execute
        the materialization operations. Call execute_pending_operations() to
        actually materialize the files.

        Args:
            step: The step to prepare for
            well: Well identifier
            input_dir: Input directory

        Returns:
            Dictionary mapping original file paths to materialized paths
        """
        if not self.needs_materialization(step):
            return {}

        try:
            # Get patterns for the well with recursive scanning
            all_patterns = get_patterns_for_well(
                well, input_dir, self.orchestrator.microscope_handler, recursive=True
            )

            if not all_patterns:
                logger.warning(f"No patterns found for well {well}")
                return {}

            # Register all files matching the patterns for materialization
            return self.register_patterns(all_patterns, well, input_dir, "read", True)
        except Exception as e:
            self._handle_failure(f"Error preparing materialization for step {step.name}", e)
            return {}

    def execute_pending_operations(self) -> int:
        """
        Execute all pending materialization operations.

        Returns:
            Number of operations executed
        """
        if not self.storage_adapter or not self.file_manager:
            return 0

        executed_count = 0
        for key in list(self.pending_operations.keys()):
            try:
                if self.storage_adapter.execute_overlay_operation(key, self.file_manager):
                    executed_count += 1
                    # Remove from pending operations
                    del self.pending_operations[key]
            except Exception as e:
                self._handle_failure(f"Error executing materialization operation for key {key}", e)

        return executed_count

    def cleanup_operations(self) -> int:
        """
        Clean up all executed materialization operations.

        Returns:
            Number of operations cleaned up
        """
        if not self.storage_adapter or not self.file_manager:
            return 0

        return self.storage_adapter.cleanup_overlay_operations(self.file_manager)

    def list_rules(self) -> List[Dict[str, Any]]:
        """
        List all registered materialization rules.

        Returns:
            A list of dictionaries containing rule information
        """
        rules = []
        for pattern, rule in self.rules.items():
            rules.append({
                "pattern": pattern,
                "step_type": rule.step_type,
                "requires_fs_read": rule.requires_fs_read,
                "requires_fs_write": rule.requires_fs_write,
                "priority": rule.priority
            })
        return sorted(rules, key=lambda r: r["priority"], reverse=True)
