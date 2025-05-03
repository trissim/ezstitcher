# ezstitcher/io/storage_adapter.py
from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Dict, List, Literal, Optional, Union, Tuple, Any, Set, Pattern
import numpy as np
import logging
import zarr
import shutil

from ezstitcher.io.overlay import OverlayMode, OverlayOperation
from ezstitcher.io.virtual_path import VirtualPath
from ezstitcher.io.virtual_path_factory import VirtualPathFactory # Corrected import
from ezstitcher.io.storage_config import StorageConfig # Added import

logger = logging.getLogger(__name__)

class StorageAdapter(ABC):
    """Abstract base class for key-value storage of pipeline artifacts (primarily NumPy arrays)."""

    # Note: *args removed as they weren't used and kwargs covers extensibility
    def __init__(self, *, storage_config: StorageConfig, **kwargs):
        super().__init__(**kwargs) # Pass along any other kwargs
        self.storage_config = storage_config # Store config
        self.overlay_mode = storage_config.overlay_mode # Use config
        self.overlay_root = storage_config.overlay_root # Use config
        self.overlay_operations = {}  # Dict[str, OverlayOperation]
        self.key_patterns: Dict[Pattern, str] = {}
        self.key_mappings: Dict[str, VirtualPath] = {}
        # Note: configure_overlay might become redundant or change purpose later

    def configure_overlay(self, mode: OverlayMode, root_dir: Optional[Path] = None):
        """Configure overlay disk writes."""
        self.overlay_mode = mode
        self.overlay_root = root_dir
        logger.info(f"Configured overlay mode: {mode.name}, root: {root_dir}")

    def register_overlay_operation(self, operation: OverlayOperation) -> bool:
        """Register an overlay operation."""
        if self.overlay_mode == OverlayMode.DISABLED:
            logger.debug(f"Overlay disabled, not registering operation for key: {operation.key}")
            return False

        self.overlay_operations[operation.key] = operation
        logger.debug(f"Registered overlay operation: {operation}")
        return True

    def execute_overlay_operation(self, key: str, file_manager) -> bool:
        """
        Execute a registered overlay operation.
        
        The data is lazily loaded only when needed for the write operation,
        reducing memory usage for large arrays.
        """
        if key not in self.overlay_operations:
            logger.warning(f"No overlay operation registered for key: {key}")
            return False

        operation = self.overlay_operations[key]
        if operation.executed:
            logger.debug(f"Overlay operation already executed for key: {key}")
            return True

        try:
            if operation.operation_type in ("write", "both"):
                # Ensure directory exists
                file_manager.ensure_directory(operation.disk_path.parent)
                
                # Lazily load data only when needed
                data = operation.data_supplier()
                
                # Write data to disk
                success = file_manager.save_image(data, operation.disk_path)
                if success:
                    logger.debug(f"Executed overlay write for key: {key} to path: {operation.disk_path}")
                    # Verify the file was actually written
                    if file_manager.exists(operation.disk_path):
                        operation.executed = True
                        return True
                    else:
                        logger.error(f"File was not created after save_image: {operation.disk_path}")
                        return False
                else:
                    logger.error(f"Failed to save image to {operation.disk_path}")
                    return False
            else:
                # For read-only operations, just mark as executed
                operation.executed = True
                return True
        except Exception as e:
            logger.error(f"Error executing overlay operation for key: {key}: {e}")
            return False

    def cleanup_overlay_operations(self, file_manager) -> int:
        """Clean up executed overlay operations."""
        cleaned_count = 0
        for key, operation in list(self.overlay_operations.items()):
            if operation.executed and operation.cleanup:
                try:
                    if file_manager.exists(operation.disk_path):
                        success = file_manager.delete_file(operation.disk_path)
                        if success:
                            logger.debug(f"Cleaned up overlay file: {operation.disk_path}")
                            cleaned_count += 1
                        else:
                            logger.warning(f"Failed to clean up overlay file: {operation.disk_path}")
                    else:
                        logger.debug(f"Overlay file already removed or never existed: {operation.disk_path}")
                        # Still count it as cleaned since it's not there
                        cleaned_count += 1

                    # Remove from operations dict regardless of file deletion success
                    # This prevents repeated cleanup attempts
                    del self.overlay_operations[key]
                    logger.debug(f"Removed overlay operation for key: {key}")
                except Exception as e:
                    logger.error(f"Error cleaning up overlay operation for key: {key}: {e}")

        return cleaned_count

    def execute_all_overlay_operations(self, file_manager) -> int:
        """Execute all registered overlay operations."""
        executed_count = 0
        for key in list(self.overlay_operations.keys()):
            if self.execute_overlay_operation(key, file_manager):
                executed_count += 1

        return executed_count

    @abstractmethod
    def write(self, key: str, data: np.ndarray) -> None:
        """Store data associated with a key."""
        pass

    @abstractmethod
    def read(self, key: str) -> np.ndarray:
        """Retrieve data associated with a key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the storage."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the data associated with a key."""
        pass

    @abstractmethod
    def list_keys(self, pattern: str = "*") -> List[str]: # Added pattern arg to abstract
        """List keys in the storage adapter matching a pattern."""
        pass

    @abstractmethod
    def persist(self, output_dir: Path) -> None:
        """
        Persist the contents of the storage to a specified directory.
        Behavior depends on the implementation (e.g., write files, no-op).
        """
        pass

    def register_pattern(self, pattern: str, template: str) -> None:
        """
        Register a pattern for logical key resolution.

        Args:
            pattern: A regular expression pattern for matching logical keys
            template: A template for converting matched keys to paths
        """
        self.key_patterns[re.compile(pattern)] = template

    def register_mapping(self, key: str, path: Union[str, Path, VirtualPath]) -> None:
        """
        Register a direct mapping from a logical key to a path.

        Args:
            key: The logical key
            path: The path to map to
        """
        if isinstance(path, (str, Path)):
            self.key_mappings[key] = VirtualPathFactory.from_path(path)
        else:
            self.key_mappings[key] = path

    def resolve_key(self, key: str) -> Optional[VirtualPath]:
        """
        Resolve a logical key to a virtual path.

        Args:
            key: The logical key

        Returns:
            A VirtualPath object, or None if the key cannot be resolved
        """
        # Check direct mappings first
        if key in self.key_mappings:
            return self.key_mappings[key]

        # Then try patterns
        for pattern, template in self.key_patterns.items():
            match = pattern.match(key)
            if match:
                # Replace capture groups in the template
                path_str = template
                for i, group in enumerate(match.groups(), 1):
                    path_str = path_str.replace(f"${i}", group)

                # Create a virtual path from the resolved path string
                return VirtualPathFactory.from_path(path_str)

        # If we get here, the key couldn't be resolved
        return None

# Removed concrete implementation of list_keys from ABC


class MemoryStorageAdapter(StorageAdapter):
    """Stores pipeline artifacts in an in-memory dictionary."""

    # Note: *args removed
    def __init__(self, *, storage_config: StorageConfig, **kwargs):
        # Pass storage_config via keyword to super
        super().__init__(storage_config=storage_config, **kwargs)
        self._store: Dict[str, np.ndarray] = {}
        logger.info("Initialized MemoryStorageAdapter.")

        # Register default key patterns
        self.register_pattern(
            r"([^/]+)/([^/]+)/([^/]+)\.tif",
            "wells/$1/channels/$2/$3.tif"
        )
        self.register_pattern(
            r"([^/]+)_([^/]+)\.tif",
            "wells/$1/channels/$2.tif"
        )

    def write(self, key: str, data: np.ndarray) -> None:
        """Store data in the in-memory dictionary."""
        # Store a copy to prevent external modifications affecting the stored array
        self._store[key] = data.copy()
        logger.debug(f"Wrote data for key '{key}' to memory. Store now contains {len(self._store)} keys.")

    def read(self, key: str) -> np.ndarray:
        """Retrieve data from the in-memory dictionary."""
        if key not in self._store:
            logger.error(f"Key '{key}' not found in memory store.")
            raise KeyError(f"Key '{key}' not found in MemoryStorageAdapter.")
        logger.debug(f"Read data for key '{key}' from memory.")
        # Return a copy to prevent external modifications to the stored array
        return self._store[key].copy()

    def exists(self, key: str) -> bool:
        """Check if a key exists in the in-memory dictionary."""
        exists = key in self._store
        logger.debug(f"Checked existence for key '{key}': {exists}.")
        return exists

    def delete(self, key: str) -> None:
        """Delete data associated with a key from the in-memory dictionary."""
        if key in self._store:
            del self._store[key]
            logger.info(f"Deleted data for key '{key}' from memory.")
        else:
            logger.warning(f"Attempted to delete non-existent key '{key}' from memory.")

    def list_keys(self, pattern: str = "*") -> List[str]:
        """
        List keys in the memory storage adapter matching a pattern.

        Args:
            pattern: Pattern to match keys against (e.g., "well_A01/*")

        Returns:
            List of keys matching the pattern
        """
        import fnmatch
        keys = [key for key in self._store if fnmatch.fnmatch(key, pattern)]
        logger.debug("Listed keys from memory store matching pattern '%s': %s",
                    pattern, keys)
        return keys

    def get_overlay_path(self, key: str) -> Optional[Path]:
        """Get the overlay disk path for a key."""
        if self.overlay_mode == OverlayMode.DISABLED or self.overlay_root is None:
            return None

        if key in self.overlay_operations and self.overlay_operations[key].executed:
            return self.overlay_operations[key].disk_path

        # Generate a path based on the key
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.overlay_root / f"{safe_key}.tif"

    def register_for_overlay(self, key: str, operation_type: Literal["read", "write", "both"] = "both", cleanup: bool = True) -> Optional[Path]:
        """Register a key for overlay disk write."""
        if self.overlay_mode == OverlayMode.DISABLED or self.overlay_root is None:
            return None

        if key not in self._store:
            logger.warning(f"Key not found in store: {key}")
            return None

        disk_path = self.get_overlay_path(key)
        
        # Create a supplier function that will lazily load the data when called
        def data_supplier():
            return self._store[key]
        
        operation = OverlayOperation(
            key=key,
            data_supplier=data_supplier,
            disk_path=disk_path,
            operation_type=operation_type,
            cleanup=cleanup
        )

        self.register_overlay_operation(operation)
        return disk_path

    def persist(self, output_dir: Path) -> None:
        """Save all stored NumPy arrays to .npy files in the specified directory."""
        logger.info(f"Persisting memory store contents to '{output_dir}'...")
        try:
            # Check if there are any keys to persist
            if not self._store:
                logger.warning("No keys in memory store, nothing to persist")
                return

            # Log the keys that will be persisted
            keys = list(self._store.keys())
            logger.debug("Memory store contains %d keys to persist: %s",
                        len(keys), keys[:5] if len(keys) > 5 else keys)

            # Ensure the output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Persist each key
            for key, data in self._store.items():
                file_path = output_dir / f"{key}.npy"
                np.save(file_path, data)
                logger.debug(f"Saved data for key '{key}' to '{file_path}'.")

            # Verify files were written
            npy_files = list(output_dir.glob("*.npy"))
            logger.info(f"Successfully persisted {len(npy_files)} files to '{output_dir}'.")
        except Exception as e:
            logger.error(f"Failed to persist memory store to '{output_dir}': {e}", exc_info=True)
            # Depending on requirements, might re-raise or handle differently
            raise

class ZarrStorageAdapter(StorageAdapter):
    """Stores pipeline artifacts in a Zarr store on disk."""

    # Note: *args removed, storage_root becomes keyword-only after storage_config
    def __init__(self, *, storage_config: StorageConfig, storage_root: Path, **kwargs):
        # Pass storage_config via keyword to super
        super().__init__(storage_config=storage_config, **kwargs)
        self.storage_root = storage_root
        self.zarr_path = self.storage_root / "data.zarr"
        try:
            # Ensure the parent directory exists
            self.storage_root.mkdir(parents=True, exist_ok=True)
            # Open the Zarr store, creating if it doesn't exist
            self.root = zarr.open(str(self.zarr_path), mode='a')
            logger.info(f"Initialized ZarrStorageAdapter at '{self.zarr_path}'.")

            # Register default key patterns
            self.register_pattern(
                r"([^/]+)/([^/]+)/([^/]+)\.tif",
                "wells/$1/channels/$2/$3.tif"
            )
            self.register_pattern(
                r"([^/]+)_([^/]+)\.tif",
                "wells/$1/channels/$2.tif"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Zarr store at '{self.zarr_path}': {e}", exc_info=True)
            raise

    def write(self, key: str, data: np.ndarray) -> None:
        """Write data to the Zarr store."""
        try:
            self.root[key] = data
            logger.debug(f"Wrote data for key '{key}' to Zarr store '{self.zarr_path}'.")
        except Exception as e:
            logger.error(f"Failed to write key '{key}' to Zarr store '{self.zarr_path}': {e}", exc_info=True)
            raise

    def read(self, key: str) -> np.ndarray:
        """Read data from the Zarr store."""
        if key not in self.root:
            logger.error(f"Key '{key}' not found in Zarr store '{self.zarr_path}'.")
            raise KeyError(f"Key '{key}' not found in ZarrStorageAdapter at '{self.zarr_path}'.")
        try:
            data = self.root[key][...] # Read the full array into memory
            logger.debug(f"Read data for key '{key}' from Zarr store '{self.zarr_path}'.")
            return data
        except Exception as e:
            logger.error(f"Failed to read key '{key}' from Zarr store '{self.zarr_path}': {e}", exc_info=True)
            raise

    def exists(self, key: str) -> bool:
        """Check if a key exists in the Zarr store."""
        exists = key in self.root
        logger.debug(f"Checked existence for key '{key}' in Zarr store '{self.zarr_path}': {exists}.")
        return exists

    def delete(self, key: str) -> None:
        """Delete data associated with a key from the Zarr store."""
        if key in self.root:
            try:
                del self.root[key]
                logger.info(f"Deleted data for key '{key}' from Zarr store '{self.zarr_path}'.")
            except Exception as e:
                logger.error(f"Failed to delete key '{key}' from Zarr store '{self.zarr_path}': {e}", exc_info=True)
                raise
        else:
            logger.warning(f"Attempted to delete non-existent key '{key}' from Zarr store '{self.zarr_path}'.")

    def list_keys(self, pattern: str = "*") -> List[str]:
        """
        List keys in the Zarr store matching a pattern.

        Args:
            pattern: Pattern to match keys against (e.g., "well_A01/*")

        Returns:
            List of keys matching the pattern
        """
        try:
            import fnmatch
            all_keys = list(self.root.keys())
            keys = [key for key in all_keys if fnmatch.fnmatch(key, pattern)]
            logger.debug("Listed keys from Zarr store '%s' matching pattern '%s': %s",
                        self.zarr_path, pattern, keys)
            return keys
        except Exception as e:
            logger.error("Failed to list keys from Zarr store '%s': %s",
                        self.zarr_path, e, exc_info=True)
            raise

    def get_overlay_path(self, key: str) -> Optional[Path]:
        """Get the overlay disk path for a key."""
        if self.overlay_mode == OverlayMode.DISABLED or self.overlay_root is None:
            return None

        if key in self.overlay_operations and self.overlay_operations[key].executed:
            return self.overlay_operations[key].disk_path

        # Generate a path based on the key
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.overlay_root / f"{safe_key}.tif"

    def register_for_overlay(self, key: str, operation_type: Literal["read", "write", "both"] = "both", cleanup: bool = True) -> Optional[Path]:
        """Register a key for overlay disk write."""
        if self.overlay_mode == OverlayMode.DISABLED or self.overlay_root is None:
            return None

        if key not in self.root:
            logger.warning(f"Key not found in Zarr store: {key}")
            return None

        disk_path = self.get_overlay_path(key)
        
        # Create a supplier function that captures the Zarr array reference.
        # The actual data read ([...]) is deferred until the supplier is called,
        # ensuring lazy loading and minimal memory footprint during registration.
        # .copy() is added to prevent issues if the underlying Zarr slice is mutated
        # concurrently before the supplier is executed.
        supplier = lambda arr=self.root[key]: arr[...].copy()

        operation = OverlayOperation(
            key=key,
            data_supplier=supplier, # Use the lazy supplier
            disk_path=disk_path,
            operation_type=operation_type,
            cleanup=cleanup
        )

        self.register_overlay_operation(operation)
        return disk_path

    def persist(self, output_dir: Path) -> None:
        """No-op for ZarrStorageAdapter as data is written directly to disk."""
        logger.info(f"Persist called for Zarr store '{self.zarr_path}'. This is a no-op as data is already persisted. Ignoring output_dir '{output_dir}'.")
        pass # Explicitly a no-op
def generate_storage_key(step_name: str, well: Optional[str] = None,
                     component: Optional[str] = None) -> str:
    """
    Generate a consistent storage key for a step's output.

    Args:
        step_name: Name of the step
        well: Well identifier (e.g., 'A01')
        component: Component identifier (e.g., 'channel_1')

    Returns:
        A storage key in the format "step_name/well/component" or similar
    """
    # Normalize step name (lowercase, replace spaces with underscores)
    normalized_step = step_name.strip().lower().replace(' ', '_')

    # Build the key based on available components
    if well and component:
        return f"{normalized_step}/{well}/{component}"
    elif well:
        return f"{normalized_step}/{well}"
    elif component:
        return f"{normalized_step}/{component}"
    else:
        return normalized_step


# Final update to select_storage signature and calls
def select_storage(
    mode: Literal["memory", "zarr"],
    *,  # Make all following args keyword-only
    storage_config: StorageConfig,
    storage_root: Optional[Path] = None
) -> StorageAdapter:
    """
    Factory function to select and instantiate a storage adapter.

    Args:
        mode: The type of storage adapter to create ('memory' or 'zarr').
        storage_config: The storage configuration object (keyword-only).
        storage_root: The root directory for storage (keyword-only). Required by Zarr,
                      potentially used by Memory for persist/overlay. Defaults to None.

    Returns:
        An instance of the selected StorageAdapter.

    Raises:
        ValueError: If an invalid mode is provided or required args are missing.
    """
    logger.info("Selecting storage adapter: mode='%s', storage_root='%s'", mode, storage_root)
    mode = mode.lower()  # Normalize mode
    if mode == "memory":
        # Pass storage_config as keyword argument
        return MemoryStorageAdapter(storage_config=storage_config)
    elif mode == "zarr":
        if storage_root is None:
            raise ValueError("storage_root is required for Zarr storage mode.")
        # Pass storage_config and storage_root as keyword arguments
        return ZarrStorageAdapter(storage_config=storage_config, storage_root=storage_root)
    else:
        logger.error("Invalid storage adapter mode specified: '%s'", mode)
        raise ValueError(f"Invalid storage mode: {mode}. Choose 'memory' or 'zarr'.")


def resolve_persist_path(
    storage_mode: Literal["legacy", "memory", "zarr"],
    workspace_path: Optional[Path] = None,
    storage_root: Optional[Path] = None
) -> Optional[Path]:
    """
    Resolve the appropriate path for persisting storage adapter data.

    Args:
        storage_mode: The storage mode ("legacy", "memory", "zarr")
        workspace_path: The workspace path from the orchestrator
        storage_root: The storage root path from the orchestrator or config

    Returns:
        The resolved path for persist() or None if persist is not applicable

    Raises:
        ValueError: If the storage mode requires a path but none is available
    """
    logger.debug(
        "Resolving persist path for mode=%s, workspace_path=%s, storage_root=%s",
        storage_mode, workspace_path, storage_root
    )

    # For zarr and legacy modes, persist is not applicable
    if storage_mode in ("zarr", "legacy"):
        logger.debug("No persist path needed for %s mode", storage_mode)
        return None

    # For memory mode, we need a path to persist to
    if storage_mode == "memory":
        # Priority 1: Use storage_root if provided (explicit configuration)
        if storage_root:
            persist_path = storage_root
            logger.debug("Using storage_root for persist path: %s", persist_path)
            return persist_path

        # Priority 2: Use workspace_path/adapter_output/memory if workspace_path is available
        if workspace_path:
            persist_path = workspace_path / "adapter_output" / "memory"
            logger.debug("Using workspace-based path for persist: %s", persist_path)
            return persist_path

        # No valid path available
        raise ValueError(
            "Cannot resolve persist path for memory mode: "
            "neither storage_root nor workspace_path is available"
        )

    # Unknown storage mode
    raise ValueError(f"Unknown storage mode: {storage_mode}")


def write_result(context, key: str, data: np.ndarray, fallback_path: Optional[Path] = None,
                file_manager = None) -> bool:
    """
    Write result data to the appropriate storage backend.

    Uses StorageAdapter if available, otherwise falls back to FileManager.

    Args:
        context: The ProcessingContext containing the orchestrator and storage_adapter
        key: The storage key to use for the adapter
        data: The data to store (typically a numpy array)
        fallback_path: The path to save to if using FileManager fallback
        file_manager: Optional FileManager instance to use for fallback

    Returns:
        bool: True if write was successful, False otherwise
    """
    # Check if we have a valid context with a storage adapter
    has_adapter = False
    if context and hasattr(context, 'orchestrator') and context.orchestrator:
        # Check if storage adapter is available and not in legacy mode
        adapter = getattr(context.orchestrator, 'storage_adapter', None)
        storage_mode = getattr(context.orchestrator, 'storage_mode', "legacy")
        has_adapter = adapter is not None and storage_mode != "legacy"

        # Enhanced debugging information
        logger.debug("write_result: has_adapter=%s, storage_mode=%s, key=%s, data_type=%s",
                    has_adapter, storage_mode, key, type(data).__name__)

        # Add shape information if it's a numpy array
        if isinstance(data, np.ndarray):
            logger.debug("Data shape: %s, dtype: %s", data.shape, data.dtype)

        if has_adapter:
            try:
                # Verify data is a numpy array
                if not isinstance(data, np.ndarray):
                    logger.warning("Data for key '%s' is not a numpy array (type: %s). Converting...",
                                  key, type(data).__name__)
                    # Try to convert to numpy array
                    data = np.asarray(data)

                # Store the data using the storage adapter
                adapter.write(key, data)
                adapter_type = type(adapter).__name__
                logger.debug("Successfully stored array with key '%s' using %s",
                            key, adapter_type)
                return True
            except (ValueError, KeyError, IOError) as e:
                logger.error("Failed to write key '%s' to storage adapter: %s",
                            key, e, exc_info=True)
                # Fall through to FileManager fallback if fallback_path is provided

    # Fall back to FileManager if adapter not available or write failed
    if fallback_path is not None:
        if file_manager is None and context and hasattr(context, 'orchestrator'):
            file_manager = getattr(context.orchestrator, 'file_manager', None)

        if file_manager:
            try:
                file_manager.save_image(data, fallback_path)
                logger.debug("Saved image to '%s' using FileManager (fallback)",
                            fallback_path)
                return True
            except (ValueError, IOError) as e:
                logger.error("Failed to save image to '%s': %s",
                            fallback_path, e, exc_info=True)
                return False
        else:
            logger.error("No FileManager available for fallback save")
            return False

    # If we get here with no fallback_path, it means we couldn't use the adapter
    # and had no fallback option
    if has_adapter:
        logger.warning("Could not write data with key '%s' - adapter failed and no fallback",
                      key)

    return False
