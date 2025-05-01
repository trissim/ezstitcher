# ezstitcher/io/storage_adapter.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Literal, Optional
import numpy as np
import logging
import zarr
import shutil

logger = logging.getLogger(__name__)

class StorageAdapter(ABC):
    """Abstract base class for key-value storage of pipeline artifacts (primarily NumPy arrays)."""

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
    def list_keys(self) -> List[str]:
        """List all keys currently stored."""
        pass

    @abstractmethod
    def persist(self, output_dir: Path) -> None:
        """
        Persist the contents of the storage to a specified directory.
        Behavior depends on the implementation (e.g., write files, no-op).
        """
        pass


class MemoryStorageAdapter(StorageAdapter):
    """Stores pipeline artifacts in an in-memory dictionary."""

    def __init__(self):
        self._store: Dict[str, np.ndarray] = {}
        logger.info("Initialized MemoryStorageAdapter.")

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

    def list_keys(self) -> List[str]:
        """List all keys currently stored in the in-memory dictionary."""
        keys = list(self._store.keys())
        logger.debug(f"Listed keys from memory store: {keys}")
        return keys

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

    def __init__(self, storage_root: Path):
        self.storage_root = storage_root
        self.zarr_path = self.storage_root / "data.zarr"
        try:
            # Ensure the parent directory exists
            self.storage_root.mkdir(parents=True, exist_ok=True)
            # Open the Zarr store, creating if it doesn't exist
            self.root = zarr.open(str(self.zarr_path), mode='a')
            logger.info(f"Initialized ZarrStorageAdapter at '{self.zarr_path}'.")
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

    def list_keys(self) -> List[str]:
        """List all keys (top-level arrays/groups) currently stored in the Zarr store."""
        try:
            keys = list(self.root.keys())
            logger.debug(f"Listed keys from Zarr store '{self.zarr_path}': {keys}")
            return keys
        except Exception as e:
            logger.error(f"Failed to list keys from Zarr store '{self.zarr_path}': {e}", exc_info=True)
            raise

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
        A consistent storage key string
    """
    step_name_safe = step_name.replace(" ", "_").lower()
    well_part = f"_{well}" if well else ""
    component_part = f"_{component}" if component else ""
    return f"{step_name_safe}{well_part}{component_part}"


def select_storage(mode: Literal["memory", "zarr"], storage_root: Path) -> StorageAdapter:
    """
    Factory function to select and instantiate a storage adapter.

    Args:
        mode: The type of storage adapter to create ('memory' or 'zarr').
        storage_root: The root directory for storage. Used by Zarr directly,
                      and by Memory adapter during persist.

    Returns:
        An instance of the selected StorageAdapter.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    logger.info("Selecting storage adapter: mode='%s', storage_root='%s'", mode, storage_root)
    if mode == "memory":
        # Note: storage_root is not used by MemoryStorageAdapter at init, but is conceptually
        # linked to where it *would* persist if asked.
        return MemoryStorageAdapter()
    if mode == "zarr":
        return ZarrStorageAdapter(storage_root=storage_root)

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

        logger.debug("write_result: has_adapter=%s, storage_mode=%s, key=%s",
                    has_adapter, storage_mode, key)

        if has_adapter:
            try:
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