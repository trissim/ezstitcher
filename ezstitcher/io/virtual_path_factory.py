"""
Factory for creating VirtualPath objects.

This module provides a factory for creating VirtualPath objects from various sources,
such as physical paths, storage keys, or URIs.
"""

import threading
from pathlib import Path
from typing import Any, Optional, Union

from ezstitcher.io.virtual_path import VirtualPath, PhysicalPath
from ezstitcher.io.memory_path import MemoryPath
from ezstitcher.io.zarr_path import ZarrPath
from ezstitcher.io.storage_config import StorageConfig


# Thread lock for protecting class-level state
_factory_lock = threading.RLock()


class VirtualPathFactory:
    """
    Factory for creating VirtualPath objects.

    This class provides methods for creating VirtualPath objects from various sources,
    such as physical paths, storage keys, or URIs.

    A default root context can be set at the class level to apply to all virtual paths
    created by this factory, unless overridden by a method-specific root_context parameter.
    """

    # Class-level default root context and storage config
    default_root_context: Optional[str] = None
    _storage_config: Optional[StorageConfig] = None

    @classmethod
    def set_default_root_context(cls, root_context: str, *, storage_config: Optional[StorageConfig] = None) -> None:
        """
        Set the default root context for all virtual paths created by this factory.

        Args:
            root_context: The default root context
            storage_config: Optional storage configuration
        """
        with _factory_lock:
            cls.default_root_context = root_context
            cls._storage_config = storage_config

    @classmethod
    def get_default_root_context(cls) -> Optional[str]:
        """
        Get the default root context.

        Returns:
            The default root context, or None if not set
        """
        with _factory_lock:
            return cls.default_root_context

    @classmethod
    def get_storage_config(cls) -> Optional[StorageConfig]:
        """
        Get the storage configuration.

        Returns:
            The storage configuration, or None if not set
        """
        with _factory_lock:
            return cls._storage_config

    @classmethod
    def from_path(cls, path: Union[str, Path], root_context: Optional[str] = None) -> VirtualPath:
        """
        Create a VirtualPath from a physical path.

        Args:
            path: The physical path
            root_context: The root context (overrides default_root_context if provided)

        Returns:
            A VirtualPath object
        """
        # Use provided root_context or fall back to default_root_context
        with _factory_lock:
            effective_root_context = root_context if root_context is not None else cls.default_root_context

        if isinstance(path, VirtualPath):
            if effective_root_context is not None:
                return path.with_root_context(effective_root_context)
            return path
        else:
            return PhysicalPath(path, effective_root_context)

    @classmethod
    def from_storage_key(cls, key: str, backend: Any = None, root_context: Optional[str] = None) -> VirtualPath:
        """
        Create a VirtualPath from a storage key.

        Args:
            key: The storage key
            backend: The backend to use
            root_context: The root context (overrides default_root_context if provided)

        Returns:
            A VirtualPath object
        """
        # Use provided root_context or fall back to default_root_context
        with _factory_lock:
            effective_root_context = root_context if root_context is not None else cls.default_root_context

        if key.startswith('memory:'):
            return MemoryPath.from_storage_key(key, backend, effective_root_context)
        if key.startswith('zarr:'):
            return ZarrPath.from_storage_key(key, backend, effective_root_context)
        # Assume it's a physical path
        return PhysicalPath.from_storage_key(key, backend, effective_root_context)

    @classmethod
    def from_uri(cls, uri: str, root_context: Optional[str] = None) -> VirtualPath:
        """
        Create a VirtualPath from a URI.

        Args:
            uri: The URI
            root_context: The root context (overrides default_root_context if provided)

        Returns:
            A VirtualPath object
        """
        # Use provided root_context or fall back to default_root_context
        with _factory_lock:
            effective_root_context = root_context if root_context is not None else cls.default_root_context

        if uri.startswith('memory:'):
            return MemoryPath.from_storage_key(uri, None, effective_root_context)
        if uri.startswith('zarr:'):
            return ZarrPath.from_storage_key(uri, None, effective_root_context)
        # Assume it's a physical path
        return PhysicalPath(uri, effective_root_context)

    @classmethod
    def create_file_zarr(cls, path: Path, *, root_context: Optional[str] = None) -> Tuple["VirtualPath", Any]:
        """
        Create a new file-based Zarr store and return the root path.

        Args:
            path: The path to the Zarr store
            root_context: The root context (keyword-only)

        Returns:
            A tuple of (root VirtualPath, store)
        """
        from ezstitcher.io.zarr_path import ZarrPath  # lazy import to avoid cycles
        return ZarrPath.create_file_zarr(path, root_context=root_context)

    @classmethod
    def create_memory_zarr(cls, *, root_context: Optional[str] = None) -> Tuple["VirtualPath", Any]:
        """
        Create a new in-memory Zarr store and return the root path.

        Args:
            root_context: The root context (keyword-only)

        Returns:
            A tuple of (root VirtualPath, store)
        """
        from ezstitcher.io.zarr_path import ZarrPath  # lazy import to avoid cycles
        return ZarrPath.create_memory_zarr(root_context=root_context)

    @classmethod
    def open_zarr(cls, path: Path, *, root_context: Optional[str] = None) -> Tuple["VirtualPath", Any]:
        """
        Open an existing Zarr store and return the root path.

        Args:
            path: The path to the Zarr store
            root_context: The root context (keyword-only)

        Returns:
            A tuple of (root VirtualPath, store)
        """
        from ezstitcher.io.zarr_path import ZarrPath  # lazy import to avoid cycles
        return ZarrPath.open_zarr(path, root_context=root_context)
