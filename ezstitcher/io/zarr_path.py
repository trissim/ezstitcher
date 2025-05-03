"""
Zarr-based implementation of VirtualPath.

This module provides a VirtualPath implementation that works with Zarr arrays,
allowing for operations on virtual paths that correspond to locations within
a Zarr hierarchy.
"""

import os
import io
import threading
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Union, Iterator, Any, Tuple, Set
import re

import zarr
import numpy as np

from ezstitcher.io.virtual_path import VirtualPath
from ezstitcher.io.virtual_path_factory import VirtualPathFactory


# Thread lock for protecting the Zarr cache
_zarr_cache_lock = threading.RLock()


class ZarrPath(VirtualPath):
    """
    A virtual path that corresponds to a location within a Zarr hierarchy.

    This class implements the VirtualPath interface for Zarr storage,
    allowing operations on paths that correspond to Zarr groups and arrays.
    """

    def __init__(self, path: Union[str, PurePosixPath], store: Optional[zarr.storage.Store] = None,
                 root: Optional[str] = None, root_context: Optional[str] = None):
        """
        Initialize a Zarr path.

        Args:
            path: The path within the Zarr hierarchy
            store: The Zarr store to use (default: None, which creates a memory store)
            root: The root path within the store (default: None, which uses the store root)
            root_context: The root context
        """
        if isinstance(path, PurePosixPath):
            self._path = path
        else:
            # Normalize path to use forward slashes and remove trailing slash
            normalized_path = str(path).replace('\\', '/')
            if normalized_path.endswith('/') and normalized_path != '/':
                normalized_path = normalized_path[:-1]
            self._path = PurePosixPath(normalized_path)

        # Initialize store
        self._store = store if store is not None else zarr.MemoryStore()
        self._root = root if root is not None else ''
        self._root_context = root_context

        # Cache for Zarr objects with thread lock
        self._zarr_cache = {}
        self._cache_lock = threading.RLock()

    def _get_full_path(self) -> str:
        """
        Get the full path within the Zarr store.

        Returns:
            The full path string
        """
        path_str = str(self._path)
        if path_str.startswith('/'):
            path_str = path_str[1:]  # Remove leading slash

        if self._root:
            if self._root.endswith('/'):
                return f"{self._root}{path_str}"
            else:
                return f"{self._root}/{path_str}"
        else:
            return path_str

    def _get_zarr_object(self) -> Optional[Union[zarr.Group, zarr.Array]]:
        """
        Get the Zarr object (group or array) at this path.

        Returns:
            A Zarr group or array, or None if the path doesn't exist
        """
        full_path = self._get_full_path()

        # Check cache first
        with self._cache_lock:
            if full_path in self._zarr_cache:
                return self._zarr_cache[full_path]

        # Try to get the object
        try:
            # First try to open as a group
            obj = zarr.open_group(self._store, path=full_path)
            with self._cache_lock:
                self._zarr_cache[full_path] = obj
            return obj
        except (zarr.errors.PathNotFoundError, KeyError):
            # Path doesn't exist
            return None
        except Exception:
            # Not a group, try as an array
            try:
                obj = zarr.open_array(self._store, path=full_path)
                with self._cache_lock:
                    self._zarr_cache[full_path] = obj
                return obj
            except (zarr.errors.PathNotFoundError, KeyError):
                # Path doesn't exist
                return None
            except Exception:
                # Not an array either
                return None

    def __str__(self) -> str:
        """Return a string representation of the path."""
        return str(self._path)

    def __truediv__(self, other) -> 'ZarrPath':
        """
        Join this path with another.

        Args:
            other: The path to join with this one

        Returns:
            A new ZarrPath representing the joined path
        """
        if isinstance(other, VirtualPath):
            # If other is a VirtualPath, try to convert it to a string
            other_str = str(other)
        else:
            other_str = str(other)

        return ZarrPath(self._path / other_str, store=self._store, root=self._root, root_context=self._root_context)

    @staticmethod
    def create_memory_zarr(root_context: Optional[str] = None) -> Tuple['ZarrPath', zarr.storage.Store]:
        """
        Create a new in-memory Zarr store and return the root path.

        Args:
            root_context: The root context

        Returns:
            A tuple of (root ZarrPath, store)
        """
        # Delegate to VirtualPathFactory
        store = zarr.MemoryStore()
        # Create root group
        zarr.group(store=store)
        return ZarrPath('/', store=store, root_context=root_context), store

    @staticmethod
    def create_file_zarr(path: Union[str, Path], root_context: Optional[str] = None) -> Tuple['ZarrPath', zarr.storage.Store]:
        """
        Create a new file-based Zarr store and return the root path.

        Args:
            path: The path to the Zarr store
            root_context: The root context

        Returns:
            A tuple of (root ZarrPath, store)
        """
        # Delegate to VirtualPathFactory
        path = Path(path)
        store = zarr.DirectoryStore(str(path))
        # Create root group
        zarr.group(store=store)
        return ZarrPath('/', store=store, root_context=root_context), store

    @staticmethod
    def open_zarr(path: Union[str, Path], root_context: Optional[str] = None) -> Tuple['ZarrPath', zarr.storage.Store]:
        """
        Open an existing Zarr store and return the root path.

        Args:
            path: The path to the Zarr store
            root_context: The root context

        Returns:
            A tuple of (root ZarrPath, store)
        """
        # Delegate to VirtualPathFactory
        path = Path(path)
        store = zarr.DirectoryStore(str(path))
        return ZarrPath('/', store=store, root_context=root_context), store

    def name(self) -> str:
        """Return the name of the file or directory."""
        return self._path.name

    def stem(self) -> str:
        """Return the stem (name without suffix) of the file."""
        return self._path.stem

    def suffix(self) -> str:
        """Return the suffix (extension) of the file."""
        return self._path.suffix

    @property
    def root_context(self) -> Optional[str]:
        """
        Get the root context for this path.

        Returns:
            The root context, or None if not set
        """
        return self._root_context

    def with_root_context(self, root_context: str) -> 'ZarrPath':
        """
        Create a new zarr path with the specified root context.

        Args:
            root_context: The root context

        Returns:
            A new zarr path with the specified root context
        """
        return ZarrPath(self._path, store=self._store, root=self._root, root_context=root_context)

    def parent(self) -> 'ZarrPath':
        """Return the parent directory of this path."""
        return ZarrPath(self._path.parent, store=self._store, root=self._root, root_context=self._root_context)

    def exists(self) -> bool:
        """Check if the path exists."""
        # Special case for root
        if str(self._path) == '/':
            return True

        # Check if the path exists in the Zarr store
        return self._get_zarr_object() is not None

    def is_file(self) -> bool:
        """
        Check if the path is a file (Zarr array).

        In Zarr, 'files' are arrays.
        """
        obj = self._get_zarr_object()
        return obj is not None and isinstance(obj, zarr.Array)

    def is_dir(self) -> bool:
        """
        Check if the path is a directory (Zarr group).

        In Zarr, 'directories' are groups.
        """
        obj = self._get_zarr_object()
        return obj is not None and isinstance(obj, zarr.Group)

    def glob(self, pattern: str) -> Iterator['ZarrPath']:
        """
        Glob the given pattern in this path.

        Args:
            pattern: The glob pattern to match

        Returns:
            An iterator of matching ZarrPath objects
        """
        # Convert glob pattern to regex pattern
        regex_pattern = self._glob_to_regex(pattern)

        # Get the Zarr object at this path
        obj = self._get_zarr_object()
        if obj is None or not isinstance(obj, zarr.Group):
            # Not a group, so no children to glob
            return

        # Get all keys in the store that start with this path
        base_path = self._get_full_path()
        if base_path and not base_path.endswith('/'):
            base_path += '/'

        # List all keys in the store
        all_keys = []
        for key in self._store.keys():
            key_str = str(key)
            if key_str.startswith(base_path):
                # Remove the base path prefix
                rel_key = key_str[len(base_path):]
                # Split by '/' to get the first component
                first_component = rel_key.split('/', 1)[0]
                if first_component and first_component not in all_keys:
                    all_keys.append(first_component)

        # Match keys against the pattern
        for key in all_keys:
            if re.fullmatch(regex_pattern, key):
                yield ZarrPath(self._path / key, store=self._store, root=self._root, root_context=self._root_context)

    def _glob_to_regex(self, pattern: str) -> str:
        """
        Convert a glob pattern to a regex pattern.

        Args:
            pattern: The glob pattern

        Returns:
            A regex pattern string
        """
        # Escape all regex special characters except * and ?
        pattern = re.escape(pattern)

        # Convert glob * to regex .*
        pattern = pattern.replace('\\*', '.*')

        # Convert glob ? to regex .
        pattern = pattern.replace('\\?', '.')

        # Handle ** (match any number of directories)
        pattern = pattern.replace('.*\\.\\*', '.*')

        return f'^{pattern}$'

    def relative_to(self, other: Union[str, 'VirtualPath']) -> 'ZarrPath':
        """
        Return a version of this path relative to the other path.

        Args:
            other: The path to make this path relative to

        Returns:
            A new ZarrPath representing the relative path
        """
        if isinstance(other, VirtualPath) and not isinstance(other, ZarrPath):
            raise ValueError(f"Cannot make {self} relative to non-Zarr path {other}")

        other_str = str(other)
        self_str = str(self._path)

        if not self_str.startswith(other_str):
            raise ValueError(f"{self_str} is not in the subpath of {other_str}")

        if other_str == '/':
            # Special case for root directory
            rel_path = self_str[1:]
        elif self_str == other_str:
            # Same path
            rel_path = '.'
        else:
            # Remove other_str prefix and leading slash
            rel_path = self_str[len(other_str) + 1:]

        return ZarrPath(rel_path, store=self._store, root=self._root, root_context=self._root_context)

    def is_relative_to(self, other: Union[str, 'VirtualPath']) -> bool:
        """
        Check if this path is relative to the other path.

        Args:
            other: The path to check against

        Returns:
            True if this path is relative to the other path, False otherwise
        """
        try:
            self.relative_to(other)
            return True
        except ValueError:
            return False

    def resolve(self) -> 'ZarrPath':
        """
        Resolve the path to its absolute form.

        Returns:
            A new ZarrPath representing the absolute path
        """
        # Zarr paths are already absolute, so just return a copy
        return ZarrPath(self._path, store=self._store, root=self._root, root_context=self._root_context)

    def with_name(self, name: str) -> 'ZarrPath':
        """
        Return a new path with the name changed.

        Args:
            name: The new name

        Returns:
            A new ZarrPath with the name changed

        Raises:
            ValueError: If this path doesn't have a name component
        """
        if self._path.name == '':
            raise ValueError(f"{self} has an empty name")

        return ZarrPath(self._path.parent / name, store=self._store, root=self._root, root_context=self._root_context)

    def with_suffix(self, suffix: str) -> 'ZarrPath':
        """
        Return a new path with the suffix changed.

        Args:
            suffix: The new suffix (must start with a dot)

        Returns:
            A new ZarrPath with the suffix changed
        """
        if not suffix.startswith('.') and suffix != '':
            suffix = '.' + suffix

        name = self._path.stem + suffix
        return self.with_name(name)

    def iterdir(self) -> Iterator['ZarrPath']:
        """
        Iterate over the arrays and groups in this group.

        Returns:
            An iterator of ZarrPath objects
        """
        # Get the Zarr object at this path
        obj = self._get_zarr_object()
        if obj is None or not isinstance(obj, zarr.Group):
            raise NotADirectoryError(f"{self} is not a directory (Zarr group)")

        # Get all direct children
        base_path = self._get_full_path()
        if base_path and not base_path.endswith('/'):
            base_path += '/'

        # List all keys in the store
        children = set()
        for key in self._store.keys():
            key_str = str(key)
            if key_str.startswith(base_path):
                # Remove the base path prefix
                rel_key = key_str[len(base_path):]
                # Split by '/' to get the first component
                first_component = rel_key.split('/', 1)[0]
                if first_component and first_component not in children:
                    children.add(first_component)
                    yield ZarrPath(self._path / first_component, store=self._store, root=self._root, root_context=self._root_context)

    def open(self, mode: str = 'r', **kwargs) -> Any:
        """
        Open the array at this path.

        For Zarr paths, this returns a file-like object that wraps a Zarr array.

        Args:
            mode: The mode to open the file in
            **kwargs: Additional arguments to pass to the underlying open function

        Returns:
            A file-like object
        """
        if 'w' in mode or 'a' in mode or '+' in mode:
            # Writing or appending mode
            if self.exists() and self.is_dir():
                raise IsADirectoryError(f"{self} is a directory (Zarr group)")

            # Create a BytesIO buffer that will be written to the Zarr array when closed
            if 'a' in mode and self.exists():
                # Appending to existing array
                content = self.read_bytes()
            else:
                # New content
                content = b''

            buffer = io.BytesIO(content)
            if 'a' in mode:
                buffer.seek(0, 2)  # Seek to end

            # Create a wrapper that updates the Zarr array when closed
            return self._create_file_wrapper(buffer, mode, **kwargs)
        else:
            # Reading mode
            if not self.exists():
                raise FileNotFoundError(f"{self} does not exist")
            if self.is_dir():
                raise IsADirectoryError(f"{self} is a directory (Zarr group)")

            # Read the array and return a BytesIO
            content = self.read_bytes()
            return io.BytesIO(content)

    def _create_file_wrapper(self, buffer: io.BytesIO, mode: str, **kwargs) -> Any:
        """
        Create a wrapper for a BytesIO that updates the Zarr array when closed.

        Args:
            buffer: The buffer to wrap
            mode: The file mode
            **kwargs: Additional arguments

        Returns:
            A wrapped BytesIO
        """
        original_close = buffer.close

        def new_close():
            # Get the content from the buffer
            buffer.seek(0)
            content = buffer.getvalue()

            # Write to the Zarr array
            self.write_bytes(content)

            # Call the original close method
            original_close()

        buffer.close = new_close
        return buffer

    def read_bytes(self) -> bytes:
        """
        Read the contents of the array as bytes.

        For large arrays (>10MB), this method streams the data in chunks
        to avoid loading the entire array into memory at once.

        Returns:
            The contents of the array as bytes
        """
        obj = self._get_zarr_object()
        if obj is None:
            raise FileNotFoundError(f"{self} does not exist")
        if isinstance(obj, zarr.Group):
            raise IsADirectoryError(f"{self} is a directory (Zarr group)")

        # If it's a 1D byte array, return it directly
        if obj.dtype == np.dtype('uint8') and obj.ndim == 1:
            # Check if array is large (>10MB)
            array_size_bytes = obj.size
            if array_size_bytes <= 10 * 1024 * 1024:  # 10MB
                return bytes(obj[:])
            else:
                # Stream large arrays in chunks to avoid doubling memory usage
                buffer = io.BytesIO()
                chunk_size = 1024 * 1024  # 1MB chunks
                for i in range(0, array_size_bytes, chunk_size):
                    end = min(i + chunk_size, array_size_bytes)
                    chunk = obj[i:end]
                    buffer.write(chunk.tobytes(order='C'))
                buffer.seek(0)
                return buffer.getvalue()

        # Otherwise, serialize the array
        buffer = io.BytesIO()
        np.save(buffer, obj[:])
        buffer.seek(0)
        return buffer.read()

    def read_text(self, encoding: str = 'utf-8') -> str:
        """
        Read the contents of the array as text.

        Args:
            encoding: The encoding to use

        Returns:
            The contents of the array as text
        """
        return self.read_bytes().decode(encoding)

    def write_bytes(self, data: bytes, overwrite: bool = False) -> int:
        """
        Write bytes to the array.

        Args:
            data: The bytes to write
            overwrite: If True, overwrite existing data; if False, raise FileExistsError if path exists

        Returns:
            The number of bytes written
        """
        # Ensure parent group exists
        parent = self.parent()
        if not parent.exists():
            parent.mkdir(parents=True)

        # Check if path exists and we're not overwriting
        full_path = self._get_full_path()
        if not overwrite and full_path in self._store:
            raise FileExistsError(f"Path already exists: {full_path}")

        # Create a 1D byte array
        zarr.create(shape=len(data), dtype=np.dtype('uint8'), store=self._store, path=full_path, overwrite=overwrite)

        # Write the data
        array = zarr.open_array(self._store, path=full_path)
        array[:] = np.frombuffer(data, dtype=np.dtype('uint8'))

        # Clear cache
        if full_path in self._zarr_cache:
            del self._zarr_cache[full_path]

        return len(data)

    def write_text(self, data: str, encoding: str = 'utf-8') -> int:
        """
        Write text to the array.

        Args:
            data: The text to write
            encoding: The encoding to use

        Returns:
            The number of bytes written
        """
        return self.write_bytes(data.encode(encoding))

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        """
        Create a group at this path.

        Args:
            parents: If True, create parent groups as needed
            exist_ok: If True, don't raise an error if the group already exists
        """
        # Check if path already exists
        if self.exists():
            if self.is_dir():
                if not exist_ok:
                    raise FileExistsError(f"{self} already exists")
                return
            else:
                raise FileExistsError(f"{self} exists and is not a directory (Zarr group)")

        # Check if parent exists
        parent = self.parent()
        if str(self._path) != '/' and not parent.exists():
            if not parents:
                raise FileNotFoundError(f"Parent directory {parent} does not exist")
            # Create parent groups
            parent.mkdir(parents=True, exist_ok=True)

        # Create the group
        full_path = self._get_full_path()
        zarr.group(store=self._store, path=full_path)

        # Clear cache
        if full_path in self._zarr_cache:
            del self._zarr_cache[full_path]

    def rmdir(self) -> None:
        """Remove the group at this path."""
        # Check if path exists
        if not self.exists():
            raise FileNotFoundError(f"{self} does not exist")
        if not self.is_dir():
            raise NotADirectoryError(f"{self} is not a directory (Zarr group)")

        # Check if group is empty
        for _ in self.iterdir():
            raise OSError(f"{self} is not empty")

        # Remove the group
        full_path = self._get_full_path()

        # Zarr doesn't have a direct way to remove a group, so we need to
        # remove all keys that start with this path
        keys_to_remove = []
        for key in self._store.keys():
            key_str = str(key)
            if key_str == full_path or key_str.startswith(full_path + '/'):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._store[key]

        # Clear cache
        if full_path in self._zarr_cache:
            del self._zarr_cache[full_path]

    def unlink(self, missing_ok: bool = False) -> None:
        """
        Remove the array at this path.

        Args:
            missing_ok: If True, don't raise an error if the array doesn't exist
        """
        # Check if path exists
        if not self.exists():
            if not missing_ok:
                raise FileNotFoundError(f"{self} does not exist")
            return
        if self.is_dir():
            raise IsADirectoryError(f"{self} is a directory (Zarr group)")

        # Remove the array
        full_path = self._get_full_path()

        # Zarr doesn't have a direct way to remove an array, so we need to
        # remove all keys that start with this path
        keys_to_remove = []
        for key in self._store.keys():
            key_str = str(key)
            if key_str == full_path or key_str.startswith(full_path + '/'):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._store[key]

        # Clear cache
        if full_path in self._zarr_cache:
            del self._zarr_cache[full_path]

    def to_physical_path(self) -> Optional[Path]:
        """
        Convert this virtual path to a physical path, if possible.

        Returns:
            A pathlib.Path object if the store is a DirectoryStore, None otherwise
        """
        if isinstance(self._store, zarr.storage.DirectoryStore):
            # For DirectoryStore, we can get the physical path
            store_path = Path(self._store.path)
            rel_path = self._get_full_path()

            # Special case for root
            if rel_path == '':
                return store_path

            return store_path / rel_path
        else:
            # For other stores, there's no physical path
            return None

    def to_storage_key(self) -> str:
        """
        Convert this Zarr path to a storage key.

        Returns:
            A string key that can be used with a storage adapter
        """
        # Use the path as the key, with a 'zarr:' prefix
        return f"zarr:{self._get_full_path()}"

    @classmethod
    def from_storage_key(cls, key: str, backend: Any = None, root_context: Optional[str] = None) -> 'ZarrPath':
        """
        Create a Zarr path from a storage key.

        Args:
            key: The storage key
            backend: The backend to use (should be a Zarr store)
            root_context: The root context

        Returns:
            A ZarrPath object
        """
        if key.startswith('zarr:'):
            # Remove the 'zarr:' prefix
            path = key[5:]
        else:
            # Assume the key is a path
            path = key

        if not path.startswith('/'):
            path = '/' + path

        return cls(path, store=backend, root_context=root_context)
