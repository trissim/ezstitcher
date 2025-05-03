"""
Memory-based implementation of VirtualPath.

This module provides a VirtualPath implementation that stores data in memory,
allowing for operations on virtual paths without requiring disk I/O.
"""

import os
import io
import threading # Added import
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Union, Iterator, Any, Tuple, Set
import re

from ezstitcher.io.virtual_path import VirtualPath


class MemoryPath(VirtualPath):
    """
    A virtual path that corresponds to an in-memory location.

    This class implements the VirtualPath interface for in-memory storage,
    allowing operations on paths and data without requiring disk I/O.
    """

    # Class-level storage for all memory paths
    # Structure: {path_str: {'type': 'file'|'dir', 'content': bytes|None}}
    _storage: Dict[str, Dict[str, Any]] = {}
    _storage_lock: threading.RLock = threading.RLock() # Correctly initialized lock

    def __init__(self, path: Union[str, PurePosixPath], root_context: Optional[str] = None):
        """
        Initialize a memory path.

        Args:
            path: The memory path as a string or PurePosixPath
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

        self._root_context = root_context

        # Ensure parent directories exist
        self._ensure_parent_dirs()

    # --- Thread-safe Storage Access Helpers ---

    @classmethod
    def _get_blob(cls, key: str) -> Optional[Dict[str, Any]]:
        """Safely get an item from storage."""
        with cls._storage_lock:
            return cls._storage.get(key)

    @classmethod
    def _set_blob(cls, key: str, value: Dict[str, Any]) -> None:
        """Safely set an item in storage."""
        with cls._storage_lock:
            cls._storage[key] = value

    @classmethod
    def _del_blob(cls, key: str) -> None:
        """Safely delete an item from storage."""
        with cls._storage_lock:
            # Use pop with default to avoid KeyError if key doesn't exist
            cls._storage.pop(key, None)

    @classmethod
    def _has_blob(cls, key: str) -> bool:
        """Safely check if an item exists in storage."""
        with cls._storage_lock:
            return key in cls._storage

    @classmethod
    def _get_all_keys(cls) -> List[str]:
        """Safely get all keys from storage."""
        with cls._storage_lock:
            return list(cls._storage.keys())

    @classmethod
    def _clear_storage(cls) -> None:
        """Safely clear the storage and ensure root exists."""
        with cls._storage_lock:
            cls._storage.clear()
            cls._storage['/'] = {'type': 'dir', 'content': None}

    @classmethod
    def _get_storage_copy(cls) -> Dict[str, Dict[str, Any]]:
        """Safely get a copy of the storage."""
        with cls._storage_lock:
            # Return a deep copy if values are mutable, but dicts are shallow copied ok here
            return cls._storage.copy()

    # --- End Storage Access Helpers ---


    def _ensure_parent_dirs(self) -> None:
        """
        Ensure that all parent directories of this path exist in the storage.
        Uses thread-safe helpers.
        """
        path_str = str(self._path)
        if path_str == '/':
            # Root directory always exists
            if not self._has_blob('/'):
                self._set_blob('/', {'type': 'dir', 'content': None})
            return

        # Collect directories to create without holding the lock for the whole loop
        current = self._path.parent
        dirs_to_create = []
        while str(current) != '/':
            if not self._has_blob(str(current)):
                 # Check existence with helper
                dirs_to_create.append(str(current))
            current = current.parent

        # Ensure root exists (check again before setting)
        if not self._has_blob('/'):
            self._set_blob('/', {'type': 'dir', 'content': None})

        # Create directories from root down, checking existence again inside lock
        for dir_path in reversed(dirs_to_create):
            # Check again inside the loop in case another thread created it
            if not self._has_blob(dir_path):
                self._set_blob(dir_path, {'type': 'dir', 'content': None})

    def __str__(self) -> str:
        """Return a string representation of the path."""
        return str(self._path)

    def __truediv__(self, other) -> 'MemoryPath':
        """
        Join this path with another.

        Args:
            other: The path to join with this one

        Returns:
            A new MemoryPath representing the joined path
        """
        if isinstance(other, VirtualPath):
            # If other is a VirtualPath, try to convert it to a string
            other_str = str(other)
        else:
            other_str = str(other)

        return MemoryPath(self._path / other_str, self._root_context)

    @staticmethod
    def reset_storage() -> None:
        """
        Reset the in-memory storage, clearing all files and directories.
        Uses thread-safe helper.

        This is primarily useful for testing.
        """
        MemoryPath._clear_storage() # Use helper

    @staticmethod
    def get_storage_snapshot() -> Dict[str, Dict[str, Any]]:
        """
        Get a snapshot of the current in-memory storage.
        Uses thread-safe helper.

        Returns:
            A dictionary representing the current state of the in-memory storage
        """
        return MemoryPath._get_storage_copy() # Use helper

    @staticmethod
    def _is_child_path(parent: str, child: str) -> bool:
        """
        Check if child is a direct child of parent.

        Args:
            parent: The parent path
            child: The child path

        Returns:
            True if child is a direct child of parent, False otherwise
        """
        if not child.startswith(parent):
            return False

        if parent == '/':
            # Special case for root directory
            relative = child[1:]
        else:
            # Remove parent prefix and leading slash
            relative = child[len(parent) + 1:]

        # Check if there are any slashes in the relative path
        return '/' not in relative

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

    def with_root_context(self, root_context: str) -> 'MemoryPath':
        """
        Create a new memory path with the specified root context.

        Args:
            root_context: The root context

        Returns:
            A new memory path with the specified root context
        """
        return MemoryPath(self._path, root_context)

    def parent(self) -> 'MemoryPath':
        """Return the parent directory of this path."""
        return MemoryPath(self._path.parent, self._root_context)

    def exists(self) -> bool:
        """Check if the path exists."""
        return self._has_blob(str(self._path)) # Use helper

    def is_file(self) -> bool:
        """Check if the path is a file."""
        blob = self._get_blob(str(self._path)) # Use helper
        return blob is not None and blob['type'] == 'file'

    def is_dir(self) -> bool:
        """Check if the path is a directory."""
        blob = self._get_blob(str(self._path)) # Use helper
        return blob is not None and blob['type'] == 'dir'

    def glob(self, pattern: str) -> Iterator['MemoryPath']:
        """
        Glob the given pattern in this path.

        Args:
            pattern: The glob pattern to match

        Returns:
            An iterator of matching MemoryPath objects
        """
        # Convert glob pattern to regex pattern
        regex_pattern = self._glob_to_regex(pattern)
        base_path = str(self._path)
        if base_path != '/' and not base_path.endswith('/'):
            base_path += '/'

        # Find all paths that match the pattern
        all_keys = self._get_all_keys() # Use helper
        for path in all_keys:
            # Skip paths that aren't under this directory
            if not path.startswith(base_path) and path != base_path:
                continue

            # Get the relative path
            if base_path == '/':
                rel_path = path[1:]  # Remove leading slash
            else:
                rel_path = path[len(base_path):]

            # Skip the directory itself
            if not rel_path:
                continue

            # Check if the relative path matches the pattern
            if re.fullmatch(regex_pattern, rel_path):
                yield MemoryPath(path, self._root_context)

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

    def relative_to(self, other: Union[str, 'VirtualPath']) -> 'MemoryPath':
        """
        Return a version of this path relative to the other path.

        Args:
            other: The path to make this path relative to

        Returns:
            A new MemoryPath representing the relative path
        """
        if isinstance(other, VirtualPath) and not isinstance(other, MemoryPath):
            raise ValueError(f"Cannot make {self} relative to non-memory path {other}")

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

        return MemoryPath(rel_path, self._root_context)

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

    def resolve(self) -> 'MemoryPath':
        """
        Resolve the path to its absolute form.

        Returns:
            A new MemoryPath representing the absolute path
        """
        # Memory paths are already absolute, so just return a copy
        return MemoryPath(self._path, self._root_context)

    def with_name(self, name: str) -> 'MemoryPath':
        """
        Return a new path with the name changed.

        Args:
            name: The new name

        Returns:
            A new MemoryPath with the name changed

        Raises:
            ValueError: If this path doesn't have a name component
        """
        if self._path.name == '':
            raise ValueError(f"{self} has an empty name")

        return MemoryPath(self._path.parent / name, self._root_context)

    def with_suffix(self, suffix: str) -> 'MemoryPath':
        """
        Return a new path with the suffix changed.

        Args:
            suffix: The new suffix (must start with a dot)

        Returns:
            A new MemoryPath with the suffix changed
        """
        if not suffix.startswith('.') and suffix != '':
            suffix = '.' + suffix

        name = self._path.stem + suffix
        return self.with_name(name)

    def iterdir(self) -> Iterator['MemoryPath']:
        """
        Iterate over the files and directories in this directory.

        Returns:
            An iterator of MemoryPath objects
        """
        if not self.is_dir():
            raise NotADirectoryError(f"{self} is not a directory")

        # Get all direct children of this directory
        base_path = str(self._path)
        if base_path != '/' and not base_path.endswith('/'):
            base_path += '/'

        all_keys = self._get_all_keys() # Use helper
        for path in all_keys:
            # Skip paths that aren't direct children of this directory
            if not MemoryPath._is_child_path(base_path, path):
                continue

            yield MemoryPath(path, self._root_context)

    def open(self, mode: str = 'r', **kwargs) -> Any:
        """
        Open the file at this path.

        Args:
            mode: The mode to open the file in
            **kwargs: Additional arguments to pass to the underlying open function

        Returns:
            A file-like object
        """
        path_str = str(self._path)

        if 'w' in mode or 'a' in mode or '+' in mode:
            # Writing or appending mode
            # Ensure parent dirs exist first (uses helpers)
            self._ensure_parent_dirs()

            # Use lock for check-then-set logic
            with self._storage_lock:
                blob = self._storage.get(path_str) # Direct access within lock is ok
                if blob is None:
                    # Create a new file blob
                    blob = {'type': 'file', 'content': b''}
                    self._storage[path_str] = blob # Direct access within lock
                elif blob['type'] != 'file':
                    raise IsADirectoryError(f"{path_str} is a directory")

                # Get current content safely within lock
                current_content = blob['content']

            # Prepare buffer outside lock
            if 'b' in mode:
                # Binary mode
                content_to_open = current_content if 'a' in mode else b''
                buffer = io.BytesIO(content_to_open)
                if 'a' in mode:
                    buffer.seek(0, 2) # Seek to end for append
                # else: default seek is 0 for write/overwrite
            else:
                # Text mode
                encoding = kwargs.get('encoding', 'utf-8')
                content_to_open = current_content.decode(encoding) if 'a' in mode else ''
                buffer = io.StringIO(content_to_open)
                if 'a' in mode:
                    buffer.seek(0, 2) # Seek to end for append
                # else: default seek is 0 for write/overwrite

            # Create a wrapper that updates the storage when closed (uses helpers)
            return self._create_file_wrapper(buffer, path_str, mode, **kwargs)
        else:
            # Reading mode
            blob = self._get_blob(path_str) # Use helper
            if blob is None:
                raise FileNotFoundError(f"{path_str} does not exist")
            if blob['type'] != 'file':
                raise IsADirectoryError(f"{path_str} is a directory")

            content = blob['content'] # Content is immutable (bytes)
            if 'b' in mode:
                # Binary mode
                return io.BytesIO(content)
            else:
                # Text mode
                return io.StringIO(content.decode(kwargs.get('encoding', 'utf-8')))

    def _create_file_wrapper(self, buffer: Union[io.BytesIO, io.StringIO], path_str: str, mode: str, **kwargs) -> Any:
        """
        Create a wrapper for a file-like object that updates the storage when closed.

        Args:
            buffer: The buffer to wrap
            path_str: The path string
            mode: The file mode
            **kwargs: Additional arguments

        Returns:
            A wrapped file-like object
        """
        original_close = buffer.close

        def new_close():
            # Get the content from the buffer
            buffer.seek(0)
            content = buffer.getvalue()

            # Update the storage
            if isinstance(content, str):
                # Convert string to bytes
                content = content.encode(kwargs.get('encoding', 'utf-8'))

            # Safely update storage using helper
            # Get current blob first to ensure type is correct
            current_blob = self._get_blob(path_str)
            if current_blob and current_blob['type'] == 'file':
                 self._set_blob(path_str, {'type': 'file', 'content': content})
            else:
                 # This case implies the file was deleted or changed type between open and close
                 # Or it was never created correctly in open()
                 # For robustness, we could log a warning or raise an error
                 # Here, we'll attempt to set it anyway, assuming it should be a file
                 self._set_blob(path_str, {'type': 'file', 'content': content})

            # Call the original close method
            original_close()

        buffer.close = new_close
        return buffer

    def read_bytes(self) -> bytes:
        """
        Read the contents of the file as bytes.

        Returns:
            The contents of the file as bytes
        """
        path_str = str(self._path)

        blob = self._get_blob(path_str) # Use helper
        if blob is None:
            raise FileNotFoundError(f"{path_str} does not exist")
        if blob['type'] != 'file':
            raise IsADirectoryError(f"{path_str} is a directory")

        return blob['content'] # bytes are immutable, safe to return directly

    def read_text(self, encoding: str = 'utf-8') -> str:
        """
        Read the contents of the file as text.

        Args:
            encoding: The encoding to use

        Returns:
            The contents of the file as text
        """
        return self.read_bytes().decode(encoding)

    def write_bytes(self, data: bytes) -> int:
        """
        Write bytes to the file.

        Args:
            data: The bytes to write

        Returns:
            The number of bytes written
        """
        path_str = str(self._path)

        # Ensure parent directories exist
        self._ensure_parent_dirs()

        # Write the data
        MemoryPath._storage[path_str] = {'type': 'file', 'content': data}

        return len(data)

    def write_text(self, data: str, encoding: str = 'utf-8') -> int:
        """
        Write text to the file.

        Args:
            data: The text to write
            encoding: The encoding to use

        Returns:
            The number of bytes written
        """
        return self.write_bytes(data.encode(encoding))

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        """
        Create a directory at this path.

        Args:
            parents: If True, create parent directories as needed
            exist_ok: If True, don't raise an error if the directory already exists
        """
        path_str = str(self._path)

        if path_str in MemoryPath._storage:
            if MemoryPath._storage[path_str]['type'] == 'dir':
                if not exist_ok:
                    raise FileExistsError(f"{path_str} already exists")
                return
            else:
                raise FileExistsError(f"{path_str} exists and is not a directory")

        # Check if parent exists
        parent_str = str(self._path.parent)
        if parent_str != path_str:  # Not root
            if parent_str not in MemoryPath._storage:
                if not parents:
                    raise FileNotFoundError(f"Parent directory {parent_str} does not exist")
                # Create parent directories
                MemoryPath(parent_str).mkdir(parents=True, exist_ok=True)
            elif MemoryPath._storage[parent_str]['type'] != 'dir':
                raise NotADirectoryError(f"Parent path {parent_str} is not a directory")

        # Create the directory
        MemoryPath._storage[path_str] = {'type': 'dir', 'content': None}

    def rmdir(self) -> None:
        """Remove the directory at this path."""
        path_str = str(self._path)

        if path_str not in MemoryPath._storage:
            raise FileNotFoundError(f"{path_str} does not exist")
        if MemoryPath._storage[path_str]['type'] != 'dir':
            raise NotADirectoryError(f"{path_str} is not a directory")

        # Check if directory is empty
        for other_path in MemoryPath._storage.keys():
            if other_path != path_str and other_path.startswith(path_str + '/'):
                raise OSError(f"{path_str} is not empty")

        # Remove the directory
        del MemoryPath._storage[path_str]

    def unlink(self, missing_ok: bool = False) -> None:
        """
        Remove the file at this path.

        Args:
            missing_ok: If True, don't raise an error if the file doesn't exist
        """
        path_str = str(self._path)

        if path_str not in MemoryPath._storage:
            if not missing_ok:
                raise FileNotFoundError(f"{path_str} does not exist")
            return
        if MemoryPath._storage[path_str]['type'] != 'file':
            raise IsADirectoryError(f"{path_str} is a directory")

        # Remove the file
        del MemoryPath._storage[path_str]

    def to_physical_path(self) -> Optional[Path]:
        """
        Convert this virtual path to a physical path, if possible.

        Returns:
            None, as memory paths don't correspond to physical paths
        """
        return None

    def to_storage_key(self) -> str:
        """
        Convert this memory path to a storage key.

        Returns:
            A string key that can be used with a storage adapter
        """
        # Use the path as the key, with a 'memory:' prefix
        return f"memory:{str(self._path)}"

    @classmethod
    def from_storage_key(cls, key: str, backend: Any = None, root_context: Optional[str] = None) -> 'MemoryPath':
        """
        Create a memory path from a storage key.

        Args:
            key: The storage key
            backend: The backend to use, if needed
            root_context: The root context

        Returns:
            A MemoryPath object
        """
        if key.startswith('memory:'):
            # Remove the 'memory:' prefix
            path = key[7:]
        else:
            # Assume the key is a path
            path = key

        return cls(path, root_context)
