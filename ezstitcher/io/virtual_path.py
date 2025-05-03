"""
Virtual path abstraction for EZStitcher.

This module provides a virtual path abstraction that allows for operations on paths
across different storage backends, similar to pathlib.Path but backend-agnostic.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union, cast
import warnings


class VirtualPath(ABC):
    """
    Abstract base class for virtual paths.

    This class provides a common interface for working with paths across
    different storage backends, similar to pathlib.Path but backend-agnostic.
    """

    @property
    @abstractmethod
    def root_context(self) -> Optional[str]:
        """
        Get the root context for this path.

        Returns:
            The root context, or None if not set
        """
        pass

    @abstractmethod
    def with_root_context(self, root_context: str) -> 'VirtualPath':
        """
        Create a new virtual path with the specified root context.

        Args:
            root_context: The root context

        Returns:
            A new virtual path with the specified root context
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the path."""
        pass

    @abstractmethod
    def __truediv__(self, other) -> 'VirtualPath':
        """
        Join this path with another.

        Args:
            other: The path to join with this one

        Returns:
            A new virtual path representing the joined path
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the file or directory.

        Returns:
            The name of the file or directory
        """
        pass

    @abstractmethod
    def stem(self) -> str:
        """
        Return the stem (name without suffix) of the file.

        Returns:
            The stem of the file
        """
        pass

    @abstractmethod
    def suffix(self) -> str:
        """
        Return the suffix (extension) of the file.

        Returns:
            The suffix of the file
        """
        pass

    @abstractmethod
    def parent(self) -> 'VirtualPath':
        """
        Return the parent directory of this path.

        Returns:
            The parent directory
        """
        pass

    @abstractmethod
    def exists(self) -> bool:
        """
        Check if the path exists.

        Returns:
            True if the path exists, False otherwise
        """
        pass

    @abstractmethod
    def is_file(self) -> bool:
        """
        Check if the path is a file.

        Returns:
            True if the path is a file, False otherwise
        """
        pass

    @abstractmethod
    def is_dir(self) -> bool:
        """
        Check if the path is a directory.

        Returns:
            True if the path is a directory, False otherwise
        """
        pass

    @abstractmethod
    def glob(self, pattern: str) -> Iterator['VirtualPath']:
        """
        Glob the given pattern in this path.

        Args:
            pattern: The glob pattern to match

        Returns:
            An iterator of matching virtual paths
        """
        pass

    @abstractmethod
    def iterdir(self) -> Iterator['VirtualPath']:
        """
        Iterate over the files and directories in this directory.

        Returns:
            An iterator of virtual paths
        """
        pass

    @abstractmethod
    def open(self, mode: str = 'r', **kwargs) -> Any:
        """
        Open the file at this path.

        Args:
            mode: The mode to open the file in
            **kwargs: Additional arguments to pass to the underlying open function

        Returns:
            A file-like object
        """
        pass

    @abstractmethod
    def read_text(self, encoding: str = 'utf-8') -> str:
        """
        Read the contents of the file as text.

        Args:
            encoding: The encoding to use

        Returns:
            The contents of the file as a string
        """
        pass

    @abstractmethod
    def read_bytes(self) -> bytes:
        """
        Read the contents of the file as bytes.

        Returns:
            The contents of the file as bytes
        """
        pass

    @abstractmethod
    def write_text(self, data: str, encoding: str = 'utf-8') -> int:
        """
        Write text to the file.

        Args:
            data: The text to write
            encoding: The encoding to use

        Returns:
            The number of characters written
        """
        pass

    @abstractmethod
    def write_bytes(self, data: bytes) -> int:
        """
        Write bytes to the file.

        Args:
            data: The bytes to write

        Returns:
            The number of bytes written
        """
        pass

    @abstractmethod
    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        """
        Create a directory at this path.

        Args:
            parents: If True, create parent directories as needed
            exist_ok: If True, don't raise an error if the directory already exists
        """
        pass

    @abstractmethod
    def rmdir(self) -> None:
        """Remove the directory at this path."""
        pass

    @abstractmethod
    def unlink(self, missing_ok: bool = False) -> None:
        """
        Remove the file at this path.

        Args:
            missing_ok: If True, don't raise an error if the file doesn't exist
        """
        pass

    @abstractmethod
    def to_physical_path(self) -> Optional[Path]:
        """
        Convert this virtual path to a physical path, if possible.

        Returns:
            A pathlib.Path object, or None if this path doesn't correspond to a physical path
        """
        pass

    @abstractmethod
    def to_storage_key(self) -> str:
        """
        Convert this virtual path to a storage key.

        Returns:
            A string key that can be used with a storage adapter
        """
        pass

    @classmethod
    @abstractmethod
    def from_storage_key(cls, key: str, backend: Any = None, root_context: Optional[str] = None) -> 'VirtualPath':
        """
        Create a virtual path from a storage key.

        Args:
            key: The storage key
            backend: The backend to use, if needed
            root_context: The root context

        Returns:
            A VirtualPath object
        """
        pass

class PhysicalPath(VirtualPath):
    """
    A virtual path that corresponds to a physical path on disk.

    This class wraps a pathlib.Path object and implements the VirtualPath interface.
    """

    def __init__(self, path: Union[str, Path], root_context: Optional[str] = None):
        """
        Initialize a physical path.

        Args:
            path: The physical path
            root_context: The root context
        """
        self._path = Path(path)
        self._root_context = root_context

    @property
    def root_context(self) -> Optional[str]:
        """
        Get the root context for this path.

        Returns:
            The root context, or None if not set
        """
        return self._root_context

    def with_root_context(self, root_context: str) -> 'PhysicalPath':
        """
        Create a new physical path with the specified root context.

        Args:
            root_context: The root context

        Returns:
            A new physical path with the specified root context
        """
        return PhysicalPath(self._path, root_context)

    def __str__(self) -> str:
        """Return a string representation of the path."""
        return str(self._path)

    def __truediv__(self, other) -> 'PhysicalPath':
        """
        Join this path with another.

        Args:
            other: The path to join with this one

        Returns:
            A new PhysicalPath representing the joined path
        """
        if isinstance(other, VirtualPath):
            # If other is a VirtualPath, try to convert it to a string
            other_str = str(other)
        else:
            other_str = str(other)

        return PhysicalPath(self._path / other_str, self._root_context)

    def name(self) -> str:
        """Return the name of the file or directory."""
        return self._path.name

    def stem(self) -> str:
        """Return the stem (name without suffix) of the file."""
        return self._path.stem

    def suffix(self) -> str:
        """Return the suffix (extension) of the file."""
        return self._path.suffix

    def parent(self) -> 'PhysicalPath':
        """Return the parent directory of this path."""
        return PhysicalPath(self._path.parent, self._root_context)

    def exists(self) -> bool:
        """Check if the path exists."""
        return self._path.exists()

    def is_file(self) -> bool:
        """Check if the path is a file."""
        return self._path.is_file()

    def is_dir(self) -> bool:
        """Check if the path is a directory."""
        return self._path.is_dir()

    def glob(self, pattern: str) -> Iterator['PhysicalPath']:
        """
        Glob the given pattern in this path.

        Args:
            pattern: The glob pattern to match

        Returns:
            An iterator of matching PhysicalPath objects
        """
        for path in self._path.glob(pattern):
            yield PhysicalPath(path, self._root_context)

    def iterdir(self) -> Iterator['PhysicalPath']:
        """
        Iterate over the files and directories in this directory.

        Returns:
            An iterator of PhysicalPath objects
        """
        for path in self._path.iterdir():
            yield PhysicalPath(path, self._root_context)

    def open(self, mode: str = 'r', **kwargs) -> Any:
        """
        Open the file at this path.

        Args:
            mode: The mode to open the file in
            **kwargs: Additional arguments to pass to the underlying open function

        Returns:
            A file-like object
        """
        return open(self._path, mode, **kwargs)

    def read_text(self, encoding: str = 'utf-8') -> str:
        """
        Read the contents of the file as text.

        Args:
            encoding: The encoding to use

        Returns:
            The contents of the file as a string
        """
        return self._path.read_text(encoding=encoding)

    def read_bytes(self) -> bytes:
        """
        Read the contents of the file as bytes.

        Returns:
            The contents of the file as bytes
        """
        return self._path.read_bytes()

    def write_text(self, data: str, encoding: str = 'utf-8') -> int:
        """
        Write text to the file.

        Args:
            data: The text to write
            encoding: The encoding to use

        Returns:
            The number of characters written
        """
        self._path.write_text(data, encoding=encoding)
        return len(data)

    def write_bytes(self, data: bytes) -> int:
        """
        Write bytes to the file.

        Args:
            data: The bytes to write

        Returns:
            The number of bytes written
        """
        self._path.write_bytes(data)
        return len(data)

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        """
        Create a directory at this path.

        Args:
            parents: If True, create parent directories as needed
            exist_ok: If True, don't raise an error if the directory already exists
        """
        self._path.mkdir(parents=parents, exist_ok=exist_ok)

    def rmdir(self) -> None:
        """Remove the directory at this path."""
        self._path.rmdir()

    def unlink(self, missing_ok: bool = False) -> None:
        """
        Remove the file at this path.

        Args:
            missing_ok: If True, don't raise an error if the file doesn't exist
        """
        self._path.unlink(missing_ok=missing_ok)

    def to_physical_path(self) -> Path:
        """
        Convert this virtual path to a physical path.

        Returns:
            A pathlib.Path object
        """
        return self._path

    def to_storage_key(self) -> str:
        """
        Convert this physical path to a storage key.

        Returns:
            A string key that can be used with a storage adapter
        """
        # Use the path as the key, replacing backslashes with forward slashes
        return str(self._path).replace('\\', '/')

    @classmethod
    def from_storage_key(cls, key: str, backend: Any = None, root_context: Optional[str] = None) -> 'PhysicalPath':
        """
        Create a physical path from a storage key.

        Args:
            key: The storage key
            backend: The backend to use, if needed
            root_context: The root context

        Returns:
            A PhysicalPath object
        """
        return cls(key, root_context)


# VirtualPathFactory moved to virtual_path_factory.py to avoid circular imports


class VirtualPathResolver:
    """
    Resolver for virtual paths.

    This class provides methods for resolving virtual paths to physical paths
    or storage keys, and for creating virtual paths from various sources.
    """

    def __init__(self, root: Optional[VirtualPath] = None, root_context: Optional[str] = None):
        """
        Initialize a virtual path resolver.

        Args:
            root: The root path for relative paths
            root_context: The root context
        """
        self.root = root
        self.root_context = root_context

    def resolve(self, path: Union[str, Path, VirtualPath]) -> VirtualPath:
        """
        Resolve a path to a VirtualPath.

        Args:
            path: The path to resolve

        Returns:
            A VirtualPath object
        """
        if isinstance(path, VirtualPath):
            # Apply root context if available
            if self.root_context is not None:
                return path.with_root_context(self.root_context)
            return path
        elif isinstance(path, (str, Path)):
            # Import here to avoid circular imports
            from ezstitcher.io.virtual_path_factory import VirtualPathFactory
            return VirtualPathFactory.from_path(path, self.root_context)
        else:
            raise TypeError(f"Cannot resolve path of type {type(path)}")

    def to_physical_path(self, path: Union[str, Path, VirtualPath]) -> Optional[Path]:
        """
        Convert a path to a physical path, if possible.

        Args:
            path: The path to convert

        Returns:
            A pathlib.Path object, or None if the path doesn't correspond to a physical path
        """
        virtual_path = self.resolve(path)
        return virtual_path.to_physical_path()

    def to_storage_key(self, path: Union[str, Path, VirtualPath]) -> str:
        """
        Convert a path to a storage key.

        Args:
            path: The path to convert

        Returns:
            A string key that can be used with a storage adapter
        """
        virtual_path = self.resolve(path)
        return virtual_path.to_storage_key()


def open_virtual(path: Union[str, Path, VirtualPath], mode: str = 'r', **kwargs) -> Any:
    """
    Open a file at the given virtual path.

    This function provides a unified way to open files across different storage backends.

    Args:
        path: The path to the file
        mode: The mode to open the file in
        **kwargs: Additional arguments to pass to the underlying open function

    Returns:
        A file-like object
    """
    if isinstance(path, VirtualPath):
        return path.open(mode, **kwargs)
    else:
        # Import here to avoid circular imports
        from ezstitcher.io.virtual_path_factory import VirtualPathFactory
        return VirtualPathFactory.from_path(path).open(mode, **kwargs)
