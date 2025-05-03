"""
Utilities for working with paths in EZStitcher.

This module provides utility functions for working with paths in EZStitcher,
including functions for checking path types and converting between different
path representations.
"""

from pathlib import Path
from typing import Any, Union

from ezstitcher.io.virtual_path import VirtualPath


def is_disk_path(path: Union[str, Path, VirtualPath, Any]) -> bool:
    """
    Determine if a path is a disk path.
    
    Return True if the given path ultimately resolves to a physical
    disk path, False otherwise (e.g. memory:// or unresolved VirtualPath).
    
    Args:
        path: The path to check
        
    Returns:
        True if the path is a disk path, False otherwise
    """
    # Handle VirtualPath objects directly
    if isinstance(path, VirtualPath):
        return path.to_physical_path() is not None
    
    # Convert to string for protocol checking
    path_str = str(path)
    
    # Check for memory:// protocol
    if path_str.startswith("memory://"):
        return False
    
    # Check if it has a to_physical_path method that returns a non-None value
    if hasattr(path, 'to_physical_path') and callable(path.to_physical_path):
        return path.to_physical_path() is not None
    
    # Default case: string or Path object is considered a disk path
    return isinstance(path, (str, Path))
