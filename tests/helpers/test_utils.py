"""
Utility functions for testing.
"""

from pathlib import Path
from typing import List, Optional


def assert_npy_files_exist(path: Path, min_count: int = 1) -> List[Path]:
    """
    Assert that the given directory contains at least min_count .npy files.

    Args:
        path: Directory path to check
        min_count: Minimum number of .npy files expected

    Returns:
        List of .npy files found

    Raises:
        AssertionError: If the directory doesn't exist or doesn't contain enough .npy files
    """
    assert path.exists(), f"Directory {path} does not exist"
    npy_files = list(path.glob("*.npy"))

    if len(npy_files) < min_count:
        # Get debug information
        files_str = str([f.name for f in npy_files]) if npy_files else "[]"

        # List all files in the directory
        all_files = list(path.glob("*"))
        all_files_str = str([f.name for f in all_files]) if all_files else "[]"

        # Raise assertion with detailed message
        raise AssertionError(
            f"Expected at least {min_count} .npy files in {path}, found {len(npy_files)}: {files_str}\n"
            f"All files in directory: {all_files_str}"
        )

    # Print the found files for debugging
    if len(npy_files) > 0:
        print(f"Found {len(npy_files)} .npy files in {path}:")
        for file in npy_files:
            print(f"  {file.name}")

    return npy_files


def assert_adapter_contains_keys(adapter, min_keys=1, prefix=None):
    """
    Assert that a storage adapter contains at least min_keys keys.

    Args:
        adapter: The storage adapter to check
        min_keys: Minimum number of keys expected
        prefix: Optional prefix to filter keys by

    Returns:
        List of keys found

    Raises:
        AssertionError: If the adapter doesn't contain enough keys
    """
    keys = adapter.list_keys()

    if prefix:
        keys = [k for k in keys if k.startswith(prefix)]

    if len(keys) < min_keys:
        # Get debug information
        adapter_type = type(adapter).__name__
        keys_str = str(keys) if keys else "[]"

        # Raise assertion with detailed message
        raise AssertionError(
            f"Expected at least {min_keys} keys in {adapter_type}, but found {len(keys)}: {keys_str}"
        )

    return keys
