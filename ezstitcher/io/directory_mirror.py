"""
Directory mirroring utilities for ezstitcher.

This module provides functions for mirroring directory structures using symlinks,
extracted from the old FileSystemManager to follow single responsibility principle.
"""

import os
import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)


class OverwriteStrategy(str, Enum):
    """Strategy for handling existing target directories when mirroring."""
    REPLACE = "replace"  # Delete and recreate target directory
    SKIP = "skip"        # Skip if target exists
    MERGE = "merge"      # Keep existing target and add/update files


def _ensure_source_directory_exists(source_dir: Path) -> bool:
    """
    Validate that source directory exists.

    Args:
        source_dir: Path to the source directory

    Returns:
        bool: True if directory exists, False otherwise
    """
    if not source_dir.is_dir():
        logger.error(f"Source directory not found: {source_dir}")
        return False
    return True


def _prepare_target_directory(target_dir: Path, strategy: OverwriteStrategy) -> bool:
    """
    Prepare target directory based on the selected strategy.

    Args:
        target_dir: Path to the target directory
        strategy: Strategy for handling existing directories

    Returns:
        bool: True if preparation succeeded, False otherwise
    """
    # If target doesn't exist, create it regardless of strategy
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        return True

    # Handle existing directory based on strategy
    if strategy == OverwriteStrategy.REPLACE:
        logger.info(f"Removing existing target directory: {target_dir}")
        try:
            shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error removing target directory {target_dir}: {e}")
            logger.info("Continuing without removing the directory...")
            # We'll continue even if removal fails
    elif strategy == OverwriteStrategy.SKIP:
        logger.info(f"Target directory exists, skipping: {target_dir}")
        return False
    # For MERGE strategy, we keep the directory as is

    # Ensure directory exists (needed for MERGE or if REPLACE failed)
    target_dir.mkdir(parents=True, exist_ok=True)
    return True


def _create_symlink(source_item: Path, target_path: Path) -> bool:
    """
    Create a symlink from source to target with error handling.

    Args:
        source_item: Source file path
        target_path: Target symlink path

    Returns:
        bool: True if symlink was created, False otherwise
    """
    try:
        # Remove existing symlink if it exists
        if target_path.exists():
            target_path.unlink()

        # Create new symlink
        os.symlink(source_item.resolve(), target_path)
        return True
    except Exception as e:
        logger.error(f"Error creating symlink from {source_item} to {target_path}: {e}")
        return False


def _process_directory_items(
    source_dir: Path,
    target_dir: Path,
    recursive: bool,
    strategy: OverwriteStrategy,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    verbose: bool = False
) -> int:
    """
    Process all items in a directory, creating symlinks for files and
    recursively processing subdirectories if requested.

    Args:
        source_dir: Source directory path
        target_dir: Target directory path
        recursive: Whether to recursively process subdirectories
        strategy: Strategy for handling existing items
        progress_callback: Optional callback for progress reporting
        verbose: Whether to log verbose progress information

    Returns:
        int: Number of symlinks created
    """
    symlinks_created = 0

    try:
        # Get all items in the source directory
        items = list(source_dir.iterdir())
        total_items = len(items)
        
        if verbose:
            logger.info(f"Found {total_items} items in {source_dir}")
        
        # Report initial progress
        if progress_callback:
            progress_callback(0, total_items, str(source_dir))

        # Process all items
        for i, item in enumerate(items):
            # Log progress periodically
            if verbose and i > 0 and i % 100 == 0:
                logger.info(f"Processed {i}/{total_items} items ({(i/total_items)*100:.1f}%)")
            
            # Report progress
            if progress_callback:
                progress_callback(i, total_items, str(source_dir))

            # Handle subdirectories
            if item.is_dir() and recursive:
                sub_target_dir = target_dir / item.name
                symlinks_created += mirror_directory_with_symlinks(
                    item, 
                    sub_target_dir, 
                    recursive=recursive,
                    strategy=OverwriteStrategy.MERGE,  # Always merge for subdirectories
                    progress_callback=progress_callback,
                    verbose=verbose
                )
                continue

            # Skip non-files
            if not item.is_file():
                continue

            # Create symlink
            target_path = target_dir / item.name
            if _create_symlink(item, target_path):
                symlinks_created += 1

        # Report completion
        if verbose:
            logger.info(f"Completed processing all {total_items} items in {source_dir}")
        
        # Final progress report
        if progress_callback:
            progress_callback(total_items, total_items, str(source_dir))

    except Exception as e:
        logger.error(f"Error processing directory {source_dir}: {e}")

    return symlinks_created


def mirror_directory_with_symlinks(
    source_dir: Union[str, Path],
    target_dir: Union[str, Path],
    recursive: bool = True,
    strategy: Union[str, OverwriteStrategy] = OverwriteStrategy.REPLACE,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    verbose: bool = False
) -> int:
    """
    Mirror a directory structure from source to target and create symlinks to all files.
    
    This function creates a mirror of the source directory structure in the target
    directory, with symlinks to all files instead of copying the actual content.
    The behavior for existing target directories is controlled by the strategy parameter.

    Args:
        source_dir: Path to the source directory to mirror
        target_dir: Path to the target directory where the mirrored structure will be created
        recursive: Whether to recursively mirror subdirectories. Defaults to True.
        strategy: Strategy for handling existing target directory. Can be a string ('replace', 
                 'skip', 'merge') or an OverwriteStrategy enum value. Defaults to REPLACE.
        progress_callback: Optional callback function for progress reporting. The callback 
                          receives (current_count, total_count, current_directory).
        verbose: Whether to log verbose progress information. Defaults to False.

    Returns:
        int: Number of symlinks created
    """
    # Convert paths to Path objects
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Convert string strategy to enum if needed
    if isinstance(strategy, str):
        try:
            strategy = OverwriteStrategy(strategy.lower())
        except ValueError:
            logger.warning(f"Invalid strategy '{strategy}', using REPLACE")
            strategy = OverwriteStrategy.REPLACE
    
    # Ensure source directory exists
    if not _ensure_source_directory_exists(source_dir):
        return 0
    
    # Prepare target directory based on strategy
    if not _prepare_target_directory(target_dir, strategy):
        return 0
    
    # Process directory items
    return _process_directory_items(
        source_dir, 
        target_dir, 
        recursive, 
        strategy, 
        progress_callback, 
        verbose
    )
