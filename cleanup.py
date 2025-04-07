#!/usr/bin/env python3
"""
Cleanup script to remove unnecessary files from the ezstitcher repository.
This script keeps only the essential files needed to run the tests.
"""

import os
import shutil
import sys
from pathlib import Path

# Directories to explicitly remove
REMOVE_DIRS = [
    "docs",
    "examples",
    "htmlcov",
    "ezstitcher.egg-info",
    "tests/test_data_old",
    "node_modules"
]

# Symlinks to remove
REMOVE_SYMLINKS = [
    "IMX"
]

# Files to explicitly remove
REMOVE_FILES = [
    ".coverage",
    "check_compatibility.py",
    "check_pypi_compatibility.py",
    "check_wheel_availability.py",
    "CLASS_BASED_README.md",
    "class_based_test_output.txt",
    "Makefile",
    "original_test_output.txt",
    "package-lock.json",
    "package.json",
    "SETUP_GITHUB.md",
    "test_focus_detection.py",
    "test_full_workflow.py",
    "test_z_stack_workflow.py",
    "TESTING.md",
    "run_coverage.py",
    "files_to_delete.txt",
    "tests/test_focus_detection.py",
    "tests/test_focus_detector.py",
    "tests/test_image_processor.py",
    "tests/test_imports_new.py",
    "tests/test_imports.py",
    "tests/test_integration.py",
    "tests/test_main.py",
    "tests/test_stitcher_manager.py",
    "tests/test_synthetic_workflow.py",
    "tests/test_with_synthetic_data.py",
    "tests/test_z_stack_handler.py",
    "tests/test_z_stack_manager.py",
    "tests/test_z_stack_workflow.py",
    "utils/analyze_focus.py",
    "utils/compare_synthetic_data.py",
    "utils/visualize_synthetic_data.py"
]

def cleanup_repository(dry_run=True):
    """Remove unnecessary files from the repository."""
    root_dir = Path(".")
    
    # Collect files and directories to remove
    to_remove = []
    symlinks_to_remove = []
    
    # Add explicit directories to remove
    for dir_path in REMOVE_DIRS:
        path = root_dir / dir_path
        if path.exists() and path.is_dir() and not path.is_symlink():
            to_remove.append(path)
    
    # Add symlinks to remove
    for link_path in REMOVE_SYMLINKS:
        path = root_dir / link_path
        if path.exists() and path.is_symlink():
            symlinks_to_remove.append(path)
    
    # Add explicit files to remove
    for file_path in REMOVE_FILES:
        path = root_dir / file_path
        if path.exists() and path.is_file():
            to_remove.append(path)
    
    # Print summary
    print(f"Found {len(to_remove) + len(symlinks_to_remove)} items to remove:")
    for path in to_remove:
        rel_path = path.relative_to(root_dir)
        print(f"  {'DIR ' if path.is_dir() else 'FILE'} {rel_path}")
    for path in symlinks_to_remove:
        rel_path = path.relative_to(root_dir)
        print(f"  SYMLINK {rel_path}")
    
    if dry_run:
        print("\nThis was a dry run. No files were actually removed.")
        print("Run with --force to actually remove the files.")
        return
    
    # Remove files and directories
    print("\nRemoving files and directories...")
    for path in to_remove:
        if path.is_file():
            path.unlink()
            print(f"Removed file: {path.relative_to(root_dir)}")
        elif path.is_dir():
            shutil.rmtree(path)
            print(f"Removed directory: {path.relative_to(root_dir)}")
    
    # Remove symlinks
    for path in symlinks_to_remove:
        path.unlink()
        print(f"Removed symlink: {path.relative_to(root_dir)}")
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    dry_run = True
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        dry_run = False
    
    cleanup_repository(dry_run)
