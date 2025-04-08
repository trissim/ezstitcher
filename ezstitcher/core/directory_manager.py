"""
Directory manager for ezstitcher.

This module provides a class for managing directories.
"""

import re
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class DirectoryManager:
    """Manage directory creation and cleanup."""
    
    @staticmethod
    def ensure_directory(directory):
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory (str or Path): Directory path to ensure exists
            
        Returns:
            Path: Path object for the directory
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    
    @staticmethod
    def create_output_directories(parent_dir, plate_name, suffixes):
        """
        Create output directories for a plate.
        
        Args:
            parent_dir (str or Path): Parent directory
            plate_name (str): Name of the plate
            suffixes (dict): Dictionary mapping directory types to suffixes
            
        Returns:
            dict: Dictionary mapping directory types to Path objects
        """
        parent_dir = Path(parent_dir)
        directories = {}
        
        for dir_type, suffix in suffixes.items():
            dir_path = parent_dir / f"{plate_name}{suffix}"
            dir_path.mkdir(parents=True, exist_ok=True)
            directories[dir_type] = dir_path
        
        return directories
    
    @staticmethod
    def clean_temp_folders(parent_dir, base_name, keep_suffixes=None):
        """
        Clean up temporary folders created during processing.
        
        Args:
            parent_dir (str or Path): Parent directory
            base_name (str): Base name of the plate folder
            keep_suffixes (list, optional): List of suffixes to keep
        """
        parent_dir = Path(parent_dir)
        
        # Default suffixes to keep
        if keep_suffixes is None:
            keep_suffixes = ['_stitched']
        
        # Use a more flexible pattern matching approach
        # This will match any folder that starts with the base_name followed by an underscore
        for item in parent_dir.glob(f"{base_name}_*"):
            if not item.is_dir():
                continue
                
            # Check if this is a folder we want to keep
            should_keep = False
            for suffix in keep_suffixes:
                if item.name.endswith(suffix):
                    should_keep = True
                    break
            
            if not should_keep:
                try:
                    logger.info(f"Removing temporary folder: {item}")
                    shutil.rmtree(item)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary folder {item}: {e}")
    
    @staticmethod
    def find_wells(timepoint_dir):
        """
        Find all wells in the timepoint directory.
        
        Args:
            timepoint_dir (str or Path): Path to the TimePoint_1 directory
            
        Returns:
            list: List of well names (e.g., ['A01', 'A02', ...])
        """
        timepoint_dir = Path(timepoint_dir)
        well_pattern = re.compile(r'([A-Z]\d{2})_')
        wells = set()

        for file in timepoint_dir.glob("*.tif"):
            match = well_pattern.search(file.name)
            if match:
                wells.add(match.group(1))

        return sorted(list(wells))
