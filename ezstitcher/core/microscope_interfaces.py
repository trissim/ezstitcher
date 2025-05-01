"""
Microscope interfaces for ezstitcher.

This module provides abstract base classes for handling microscope-specific
functionality, including filename parsing and metadata handling.
"""

import logging
import os
import re
import shutil
import sys
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Type, Mapping

# Removed: from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.io.filemanager import FileManager
from ezstitcher.core.microscope_base import FilenameParser, MetadataHandler

# Import handler classes at module level but with lazy loading to avoid circular imports
def _import_handlers():
    from ezstitcher.microscopes.opera_phenix import OperaPhenixHandler
    from ezstitcher.microscopes.imagexpress import ImageXpressHandler
    return {
        'opera_phenix': OperaPhenixHandler,
        'operaphenix': OperaPhenixHandler,
        'imagexpress': ImageXpressHandler,
        'imx': ImageXpressHandler,
    }

logger = logging.getLogger(__name__)


class MicroscopeHandler(ABC):
    """Composed class for handling microscope-specific functionality."""

    DEFAULT_MICROSCOPE = 'ImageXpress'
    _handlers_cache = None

    # Updated __init__ signature
    def __init__(self, parser: FilenameParser,
                 metadata_handler: MetadataHandler,
                 file_manager: Optional[FileManager] = None): # Inject FileManager
        """
        Initialize the microscope handler.

        Args:
            parser: Parser for microscopy filenames.
            metadata_handler: Handler for microscope metadata.
            file_manager: FileManager instance. If None, a disk-based FileManager is created.
        """
        self.parser = parser
        self.metadata_handler = metadata_handler

        if file_manager is None:
            file_manager = FileManager(backend='disk')
            logger.debug("Created default disk-based FileManager for MicroscopeHandler")

        self.file_manager = file_manager
        self.plate_folder: Optional[Path] = None # Store workspace path if needed by methods

    def post_workspace(self, workspace_path: Path, fm=FileManager(backend='disk') ) -> Path:
        """
        Hook called after workspace symlink creation.
        Applies normalization logic followed by consistent filename padding.

        Returns:
            Path to the normalized image directory
        """
        image_dir = self._normalize_workspace(workspace_path, fm = fm)

        fm.rename_files_with_consistent_padding(
            directory=image_dir,
            parser=self.parser,
            width=3,
            force_suffixes=True
        )

        return image_dir


    def _normalize_workspace(self, workspace_path: Path, fm = FileManager(backend='disk')) -> Path:
        """
        Microscope-specific normalization logic before consistent renaming.
        Override in subclasses. Default implementation just locates the image dir.

        Args:
            workspace_path: Path to the symlinked workspace

        Returns:
            Path to the image directory (flattened and ready for renaming)
        """
        return fm.find_image_directory(workspace_path)


    # Delegate methods to parser
    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Delegate to parser."""
        return self.parser.parse_filename(filename)

    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None,
                          channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None,
                          extension: str = '.tif',
                          site_padding: int = 3, z_padding: int = 3) -> str:
        """Delegate to parser."""
        return self.parser.construct_filename(
            well, site, channel, z_index, extension, site_padding, z_padding
        )

    def auto_detect_patterns(self, folder_path, well_filter=None, extensions=None,
                           group_by='channel', variable_components=None, flat=False):
        """Delegate to parser."""
        # Note: This method internally calls _find_and_filter_images which still uses FSM temporarily
        return self.parser.auto_detect_patterns(
            folder_path,
            well_filter=well_filter,
            extensions=extensions,
            group_by=group_by,
            variable_components=variable_components,
            flat=flat
        )

    def path_list_from_pattern(self, directory, pattern):
        """Delegate to parser."""
        # Note: This method internally uses FSM temporarily
        return self.parser.path_list_from_pattern(directory, pattern)

    # Delegate metadata handling methods to metadata_handler

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """Delegate to metadata handler."""
        # Metadata handler now uses its own injected file_manager
        return self.metadata_handler.find_metadata_file(plate_path)

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """Delegate to metadata handler."""
        return self.metadata_handler.get_grid_dimensions(plate_path)

    def get_pixel_size(self, plate_path: Union[str, Path]) -> float:
        """Delegate to metadata handler."""
        return self.metadata_handler.get_pixel_size(plate_path)

# Factory function
def create_microscope_handler(microscope_type: str = 'auto',
                              plate_folder: Optional[Union[str, Path]] = None,
                              file_manager: Optional[FileManager] = None) -> MicroscopeHandler:
    """
    Factory function to create a microscope handler.

    Args:
        microscope_type: 'auto', 'imagexpress', 'opera_phenix'.
        plate_folder: Required for 'auto' detection.
        file_manager: FileManager instance. Must be provided.

    Returns:
        An initialized MicroscopeHandler instance.

    Raises:
        ValueError: If file_manager is None or if microscope_type cannot be determined.
    """
    if file_manager is None:
        file_manager = FileManager(backend='disk')
        logger.debug("Created default disk-based FileManager for create_microscope_handler")

    logger.info("Using provided FileManager for microscope handler.")

    # Auto-detect microscope type if needed
    if microscope_type == 'auto':
        if not plate_folder:
            raise ValueError("plate_folder is required for auto-detection")

        plate_folder = Path(plate_folder)
        microscope_type = _auto_detect_microscope_type(plate_folder, file_manager)

    # Get handler class mapping
    handlers = _import_handlers()

    # Get the appropriate handler class
    handler_class = handlers.get(microscope_type.lower())
    if not handler_class:
        raise ValueError(f"Unsupported microscope type: {microscope_type}")

    # Create and configure the handler
    logger.info("Creating %s", handler_class.__name__)
    handler = handler_class()
    handler.file_manager = file_manager

    return handler


def _auto_detect_microscope_type(plate_folder: Path, file_manager: FileManager) -> str:
    """
    Auto-detect microscope type based on files in the plate folder.

    Args:
        plate_folder: Path to the plate folder
        file_manager: FileManager instance

    Returns:
        Detected microscope type as string

    Raises:
        ValueError: If microscope type cannot be determined
    """
    try:
        # Check for Opera Phenix (Index.xml)
        if file_manager.find_file_recursive(plate_folder, "Index.xml"):
            logger.info("Auto-detected Opera Phenix microscope type.")
            return 'operaphenix'

        if file_manager.find_file_recursive(plate_folder, "Index.xml"):
            logger.info("Auto-detected Opera Phenix microscope type.")
            return 'operaphenix'
        # Check for ImageXpress (.htd files)
        if file_manager.list_files(
            plate_folder, extensions={'.htd'}, recursive=True
        ):
            logger.info("Auto-detected ImageXpress microscope type.")
            return 'imagexpress'

        # No known microscope type detected
        msg = f"Could not auto-detect microscope type in {plate_folder}. " \
              "Neither Index.xml (Opera) nor .HTD (ImageXpress) found."
        logger.error(msg)
        raise ValueError(msg)

    except Exception as e:
        raise ValueError(f"Error during auto-detection in {plate_folder}: {e}") from e
