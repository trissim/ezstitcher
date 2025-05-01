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
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Type, Mapping, Set, Literal

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

    def __init__(self, parser: FilenameParser,
                 metadata_handler: MetadataHandler,
                 file_manager: Optional[FileManager] = None):
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

    @classmethod
    def _discover_handlers(cls) -> Dict[str, Tuple[Type[FilenameParser], Type[MetadataHandler]]]:
        """
        Discover available microscope handlers.

        Returns:
            Dict mapping microscope type names to (parser_class, metadata_handler_class) tuples
        """
        # Use cached handlers if available
        if cls._handlers_cache is not None:
            return cls._handlers_cache

        # Import specific implementations
        from ezstitcher.microscopes.opera_phenix import OperaPhenixFilenameParser, OperaPhenixMetadataHandler
        from ezstitcher.microscopes.imagexpress import ImageXpressFilenameParser, ImageXpressMetadataHandler

        # Create mapping of microscope types to handler classes
        handlers = {
            'opera_phenix': (OperaPhenixFilenameParser, OperaPhenixMetadataHandler),
            'imagexpress': (ImageXpressFilenameParser, ImageXpressMetadataHandler),
        }

        # Add normalized versions (without underscores)
        normalized_handlers = {}
        for key, value in handlers.items():
            normalized_key = key.replace('_', '')
            if normalized_key != key:
                normalized_handlers[normalized_key] = value

        # Add normalized versions to the original dict
        handlers.update(normalized_handlers)

        # Cache the result
        cls._handlers_cache = handlers

        return handlers

    @property
    @abstractmethod
    def common_dirs(self) -> str:
        """Unique identifier for this microscope handler."""
        pass

    def post_workspace(self, workspace_path: Path, fm=FileManager(backend='disk'), width=3 ) -> Path:
        """
        Hook called after workspace symlink creation.
        Applies normalization logic followed by consistent filename padding.

        Returns:
            Path to the normalized image directory
        """
        image_dir = self._normalize_workspace(workspace_path, fm = fm)

        # Check if any subdirectory matches or contains common_dirs as substring
        for item in workspace_path.iterdir():
            if item.is_dir():
                if self.common_dirs in item.name:
                    # Process this directory
                    logger.info(f"Found directory matching '{self.common_dirs}': {item}")
                    # Add your processing logic here
                    image_dir = item
                    break

        # Ensure parser is provided
        parser = self.parser

        # Map original filenames to reconstructed filenames
        rename_map = {}
        for file_path in fm.list_image_files(image_dir):
            original_name = file_path.name

            # Parse the filename components
            metadata = parser.parse_filename(original_name)
            if not metadata:
                raise ValueError(f"Could not parse filename: {original_name}")

            # Reconstruct the filename with proper padding
            # If force_suffixes is True, add default values for missing components
            #if force_suffixes:
            # Default values for missing components
            site = metadata['site'] or 1
            channel = metadata['channel'] or 1
            z_index = metadata['z_index'] or 1
            #else:
            #    # Use existing values or None
            #    site = metadata.get('site')
            #    channel = metadata.get('channel')
            #    z_index = metadata.get('z_index')

            # Reconstruct the filename with proper padding
            new_name = parser.construct_filename(
                well=metadata['well'],
                site=site,
                channel=channel,
                z_index=z_index,
                extension=metadata['extension'],
                site_padding=width,
                z_padding=width
            )

            # Add to rename map if different
            if original_name != new_name:
                rename_map[original_name] = new_name

        # Perform the renaming
        for original_name, new_name in rename_map.items():
            original_path = image_dir / original_name
            new_path = image_dir / new_name

            try:
                original_path.rename(new_path)
                logger.debug(f"Renamed {original_path} to {new_path}")
            except Exception as e:
                logger.error(f"Error renaming {original_path} to {new_path}: {e}")

        return image_dir

    @abstractmethod
    def _normalize_workspace(self, workspace_path: Path, fm = FileManager(backend='disk')) -> Path:
        """
        Microscope-specific normalization logic before consistent renaming.
        Override in subclasses. Default implementation just locates the image dir.

        Args:
            workspace_path: Path to the symlinked workspace

        Returns:
            Path to the image directory (flattened and ready for renaming)
        """
        pass


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

    # Delegate metadata handling methods to metadata_handler with context

    def find_metadata_file(self, plate_path: Union[str, Path],
                          context: Optional['ProcessingContext'] = None) -> Optional[Path]:
        """Delegate to metadata handler with context."""
        return self.metadata_handler.find_metadata_file(plate_path, context)

    def get_grid_dimensions(self, plate_path: Union[str, Path],
                           context: Optional['ProcessingContext'] = None) -> Tuple[int, int]:
        """Delegate to metadata handler with context."""
        return self.metadata_handler.get_grid_dimensions(plate_path, context)

    def get_pixel_size(self, plate_path: Union[str, Path],
                      context: Optional['ProcessingContext'] = None) -> float:
        """Delegate to metadata handler with context."""
        return self.metadata_handler.get_pixel_size(plate_path, context)

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
    metadata_file_manager = FileManager(backend='disk')
    logger.debug("Created default disk-based FileManager for create_microscope_handler")

    logger.info("Using provided FileManager for microscope handler.")

    # Auto-detect microscope type if needed
    if microscope_type == 'auto':
        if not plate_folder:
            raise ValueError("plate_folder is required for auto-detection")

        plate_folder = Path(plate_folder)
        microscope_type = _auto_detect_microscope_type(plate_folder, metadata_file_manager)
        logger.info("Auto-detected microscope type: %s", microscope_type)

    # Get handler class mapping
    handlers = _import_handlers()

    # Get the appropriate handler class
    handler_class = handlers.get(microscope_type.lower())
    if not handler_class:
        raise ValueError(f"Unsupported microscope type: {microscope_type}")

    # Create and configure the handler
    logger.info("Creating %s", handler_class.__name__)
    handler = handler_class(file_manager)
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

        # Check for ImageXpress (.htd files)
        if file_manager.list_files(
            plate_folder, extensions={'.htd','.HTD'}, recursive=True
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
