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
from typing import Dict, List, Optional, Union, Any, Tuple
from ezstitcher.core.image_locator import ImageLocator

logger = logging.getLogger(__name__)


class FilenameParser(ABC):
    """
    Abstract base class for parsing microscopy image filenames.
    """

    # Constants
    FILENAME_COMPONENTS = ['well', 'site', 'channel', 'z_index', 'extension']
    PLACEHOLDER_PATTERN = '{iii}'

    @classmethod
    @abstractmethod
    def can_parse(cls, filename: str) -> bool:
        """
        Check if this parser can parse the given filename.

        Args:
            filename (str): Filename to check

        Returns:
            bool: True if this parser can parse the filename, False otherwise
        """
        pass

    @abstractmethod
    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse a microscopy image filename to extract all components.

        Args:
            filename (str): Filename to parse

        Returns:
            dict or None: Dictionary with extracted components or None if parsing fails
        """
        pass

    @abstractmethod
    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None,
                          channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None,
                          extension: str = '.tif',
                          site_padding: int = 3, z_padding: int = 3) -> str:
        """
        Construct a filename from components.

        Args:
            well (str): Well ID (e.g., 'A01')
            site (int or str, optional): Site number or placeholder string (e.g., '{iii}')
            channel (int, optional): Channel/wavelength number
            z_index (int or str, optional): Z-index or placeholder string (e.g., '{zzz}')
            extension (str, optional): File extension
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 3)

        Returns:
            str: Constructed filename
        """
        pass

    def path_list_from_pattern(self, directory, pattern):
        """
        Get a list of filenames matching a pattern in a directory.

        Args:
            directory (str or Path): Directory to search
            pattern (str): Pattern to match with {iii} placeholder for site index

        Returns:
            list: List of matching filenames
        """
        directory = Path(directory)

        # Handle substitution of {series} if present (from Ashlar)
        if "{series}" in pattern:
            pattern = pattern.replace("{series}", "{iii}")

        # Parse the pattern to extract expected components
        pattern_metadata = self.parse_filename(pattern)
        if not pattern_metadata:
            logger.warning(f"Could not parse pattern: {pattern}")
            return []

        # Find all files in the directory
        matching_files = []
        all_images= ImageLocator.find_images_in_directory(directory)
        for file_path in all_images:
            if not file_path.is_file():
                continue

            # Parse the filename to extract its components
            file_metadata = self.parse_filename(file_path.name)
            if not file_metadata:
                continue

            # Check if all non-None components in the pattern match the file
            is_match = True
            for key, value in pattern_metadata.items():
                # Skip components that are None in the pattern (placeholders)
                if value is None:
                    continue

                # Check if the component exists in the file metadata and matches
                if key not in file_metadata or file_metadata[key] != value:
                    is_match = False
                    break

            if is_match:
                matching_files.append(file_path.name)

        # Log debug information
        logger.debug("Pattern: %s, Directory: %s, Files found: %d", pattern, directory, len(matching_files))
        if matching_files and logger.isEnabledFor(logging.DEBUG):
            logger.debug("First file: %s", matching_files[0])

        # Use natural sorting instead of lexicographical sorting
        return self._natural_sort(matching_files)

    def _natural_sort(self, file_list):
        """
        Sort filenames naturally, so that site numbers are sorted numerically.
        E.g., ["s1", "s10", "s2"] -> ["s1", "s2", "s10"]

        Args:
            file_list (list): List of filenames to sort

        Returns:
            list: Naturally sorted list of filenames
        """
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        return sorted(file_list, key=natural_sort_key)

    def group_patterns_by_component(self, patterns, component='channel', default_value='1'):
        """
        Group patterns by a specific component (channel, z_index, site, well, etc.)

        Args:
            patterns (list): List of patterns to group
            component (str): Component to group by (e.g., 'channel', 'z_index', 'site', 'well')
            default_value (str): Default value to use if component is not found

        Returns:
            dict: Dictionary mapping component values to patterns
        """
        grouped_patterns = {}

        for pattern in patterns:
            # Replace placeholder with dummy value for parsing
            pattern_with_dummy = pattern.replace(self.PLACEHOLDER_PATTERN, '001')
            metadata = self.parse_filename(pattern_with_dummy)

            if metadata and component in metadata and metadata[component] is not None:
                # Extract component value and convert to string
                value = str(metadata[component])
                if value not in grouped_patterns:
                    grouped_patterns[value] = []
                grouped_patterns[value].append(pattern)
            else:
                # Use default value if component not found
                if default_value not in grouped_patterns:
                    grouped_patterns[default_value] = []
                grouped_patterns[default_value].append(pattern)
                if component != 'z_index':  # z_index commonly defaults to 1, so don't log warning
                    logger.warning("Could not extract %s from pattern: %s", component, pattern)

        return grouped_patterns

    def auto_detect_patterns(self, folder_path, well_filter=None, extensions=None,
                           group_by=None, variable_components=None, flat=False):
        """
        Automatically detect image patterns in a folder.

        Args:
            folder_path (str or Path): Path to the folder
            well_filter (list): Optional list of wells to include
            extensions (list): Optional list of file extensions to include
            group_by (str, optional): Component to group patterns by (e.g., 'channel', 'z_index', 'well')
                                      If None, returns a flat list of patterns per well
            variable_components (list): List of components to make variable (e.g., ['site', 'z_index'])
            flat (bool): Deprecated. Use group_by=None instead.

        Returns:
            dict: Dictionary mapping wells to patterns (either grouped by component or flat list)
        """
        # Set default variable components if not provided
        if variable_components is None:
            variable_components = ['site']

        # Find all image files and group by well
        files_by_well = self._find_and_filter_images(folder_path, well_filter, extensions)

        if not files_by_well:
            return {}

        # Generate patterns for each well
        result = {}
        for well, files in files_by_well.items():
            # Generate patterns for this well
            patterns = self._generate_patterns_for_files(files, variable_components)

            # Return patterns based on requested format
            if flat or group_by is None:
                result[well] = patterns
            else:
                result[well] = self.group_patterns_by_component(patterns, component=group_by)

        return result

    def _find_and_filter_images(self, folder_path, well_filter=None, extensions=None):
        """
        Find all image files in a directory and filter by well.

        Args:
            folder_path (str or Path): Path to the folder
            well_filter (list): Optional list of wells to include
            extensions (list): Optional list of file extensions to include

        Returns:
            dict: Dictionary mapping wells to lists of image files
        """
        import time
        from collections import defaultdict
        from ezstitcher.core.image_locator import ImageLocator

        start_time = time.time()
        logger.info("Finding and filtering images in %s", folder_path)

        # Find all image files
        folder_path = Path(folder_path)
        extensions = extensions or ['.tif', '.TIF', '.tiff', '.TIFF']
        image_dir = ImageLocator.find_image_directory(folder_path)
        logger.info("Using image directory: %s", image_dir)

        # Check if this is an Opera Phenix dataset by checking for the remap_field_in_filename method
        is_opera_phenix = hasattr(self, 'remap_field_in_filename')

        # For Opera Phenix, use a more efficient file detection approach
        if is_opera_phenix:
            logger.info("Detected Opera Phenix dataset. Using optimized file detection.")
            image_paths = []

            # Check root directory first
            for ext in extensions:
                root_images = list(image_dir.glob(f"*{ext}"))
                image_paths.extend(root_images)

            # If no files in root, check immediate subdirectories
            if not image_paths:
                for subdir in image_dir.iterdir():
                    if subdir.is_dir():
                        for ext in extensions:
                            subdir_images = list(subdir.glob(f"*{ext}"))
                            image_paths.extend(subdir_images)
        else:
            # For other microscopes, use the standard approach but limit recursion depth
            image_paths = ImageLocator.find_images_in_directory(image_dir, extensions, recursive=True)

        if not image_paths:
            logger.warning("No image files found in %s", folder_path)
            return {}

        logger.info("Found %d image files in %.2f seconds. Grouping by well...",
                   len(image_paths), time.time() - start_time)
        group_start = time.time()

        # Group files by well
        files_by_well = defaultdict(list)
        for img_path in image_paths:
            metadata = self.parse_filename(img_path.name)
            if not metadata:
                continue

            well = metadata['well']
            # Case-insensitive well filtering
            if not well_filter or any(well.lower() == w.lower() for w in well_filter):
                files_by_well[well].append(img_path)

        logger.info("Grouped %d files into %d wells in %.2f seconds",
                   len(image_paths), len(files_by_well), time.time() - group_start)
        return files_by_well

    def _generate_patterns_for_files(self, files, variable_components):
        """
        Generate patterns for a list of files with specified variable components.

        Args:
            files (list): List of file paths
            variable_components (list): List of components to make variable

        Returns:
            list: List of patterns
        """
        from collections import defaultdict

        # Get unique combinations of non-variable components
        component_combinations = defaultdict(list)

        for file_path in files:
            metadata = self.parse_filename(file_path.name)
            if not metadata:
                continue

            # Create a key based on non-variable components
            key_parts = []
            for comp in self.FILENAME_COMPONENTS:
                if comp in metadata and comp not in variable_components and metadata[comp] is not None:
                    key_parts.append(f"{comp}={metadata[comp]}")

            key = ",".join(key_parts)
            component_combinations[key].append((file_path, metadata))

        # Generate patterns for each combination
        patterns = []
        for _, files_metadata in component_combinations.items():
            if not files_metadata:
                continue

            # Use the first file's metadata as a template
            _, template_metadata = files_metadata[0]

            # Create pattern by replacing variable components with placeholders
            pattern_args = {}
            for comp in self.FILENAME_COMPONENTS:
                if comp in template_metadata:  # Only include components that exist in the metadata
                    if comp in variable_components:
                        pattern_args[comp] = self.PLACEHOLDER_PATTERN
                    else:
                        pattern_args[comp] = template_metadata[comp]

            # Construct the pattern
            pattern = self.construct_filename(
                well=pattern_args['well'],
                site=pattern_args.get('site'),
                channel=pattern_args.get('channel'),
                z_index=pattern_args.get('z_index'),
                extension=pattern_args.get('extension', '.tif')
            )

            patterns.append(pattern)

        return patterns


class MetadataHandler(ABC):
    """
    Abstract base class for handling microscope metadata.
    """

    @abstractmethod
    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """
        Find the metadata file for a plate.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Path to the metadata file, or None if not found
        """
        pass

    @abstractmethod
    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """
        Get grid dimensions for stitching from metadata.

        Args:
            plate_path: Path to the plate folder

        Returns:
            (grid_size_x, grid_size_y)
        """
        pass

    @abstractmethod
    def get_pixel_size(self, plate_path: Union[str, Path]) -> Optional[float]:
        """
        Get the pixel size from metadata.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Pixel size in micrometers, or None if not available
        """
        pass


class MicroscopeHandler:
    """Composed class for handling microscope-specific functionality."""

    DEFAULT_MICROSCOPE = 'ImageXpress'
    _handlers_cache = None

    @classmethod
    def _discover_handlers(cls):
        """Discover all microscope handlers from the microscopes subpackage."""
        if cls._handlers_cache:
            return cls._handlers_cache

        import importlib, inspect, pkgutil
        from ezstitcher.microscopes import __path__ as microscopes_path

        handlers = {}

        # Find all modules in the microscopes package
        for _, module_name, _ in pkgutil.iter_modules(microscopes_path, 'ezstitcher.microscopes.'):
            try:
                module = importlib.import_module(module_name)

                # Find FilenameParser implementations in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ != module.__name__ or not issubclass(obj, FilenameParser) or obj == FilenameParser:
                        continue

                    # Extract microscope type from class name
                    microscope_type = name.replace('FilenameParser', '')

                    # Look for matching MetadataHandler
                    handler_name = f"{microscope_type}MetadataHandler"
                    handler_class = getattr(module, handler_name, None)

                    if handler_class and issubclass(handler_class, MetadataHandler):
                        handlers[microscope_type] = (obj, handler_class)
            except Exception as e:
                logger.debug(f"Error inspecting module {module_name}: {e}")

        cls._handlers_cache = handlers
        return handlers

    def __init__(self, plate_folder=None, parser=None, metadata_handler=None, microscope_type='auto'):
        """Initialize with plate folder and optional components."""
        self.plate_folder = Path(plate_folder) if plate_folder else None

        if parser is None or metadata_handler is None:
            detected_type = self._detect_microscope_type(microscope_type)
            self.parser, self.metadata_handler = self._create_handlers(detected_type, parser, metadata_handler)
        else:
            self.parser, self.metadata_handler = parser, metadata_handler

    def _detect_microscope_type(self, microscope_type):
        """Detect microscope type from files or use specified type."""
        if microscope_type.lower() != 'auto' or not self.plate_folder:
            return microscope_type if microscope_type.lower() != 'auto' else self.DEFAULT_MICROSCOPE

        try:
            # Get sample files and test each parser
            sample_files = ImageLocator.find_images_in_directory(self.plate_folder)[:10]

            if not sample_files:
                return self.DEFAULT_MICROSCOPE

            matches =  {}
            for name, (parser_class, _) in self._discover_handlers().items():
                matches[name] = 0
                for f in sample_files:
                    if parser_class.can_parse(f.name):
                        matches[name] += 1


            best_match = max(matches.items(), key=lambda x: x[1]) if matches else (self.DEFAULT_MICROSCOPE, 0)
            if best_match[1] > 0:
                logger.info(f"Auto-detected {best_match[0]} format ({best_match[1]}/{len(sample_files)} files matched)")
                return best_match[0]

            return self.DEFAULT_MICROSCOPE
        except Exception as e:
            logger.error(f"Error during auto-detection: {e}")
            return self.DEFAULT_MICROSCOPE

    def _create_handlers(self, microscope_type, parser=None, metadata_handler=None):
        """Create parser and metadata handler for the specified microscope type."""
        handlers = self._discover_handlers()
        parser_class, handler_class = handlers.get(microscope_type, handlers.get(self.DEFAULT_MICROSCOPE, (None, None)))

        return (parser or (parser_class() if parser_class else None),
                metadata_handler or (handler_class() if handler_class else None))

    # Delegate filename parsing methods to parser

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
        return self.parser.path_list_from_pattern(directory, pattern)

    # Delegate metadata handling methods to metadata_handler

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """Delegate to metadata handler."""
        return self.metadata_handler.find_metadata_file(plate_path)

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """Delegate to metadata handler."""
        return self.metadata_handler.get_grid_dimensions(plate_path)

    def get_pixel_size(self, plate_path: Union[str, Path]) -> Optional[float]:
        """Delegate to metadata handler."""
        return self.metadata_handler.get_pixel_size(plate_path)

    def init_workspace(self, plate_path: Union[str, Path], workspace_path: Union[str, Path]) -> int:
        """Mirror the plate directory and create symlinks to all files.

        For Opera Phenix, also renames symlinks based on field indices from Index.xml.

        Args:
            plate_path: Path to the source plate directory
            workspace_path: Path to the target workspace directory

        Returns:
            int: Number of symlinks created
        """
        # Import here to avoid circular imports
        from ezstitcher.core.file_system_manager import FileSystemManager
        # Import time for performance logging
        import time
        import sys

        plate_path = Path(plate_path)
        workspace_path = Path(workspace_path)
        self.plate_folder = workspace_path

        print(f"Starting to mirror directory from {plate_path} to {workspace_path}")
        start_time = time.time()

        # Create basic directory structure with symlinks
        print("Creating symlinks...")
        sys.stdout.flush()  # Force output to be displayed immediately
        symlink_count = FileSystemManager.mirror_directory_with_symlinks(plate_path, workspace_path)

        print(f"Mirroring completed in {time.time() - start_time:.2f} seconds. Created {symlink_count} symlinks.")
        sys.stdout.flush()  # Force output to be displayed immediately

        # Check if the parser has a remap_field_in_filename method (Opera Phenix specific)
        if hasattr(self.parser, 'remap_field_in_filename'):
            print("Detected Opera Phenix dataset. Checking for metadata file...")
            sys.stdout.flush()  # Force output to be displayed immediately

            # Find metadata file (Index.xml for Opera Phenix)
            metadata_file = self.metadata_handler.find_metadata_file(plate_path)


            ### SMELLY: this responsibility should be dedicated to an operaphenix specific class ###
            if metadata_file and hasattr(self.metadata_handler, 'create_xml_parser'):
                print(f"Found metadata file: {metadata_file}. Starting field remapping.")
                sys.stdout.flush()  # Force output to be displayed immediately
                remap_start_time = time.time()

                # Create XML parser using the metadata file
                print("Creating XML parser...")
                sys.stdout.flush()  # Force output to be displayed immediately
                xml_parser = self.metadata_handler.create_xml_parser(metadata_file)

                # Find image files in the workspace - limit to direct files in the workspace
                # rather than searching all subdirectories
                print("Finding image files in workspace...")
                sys.stdout.flush()  # Force output to be displayed immediately

                # Find the image directory (handles both root and subdirectory cases)
                image_dir = ImageLocator.find_image_directory(workspace_path)
                print(f"Found image directory: {image_dir}")
                sys.stdout.flush()  # Force output to be displayed immediately

                # Find all image files in the directory using default extensions
                image_files = ImageLocator.find_images_in_directory(
                    image_dir,
                    recursive=True
                )

                total_files = len(image_files)
                print(f"Found {total_files} image files. Remapping field IDs...")
                sys.stdout.flush()  # Force output to be displayed immediately

                # Get field ID mapping
                print("Getting field ID mapping from XML...")
                sys.stdout.flush()
                field_mapping = xml_parser.get_field_id_mapping()

                # Print the first 20 entries of the field mapping
                print("Field ID mapping (first 20 entries):")
                sorted_field_ids = sorted(field_mapping.keys())[:20]  # Get first 20 field IDs
                for field_id in sorted_field_ids:
                    new_field_id = field_mapping[field_id]
                    print(f"  Field {field_id:3d} -> {new_field_id:3d}")

                # Create a temporary subfolder for all files that need to be renamed
                temp_folder_name = f"temp_rename_{uuid.uuid4().hex[:8]}"
                # Use the image_dir we already found
                temp_folder = image_dir / temp_folder_name
                #temp_folder = workspace_path / temp_folder_name
                temp_folder.mkdir(exist_ok=True)
                print(f"Created temporary folder: {temp_folder}")
                sys.stdout.flush()

                ### SMELLY: we should log everything and reuse it
                # Calculate progress reporting interval based on total number of files
                # For large datasets, report less frequently
                if total_files > 10000:
                    report_interval = 1000  # Report every 1000 files for very large datasets
                elif total_files > 1000:
                    report_interval = 100   # Report every 100 files for large datasets
                else:
                    report_interval = 10    # Report every 10 files for small datasets


                # Remap field IDs in filenames
                additional_symlinks = 0
                remapped_files = 0
                skipped_files = 0
                renamed_files = {}

                print(f"Starting to process {total_files} files...")
                sys.stdout.flush()

                # For each image in the folder:
                # 1. Get the field number from the filename
                # 2. Use the field number to find the new field number from the mapping
                # 3. Generate a new filename using the filename parser
                # 4. Move the image to a temporary folder
                processed_files = 0
                for image_file in image_files:
                    processed_files += 1

                    # Log progress at appropriate intervals
                    if processed_files > 0 and processed_files % report_interval == 0:
                        percent_done = (processed_files/total_files)*100
                        msg = f"Processed {processed_files}/{total_files} files"
                        msg += f" ({percent_done:.1f}%)"
                        print(msg)
                        sys.stdout.flush()  # Force output to be displayed immediately

                    # Parse the filename to get metadata
                    metadata = self.parser.parse_filename(image_file.name)

                    if metadata and 'site' in metadata and metadata['site'] is not None:
                        # Get the field ID (site number)
                        field_id = metadata['site']

                        # Find the new field ID from the mapping
                        new_field_id = field_mapping.get(field_id, field_id)

                        # Create a new filename with the remapped field ID
                        new_filename = self.parser.construct_filename(
                            well=metadata['well'],
                            site=new_field_id,  # Use the remapped field ID
                            channel=metadata['channel'],
                            z_index=metadata['z_index'],
                            extension=metadata['extension'],
                            site_padding=3,
                            z_padding=3
                        )

                        temp_path = temp_folder / new_filename
                        shutil.move(str(image_dir / image_file), str(temp_path))
                        renamed_files[new_filename] = temp_path

                symlink_count += additional_symlinks
                print(f"Field remapping completed in {time.time() - remap_start_time:.2f} seconds.")
                print(f"Remapped {remapped_files} files")
                print(f"Created {additional_symlinks} new symlinks")
                sys.stdout.flush()  # Force output to be displayed immediately

                # Move the renamed symlinks back to the original location with their new names
                print(f"Moving {len(renamed_files)} renamed files back to original location...")
                sys.stdout.flush()
                move_start_time = time.time()

                for new_filename, temp_path in renamed_files.items():
                    try:
                        # Move the file back to the original location with the new filename
                        dest_path = image_dir / new_filename
                        shutil.move(str(temp_path), str(dest_path))
                    except Exception as e:
                        print(f"Error moving {temp_path} to {dest_path}: {e}")

                print(f"Moved files back in {time.time() - move_start_time:.2f} seconds.")
                sys.stdout.flush()

                # Clean up the temporary folder
                try:
                    if temp_folder.exists():
                        # Check if the folder is empty before removing
                        remaining_files = list(temp_folder.iterdir())
                        if remaining_files:
                            num_remaining = len(remaining_files)
                            print(f"Warning: {num_remaining} files remain in the temp folder")
                            print("These files may have had conflicts during renaming.")
                        shutil.rmtree(temp_folder)
                except Exception as e:
                    print(f"Error removing temporary folder: {e}")

def create_microscope_handler(microscope_type: str = 'auto', **kwargs) -> MicroscopeHandler:
    """Create the appropriate microscope handler."""
    return MicroscopeHandler(microscope_type=microscope_type, **kwargs)
