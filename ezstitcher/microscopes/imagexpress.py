"""
ImageXpress microscope implementations for ezstitcher.

This module provides concrete implementations of FilenameParser and MetadataHandler
for ImageXpress microscopes.
"""

import os
import re
import logging
import tifffile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

#from ezstitcher.core.microscope_interfaces import FilenameParser, MetadataHandler
from ezstitcher.core.microscope_base import FilenameParser, MetadataHandler
from ezstitcher.core.microscope_interfaces import MicroscopeHandler

# Removed: from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.io.filemanager import FileManager # Added

logger = logging.getLogger(__name__)

class ImageXpressHandler(MicroscopeHandler):
    """
    MicroscopeHandler implementation for Molecular Devices ImageXpress systems.

    This handler binds the ImageXpress filename parser and metadata handler,
    enforcing semantic alignment between file layout parsing and metadata resolution.
    """

    def __init__(self):
        super().__init__(
            parser=ImageXpressFilenameParser(),
            metadata_handler=ImageXpressMetadataHandler()
        )

    @property
    def common_dirs(self) -> str:
        """Subdirectory names commonly used by ImageXpress"""
        return 'TimePoint'

    def _normalize_workspace(self, workspace_path: Path, fm=FileManager(backend='disk')) -> Path:
        """
        Flattens the Z-step folder structure and renames image files for
        consistent padding and Z-plane resolution.

        Args:
            workspace_path: Path to the symlinked workspace
            fm: FileManager instance for file operations

        Returns:
            Path to the flattened image directory.
        """
        # Find all subdirectories in workspace
        subdirs = [d for d in workspace_path.iterdir() if d.is_dir()]
        
        # Check if any subdirectory contains common_dirs string
        common_dir_found = False
        for subdir in subdirs:
            if self.common_dirs in subdir.name:
                # Found a matching directory, process it
                self._flatten_zsteps(subdir, fm)
                common_dir_found = True
        
        # If no common directory found, process the workspace directly
        if not common_dir_found:
            self._flatten_zsteps(workspace_path, fm)
        
        # Return the image directory
        return workspace_path

    def _flatten_zsteps(self, directory: Path, fm: FileManager):
        """
        Process Z-step folders in the given directory.
        
        Args:
            directory: Directory that might contain Z-step folders
            fm: FileManager instance for file operations
        """
        # Check for Z step folders
        zstep_pattern = re.compile(r"ZStep[_-]?(\d+)", re.IGNORECASE)
        
        potential_z_folders = [
            d for d in directory.iterdir()
            if d.is_dir() and zstep_pattern.search(d.name)
        ]
        
        if not potential_z_folders:
            logger.info(f"No Z step folders found in {directory}. Skipping flattening.")
            return
        
        # Sort Z folders by index
        z_folders = sorted([
            (int(zstep_pattern.search(d.name).group(1)), d)
            for d in potential_z_folders
        ], key=lambda x: x[0])
        
        # Process each Z folder
        for z_index, z_dir in z_folders:
            for img_file in z_dir.glob("*"):
                if not img_file.is_file():
                    continue
                
                # Parse the original filename to extract components
                components = self.parser.parse_filename(img_file.name)
                
                if not components:
                    continue
                
                # Update the z_index in the components
                components['z_index'] = z_index
                
                # Use the parser to construct a new filename with the updated z_index
                new_name = self.parser.construct_filename(
                    well=components['well'],
                    site=components['site'],
                    channel=components['channel'],
                    z_index=z_index,
                    extension=components['extension']
                )
                
                # Move to the parent directory
                new_path = directory / new_name
                
                try:
                    fm.rename(img_file, new_path)
                    logger.debug(f"Moved {img_file} to {new_path}")
                except Exception as e:
                    logger.warning(f"Failed to move {img_file} to {new_path}: {e}")
        
        # Remove Z folders after all files have been moved
        for _, z_dir in z_folders:
            try:
                fm.remove_directory(z_dir)
                logger.debug(f"Removed Z-step folder: {z_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove Z-step folder {z_dir}: {e}")


class ImageXpressFilenameParser(FilenameParser):
    """
    Parser for ImageXpress microscope filenames.

    Handles standard ImageXpress format filenames like:
    - A01_s001_w1.tif
    - A01_s1_w1_z1.tif
    """

    # Regular expression pattern for ImageXpress filenames
    _pattern = re.compile(r'(?:.*?_)?([A-Z]\d+)(?:_s(\d+|\{[^\}]*\}))?(?:_w(\d+|\{[^\}]*\}))?(?:_z(\d+|\{[^\}]*\}))?(\.\w+)?$')

    @classmethod
    def can_parse(cls, filename: str) -> bool:
        """
        Check if this parser can parse the given filename.

        Args:
            filename (str): Filename to check

        Returns:
            bool: True if this parser can parse the filename, False otherwise
        """
        # Extract just the basename
        basename = os.path.basename(filename)
        # Check if the filename matches the ImageXpress pattern
        return bool(cls._pattern.match(basename))

    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse an ImageXpress filename to extract all components, including extension.

        Args:
            filename (str): Filename to parse

        Returns:
            dict or None: Dictionary with extracted components or None if parsing fails
        """
        basename = os.path.basename(filename)
        match = self._pattern.match(basename)

        if match:
            well, site_str, channel_str, z_str, ext = match.groups()

            #handle {} place holders
            parse_comp = lambda s: None if not s or '{' in s else int(s)
            site = parse_comp(site_str)
            channel = parse_comp(channel_str)
            z_index = parse_comp(z_str)

            # Use the parsed components in the result
            result = {
                'well': well,
                'site': site,
                'channel': channel,
                'z_index': z_index,
                'extension': ext if ext else '.tif'  # Default if somehow empty
            }

            return result
        else:
            logger.debug(f"Could not parse ImageXpress filename: {filename}")
            return None

    def construct_filename(self, well: str, site: Optional[Union[int, str]] = None,
                          channel: Optional[int] = None,
                          z_index: Optional[Union[int, str]] = None,
                          extension: str = '.tif',
                          site_padding: int = 3, z_padding: int = 3) -> str:
        """
        Construct an ImageXpress filename from components, only including parts if provided.

        Args:
            well (str): Well ID (e.g., 'A01')
            site (int or str, optional): Site number or placeholder string (e.g., '{iii}')
            channel (int, optional): Channel number
            z_index (int or str, optional): Z-index or placeholder string (e.g., '{zzz}')
            extension (str, optional): File extension
            site_padding (int, optional): Width to pad site numbers to (default: 3)
            z_padding (int, optional): Width to pad Z-index numbers to (default: 3)

        Returns:
            str: Constructed filename
        """
        if not well:
            raise ValueError("Well ID cannot be empty or None.")

        parts = [well]
        if site is not None:
            if isinstance(site, str):
                # If site is a string (e.g., '{iii}'), use it directly
                parts.append(f"_s{site}")
            else:
                # Otherwise, format it as a padded integer
                parts.append(f"_s{site:0{site_padding}d}")

        if channel is not None:
            parts.append(f"_w{channel}")

        if z_index is not None:
            if isinstance(z_index, str):
                # If z_index is a string (e.g., '{zzz}'), use it directly
                parts.append(f"_z{z_index}")
            else:
                # Otherwise, format it as a padded integer
                parts.append(f"_z{z_index:0{z_padding}d}")

        base_name = "".join(parts)
        return f"{base_name}{extension}"


class ImageXpressMetadataHandler(MetadataHandler):
    """
    Metadata handler for ImageXpress microscopes.

    Handles finding and parsing HTD files for ImageXpress microscopes.
    """
    def __init__(self, file_manager: Optional[FileManager] = None):
        """
        Initialize the metadata handler.

        Args:
            file_manager: FileManager instance. If None, a disk-based FileManager is created.
        """
        if file_manager is None:
            file_manager = FileManager(backend='disk')
            logger.debug("Created default disk-based FileManager for ImageXpressMetadataHandler")

        self.file_manager = file_manager

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """
        Find the HTD file for an ImageXpress plate.
        Note: This implementation still relies on Path.glob, assuming the
              FileManager's backend provides filesystem-like access.
              A more robust implementation might use file_manager.list_files.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Path to the HTD file, or None if not found
        """
        plate_path = Path(plate_path)

        # Look for ImageXpress HTD file in plate directory
        htd_files = list(plate_path.glob("*.HTD"))
        if htd_files:
            for htd_file in htd_files:
                if 'plate' in htd_file.name.lower():
                    return htd_file
            return htd_files[0]

        return None

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Tuple[int, int]:
        """
        Get grid dimensions for stitching from HTD file.

        Args:
            plate_path: Path to the plate folder

        Returns:
            (grid_size_x, grid_size_y)

        Raises:
            ValueError: If grid dimensions cannot be determined from metadata
        """
        htd_file = self.find_metadata_file(plate_path)
        if not htd_file:
            raise ValueError(f"Cannot find HTD file in {plate_path}. Grid dimensions cannot be determined.")

        # Parse HTD file
        try:
            with open(htd_file, 'r') as f:
                htd_content = f.read()

            # Extract grid dimensions - try multiple formats
            # First try the new format with "XSites" and "YSites"
            cols_match = re.search(r'"XSites", (\d+)', htd_content)
            rows_match = re.search(r'"YSites", (\d+)', htd_content)

            # If not found, try the old format with SiteColumns and SiteRows
            if not (cols_match and rows_match):
                cols_match = re.search(r'SiteColumns=(\d+)', htd_content)
                rows_match = re.search(r'SiteRows=(\d+)', htd_content)

            if cols_match and rows_match:
                grid_size_x = int(cols_match.group(1))
                grid_size_y = int(rows_match.group(1))
                logger.info(f"Using grid dimensions from HTD file: {grid_size_x}x{grid_size_y}")
                return grid_size_x, grid_size_y
            else:
                raise ValueError(f"Could not find grid dimensions in HTD file {htd_file}")
        except Exception as e:
            logger.error(f"Error parsing HTD file {htd_file}: {e}")
            raise ValueError(f"Failed to extract grid dimensions from HTD file: {e}") from e

    def get_pixel_size(self, plate_path: Union[str, Path]) -> float:
        """
        Gets pixel size by reading TIFF tags from an image file via FileManager.

        Args:
            plate_path: Path to the plate folder

        Returns:
            Pixel size in micrometers

        Raises:
            ValueError: If pixel size cannot be determined from metadata
        """
        # TRANSITIONAL: Disk-only logic. This implementation assumes:
        # 1. The backend used by file_manager supports listing image files (like DiskStorageBackend).
        # 2. The backend allows direct reading of TIFF file tags (implicitly assumes local file access
        #    or a backend that can expose file-like objects compatible with tifffile).
        # 3. Images are in TIFF format.
        # Future backends might require different ways to access metadata.
        try:
            # Use file_manager to list potential image files
            image_files = self.file_manager.list_image_files(plate_path, extensions={'.tif', '.tiff'}, recursive=True)
            if not image_files:
                raise ValueError(f"No TIFF images found in {plate_path} to read pixel size.")

            # Attempt to read tags from the first found image
            # Assumes direct path access is possible via the backend (true for Disk)
            first_image_path = image_files[0]
            # For non-disk backends, might need: with self.file_manager.open_file(first_image_path) as f: ...
            with tifffile.TiffFile(first_image_path) as tif:
                 # Try to get ImageDescription tag
                 if tif.pages[0].tags.get('ImageDescription'):
                     desc = tif.pages[0].tags['ImageDescription'].value
                     # Look for spatial calibration using regex
                     match = re.search(r'id="spatial-calibration-x"[^>]*value="([0-9.]+)"', desc)
                     if match:
                         logger.info(f"Found pixel size metadata {str(float(match.group(1)))} in {first_image_path}")
                         return float(match.group(1))

                     # Alternative pattern for some formats
                     match = re.search(r'Spatial Calibration: ([0-9.]+) [uµ]m', desc)
                     if match:
                         logger.info(f"Found pixel size metadata {str(float(match.group(1)))} in {first_image_path}")
                         return float(match.group(1))

            # If we get here, we couldn't find the pixel size


        except Exception as e:
            logger.error(f"Error getting pixel size from {plate_path}: {e}", exc_info=True)
            raise ValueError(f"Failed to determine pixel size from images in {plate_path}: {e}") from e
