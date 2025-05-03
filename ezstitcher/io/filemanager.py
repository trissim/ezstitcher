from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, BinaryIO, TextIO, Callable
import numpy as np
import logging
import tifffile
import os
import shutil

# Import necessary components from the io module
# Assuming StorageBackend, DiskStorageBackend, DEFAULT_IMAGE_EXTENSIONS are available via __init__ or direct import
from ezstitcher.io.storage_backend import BasicStorageBackend, MicroscopyStorageBackend, DiskStorageBackend, FakeStorageBackend # Use specific interfaces/classes
from ezstitcher.io.constants import DEFAULT_IMAGE_EXTENSIONS # Import constant explicitly
from ezstitcher.io.types import ImageArray # Import type alias explicitly
from ezstitcher.io.virtual_path import VirtualPath, open_virtual
from ezstitcher.io.virtual_path_factory import VirtualPathFactory
from ezstitcher.io.storage_adapter import StorageAdapter
from ezstitcher.io.storage_config import StorageConfig # Added import

logger = logging.getLogger(__name__)

def create_backend(backend_type: str = "filesystem") -> MicroscopyStorageBackend:
    """
    Factory function to create a storage backend based on a string identifier.

    Args:
        backend_type: String identifier for the backend type.
            - "filesystem" or "disk": Uses DiskStorageBackend for local file system
            - "memory" or "fake": Uses FakeStorageBackend for in-memory testing
            - Future: "zarr", "s3", etc.

    Returns:
        A storage backend instance implementing MicroscopyStorageBackend

    Raises:
        ValueError: If the backend_type is not recognized
    """
    backend_type = backend_type.lower()

    if backend_type in ("filesystem", "disk"):
        return DiskStorageBackend()
    elif backend_type in ("memory", "fake"):
        return FakeStorageBackend()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}. "
                         f"Supported types: filesystem, memory")

class FileManager:
    """
    Manages file operations using a storage backend.

    This class provides a high-level interface for essential file operations,
    acting purely as an adapter/facade that delegates to a storage backend
    for the actual implementation. It replaces the static FileSystemManager with an instance-based approach that supports dependency injection and multiple backend types (e.g., local disk, cloud storage).

    Scope Clarification:
    - This class is strictly an adapter.
    - It does NOT perform data validation (e.g., image format checks).
    - It does NOT handle logging configuration (uses standard logging).
    - It does NOT implement caching logic.
    These responsibilities belong to upstream callers or the storage backends.

    Interface Segregation:
    - Exposes only methods directly needed for core application workflows
      (image loading/saving, directory management, file listing/finding).
    - Complex, domain-specific logic (like Z-stack handling or advanced renaming)
      is delegated but marked as potentially belonging in separate components/utilities
      in the future to maintain focus.
    """

    # Type hint backend with the most specific interface it needs to support
    # If FileManager only needs basic ops, use BasicStorageBackend.
    # If it needs microscopy ops, use MicroscopyStorageBackend.
    backend: MicroscopyStorageBackend
    root_dir: Path
    storage_adapter: Any  # Type hint for storage adapter

    def __init__(self,
                 root_dir: Optional[Union[str, Path]] = None,
                 backend: Optional[Union[str, MicroscopyStorageBackend]] = None,
                 storage_adapter: Any = None,
                 *, # Make subsequent arguments keyword-only
                 storage_config: StorageConfig):
        """
        Initialize the file manager.

        Args:
            root_dir: Root directory for all file operations. If None, uses current directory.
            backend: Either a string identifier ("filesystem", "memory") or a backend instance.
                     If string, uses create_backend to instantiate the appropriate backend.
                     If None, defaults to "filesystem".
            storage_adapter: Optional storage adapter for logical key resolution.
            storage_config: Configuration object defining storage behavior (keyword-only).
        """
        self.storage_config = storage_config # Added assignment
        # Set root directory
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()

        # Set storage adapter
        self.storage_adapter = storage_adapter

        # Handle backend parameter
        if backend is None:
            # Default to filesystem backend
            self.backend = DiskStorageBackend()
        elif isinstance(backend, str):
            # Use factory function to create backend from string
            self.backend = create_backend(backend)
        elif isinstance(backend, MicroscopyStorageBackend):
            # Use provided backend instance
            self.backend = backend
        else:
            raise TypeError(f"Backend must be a string or MicroscopyStorageBackend instance, "
                            f"got {type(backend)}")

        # Optional runtime check (can be removed if type hints are sufficient)
        if not isinstance(self.backend, MicroscopyStorageBackend):
             raise TypeError(f"Backend must implement MicroscopyStorageBackend, got {type(self.backend)}")

        logger.debug(f"FileManager initialized with backend: {type(self.backend).__name__} "
                     f"and root_dir: {self.root_dir}")


    # --- Logical Key Methods ---

    def resolve_key(self, key: str) -> Optional[Path]:
        """
        Resolve a logical key to a physical path.

        Args:
            key: The logical key to resolve

        Returns:
            A Path object, or None if the key cannot be resolved
        """
        if self.storage_adapter is None:
            logger.debug(f"No storage adapter available to resolve key: {key}")
            return None

        virtual_path = self.storage_adapter.resolve_key(key)
        if virtual_path is None:
            logger.debug(f"Failed to resolve key: {key}")
            return None

        physical_path = virtual_path.to_physical_path()
        logger.debug(f"Resolved key '{key}' to physical path: {physical_path}")
        return physical_path

    def load_from_key(self, key: str) -> Optional[ImageArray]:
        """
        Load an image using a logical key, prioritizing adapter resolution.

        Attempts to resolve the logical key to a physical path using the
        storage adapter. If successful, loads the image from the physical path
        using the backend. If key resolution fails, it falls back to attempting
        a direct read from the storage adapter itself (if available and supports read).

        Args:
            key: The logical key for the image

        Returns:
            Image as numpy array, or None if loading fails
        """
        # Try to resolve the key to a physical path
        path = self.resolve_key(key)
        if path is None:
            # If key resolution fails, try to read directly from storage adapter
            if self.storage_adapter and hasattr(self.storage_adapter, 'read'):
                try:
                    logger.debug(f"Attempting to read key '{key}' directly from storage adapter")
                    return self.storage_adapter.read(key)
                except Exception as e:
                    logger.error(f"Failed to read key '{key}' from storage adapter: {e}")
                    return None
            else:
                logger.error(f"Failed to resolve key '{key}' and no storage adapter available")
                return None

        # Load the image from the resolved path
        return self.load_image(path)

    def write_to_key(self, key: str, image: ImageArray, metadata: Optional[Dict] = None) -> bool:
        """
        Write an image using a logical key.

        Args:
            key: The logical key for the image
            image: The image data to write
            metadata: Optional metadata to include

        Returns:
            True if successful, False otherwise
        """
        # Try to resolve the key to a physical path
        path = self.resolve_key(key)
        if path is None:
            # If key resolution fails, try to write directly to storage adapter
            if self.storage_adapter and hasattr(self.storage_adapter, 'write'):
                try:
                    logger.debug(f"Attempting to write key '{key}' directly to storage adapter")
                    self.storage_adapter.write(key, image)
                    return True
                except Exception as e:
                    logger.error(f"Failed to write key '{key}' to storage adapter: {e}")
                    return False
            else:
                logger.error(f"Failed to resolve key '{key}' and no storage adapter available")
                return False

        # Write the image to the resolved path
        return self.save_image(image, path, metadata)

    # --- Core Image I/O Methods ---

    def load_image(self, file_path: Union[str, Path]) -> Optional[ImageArray]:
        """
        Load an image from a file using the backend.

        This method delegates to the storage backend for loading images from physical paths.
        For virtual paths or storage keys, use load_from_virtual_path() or load_from_key() instead.

        Args:
            file_path: The physical file path

        Returns:
            The image as a NumPy array, or None if loading fails
        """
        logger.debug(f"FileManager delegating load_image for {file_path}")
        return self.backend.load_image(file_path)

    def load_from_virtual_path(self, virtual_path: VirtualPath) -> Optional[ImageArray]:
        """
        Load an image from a virtual path.

        Args:
            virtual_path: The virtual path to the image

        Returns:
            The image as a NumPy array, or None if loading fails
        """
        try:
            # Try to get a physical path first
            physical_path = virtual_path.to_physical_path()
            if physical_path is not None:
                return self.load_image(physical_path)

            # If no physical path, read bytes directly
            bytes_data = virtual_path.read_bytes()
            return tifffile.imread(bytes_data)
        except Exception as e:
            logger.error(f"Failed to read image from VirtualPath '{virtual_path}': {e}")
            return None

    def save_image(self, image: ImageArray, output_path: Union[str, Path], metadata: Optional[Dict] = None) -> bool:
        """
        Save an image to a file using the backend.

        This method delegates to the storage backend for saving images to physical paths.
        For virtual paths or storage keys, use save_to_virtual_path() or save_to_key() instead.

        Args:
            image: The image data
            output_path: The physical file path
            metadata: Optional metadata to include

        Returns:
            True if successful, False otherwise
        """
        # Ensure the parent directory exists
        try:
            parent_dir = Path(output_path).parent
            parent_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create parent directory for '{output_path}': {e}")
            # Continue anyway, the backend might handle this

        # Write the image
        logger.debug(f"FileManager delegating save_image to {output_path}")
        return self.backend.save_image(image, output_path, metadata)

    def save_to_virtual_path(self, image: ImageArray, virtual_path: VirtualPath, metadata: Optional[Dict] = None) -> bool:
        """
        Save an image to a virtual path.

        Args:
            image: The image data
            virtual_path: The virtual path
            metadata: Optional metadata to include

        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to get a physical path first
            physical_path = virtual_path.to_physical_path()
            if physical_path is not None:
                return self.save_image(image, physical_path, metadata)

            # If no physical path, ensure parent directory exists and write bytes directly
            parent = virtual_path.parent()
            parent.mkdir(parents=True, exist_ok=True)

            # Use tifffile to write the image to a bytes object
            bytes_data = tifffile.imwrite(None, image)
            virtual_path.write_bytes(bytes_data)
            return True
        except Exception as e:
            logger.error(f"Failed to write image to VirtualPath '{virtual_path}': {e}")
            return False

    def save_to_key(self, image: ImageArray, key: str, metadata: Optional[Dict] = None) -> bool:
        """
        Save an image using a storage key.

        Args:
            image: The image data
            key: The storage key
            metadata: Optional metadata to include

        Returns:
            True if successful, False otherwise
        """
        if self.storage_adapter is None:
            logger.error("Cannot save to key: no storage adapter available")
            return False

        try:
            self.storage_adapter.write(key, image, metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to write image to storage adapter with key '{key}': {e}")
            return False

    # --- File/Directory Utility Methods ---

    def list_image_files(self, directory: Union[str, Path], extensions: Optional[Set[str]] = DEFAULT_IMAGE_EXTENSIONS, recursive: bool = True) -> List[Path]:
        """
        List image files in a directory using the backend.

        This method delegates to the storage backend for listing files in physical directories.
        For virtual paths, use list_virtual_image_files() instead.
        For storage keys, use list_keys() instead.

        Args:
            directory: Directory to search.
            extensions: Set of file extensions (lowercase, with dot, e.g., {'.tif'}).
                        If None, the backend should use a predefined default set
                        (e.g., DEFAULT_IMAGE_EXTENSIONS from constants).
            recursive: Whether to search recursively.

        Returns:
            List of paths to image files found by the backend.
        """
        logger.debug(f"FileManager delegating list_image_files for {directory} (recursive={recursive}, extensions={extensions})")
        # Pass extensions directly; backend handles None case
        return self.backend.list_files(directory, extensions=extensions, recursive=recursive)

    def list_virtual_image_files(self, virtual_directory: VirtualPath, extensions: Optional[Set[str]] = DEFAULT_IMAGE_EXTENSIONS, recursive: bool = True) -> List[VirtualPath]:
        """
        List image files in a virtual directory.

        Args:
            virtual_directory: Virtual directory to search.
            extensions: Set of file extensions (lowercase, with dot, e.g., {'.tif'}).
                        If None, uses DEFAULT_IMAGE_EXTENSIONS.
            recursive: Whether to search recursively.

        Returns:
            List of virtual paths to image files.
        """
        try:
            # Try to get a physical path first
            physical_path = virtual_directory.to_physical_path()
            if physical_path is not None:
                # Get physical paths from backend
                physical_files = self.list_image_files(physical_path, extensions, recursive)
                # Convert back to virtual paths
                return [virtual_directory.join(p.relative_to(physical_path)) for p in physical_files]

            # If no physical path, use virtual_directory's methods directly
            return virtual_directory.glob("**/*" if recursive else "*", extensions=extensions)
        except Exception as e:
            logger.error(f"Failed to list image files in virtual directory '{virtual_directory}': {e}")
            return []

    def list_keys(self, pattern: str = "*") -> List[str]:
        """
        List keys in the storage adapter matching a pattern.

        Args:
            pattern: Pattern to match keys against (e.g., "well_A01/*")

        Returns:
            List of keys matching the pattern
        """
        if self.storage_adapter is None:
            logger.error("Cannot list keys: no storage adapter available")
            return []

        try:
            return self.storage_adapter.list_keys(pattern)
        except Exception as e:
            logger.error(f"Failed to list keys with pattern '{pattern}': {e}")
            return []

    def list_files(self, directory: Union[str, Path], pattern: Optional[str] = None, extensions: Optional[Set[str]] = None, recursive: bool = False) -> List[Path]:
        """
        List all files in a directory via the backend.

        Args:
            directory: Directory to search.
            pattern: Optional pattern to filter files (e.g., "*.txt").
            recursive: Whether to search recursively.

        Returns:
            List of paths to files found by the backend.
        """
        logger.debug(f"FileManager delegating list_files for {directory} (recursive={recursive}, pattern={pattern})")
        return self.backend.list_files(directory, pattern=pattern, recursive=recursive, extensions=extensions)

    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary, using the backend.

        This method delegates to the storage backend for ensuring physical directories.
        For virtual paths, use ensure_virtual_directory() instead.

        Args:
            directory: The physical directory path to ensure

        Returns:
            The absolute Path object for the directory
        """
        logger.debug(f"FileManager delegating ensure_directory for {directory}")
        return self.backend.ensure_directory(directory)

    def ensure_virtual_directory(self, virtual_directory: VirtualPath) -> VirtualPath:
        """
        Ensure a virtual directory exists, creating it if necessary.

        Args:
            virtual_directory: The virtual directory path to ensure

        Returns:
            The virtual directory path
        """
        try:
            # Try to get a physical path first
            physical_path = virtual_directory.to_physical_path()
            if physical_path is not None:
                self.ensure_directory(physical_path)
                return virtual_directory

            # If no physical path, use mkdir directly
            virtual_directory.mkdir(parents=True, exist_ok=True)
            return virtual_directory
        except Exception as e:
            logger.error(f"Failed to create directory at VirtualPath '{virtual_directory}': {e}")
            raise

    def find_image_directory(self, plate_folder: Union[str, Path], extensions: Optional[Set[str]] = None) -> Path:
        """
        Find the primary directory containing images within a plate folder, via the backend.

        Note: This method involves some domain interpretation ('plate folder').
              While delegated, consider if this logic fits better in a higher-level
              component or utility specific to experiment structure analysis later.

        Args:
            plate_folder: Base directory path to search within.
            extensions: Set of file extensions (lowercase, with dot, e.g., {'.tif'}).
                        If None, the backend should use a predefined default set
                        (e.g., DEFAULT_IMAGE_EXTENSIONS from constants).

        Returns:
            Path to the directory containing images (determined by backend).
            Raises error if not found (handled by backend).
        """
        # COMMENT: TRANSITIONAL - Delegate to backend.
        # Backend implements the search logic, potentially checking common subdirs
        # like 'Images', and applies default extensions logic when `extensions` is None.
        logger.debug(f"FileManager delegating find_image_directory for {plate_folder} (extensions={extensions})")
        # Pass extensions directly; backend handles None case
        return self.backend.find_image_directory(plate_folder, extensions=extensions)

    def find_file_recursive(self, directory: Union[str, Path], filename: str) -> Optional[Path]:
        """
        Find a file by name recursively within a directory, via the backend.

        Args:
            directory: Directory to start the search from.
            filename: Name of the file to find.

        Returns:
            Path to the first matching file found, or None if not found (backend result).
        """
        # COMMENT: TRANSITIONAL - Delegate to backend.
        # Backend handles the recursive search implementation.
        logger.debug(f"FileManager delegating find_file_recursive for '{filename}' in {directory}")
        return self.backend.find_file_recursive(directory, filename)

    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """
        Delete a file via the backend.

        Args:
            file_path: Path to the file to delete.

        Returns:
            True if successful, False otherwise (determined by backend).
            Returns True if the file doesn't exist (idempotent behavior).
        """
        path = Path(file_path)

        # Check if file exists before attempting to delete
        if not self.exists(path):
            logger.debug(f"Tried to delete nonexistent file: {path}")
            return True  # Return success for idempotent behavior

        # COMMENT: TRANSITIONAL - Delegate to backend.
        # Backend handles the actual deletion and error handling.
        logger.debug(f"FileManager delegating delete_file for {path}")
        return self.backend.delete_file(path)

    def copy_file(self, source_path: Union[str, Path], dest_path: Union[str, Path]) -> bool:
        """
        Copy a file from source to destination via the backend.

        Args:
            source_path: Path to the source file.
            dest_path: Path to the destination file.

        Returns:
            True if successful, False otherwise (determined by backend).
        """
        # COMMENT: TRANSITIONAL - Delegate to backend.
        # Backend handles the copy operation, including overwriting logic if applicable.
        logger.debug(f"FileManager delegating copy_file from {source_path} to {dest_path}")
        return self.backend.copy_file(source_path, dest_path)

    def exists(self, path: Union[str, Path]) -> bool:
        """
        Check if a file or directory exists using the backend.

        This method delegates to the storage backend for checking physical paths.
        For virtual paths or storage keys, use virtual_path_exists() or key_exists() instead.

        Args:
            path: The physical path to check

        Returns:
            True if the file or directory exists, False otherwise
        """
        return self.backend.exists(path)

    def virtual_path_exists(self, virtual_path: VirtualPath) -> bool:
        """
        Check if a virtual path exists.

        Args:
            virtual_path: The virtual path to check

        Returns:
            True if the virtual path exists, False otherwise
        """
        return virtual_path.exists()

    def key_exists(self, key: str) -> bool:
        """
        Check if a storage key exists.

        Args:
            key: The storage key to check

        Returns:
            True if the key exists, False otherwise
        """
        if self.storage_adapter is None:
            logger.error("Cannot check key existence: no storage adapter available")
            return False

        return self.storage_adapter.exists(key)

    def remove(self, file_path: Union[str, Path]) -> bool:
        """
        Remove a file using the backend.

        This method delegates to the storage backend for removing physical files.
        For virtual paths or storage keys, use remove_virtual_path() or remove_key() instead.

        This is an alias for delete_file to maintain compatibility with code that uses remove.

        Args:
            file_path: The physical file path

        Returns:
            True if successful, False otherwise
        """
        return self.delete_file(file_path)

    def remove_virtual_path(self, virtual_path: VirtualPath) -> bool:
        """
        Remove a file at a virtual path.

        Args:
            virtual_path: The virtual path to remove

        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to get a physical path first
            physical_path = virtual_path.to_physical_path()
            if physical_path is not None:
                return self.delete_file(physical_path)

            # If no physical path, use unlink directly
            virtual_path.unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to remove file at VirtualPath '{virtual_path}': {e}")
            return False

    def remove_key(self, key: str) -> bool:
        """
        Remove a file using a storage key.

        Args:
            key: The storage key to remove

        Returns:
            True if successful, False otherwise
        """
        if self.storage_adapter is None:
            logger.error("Cannot remove key: no storage adapter available")
            return False

        try:
            if not self.storage_adapter.exists(key):
                # File doesn't exist, consider it a success (idempotent behavior)
                return True

            self.storage_adapter.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to remove file from storage adapter with key '{key}': {e}")
            return False

    def remove_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> bool:
        """
        Remove a directory and optionally all its contents via the backend.

        Args:
            directory_path: Path to the directory to remove
            recursive: Whether to remove the directory recursively (including all contents)

        Returns:
            True if successful, False otherwise (determined by backend)
        """
        logger.debug(f"FileManager delegating remove_directory for {directory_path} (recursive={recursive})")
        return self.backend.remove_directory(directory_path, recursive=recursive)

    # --- Methods with Complex/Domain-Specific Logic (Candidates for Future Refactoring) ---
    # These methods are currently delegated but contain logic that might be better suited
    # for dedicated utility classes or components outside the core FileManager adapter role.

    def rename_files_with_consistent_padding(self, directory: Union[str, Path], parser: Any, width: int = 3, force_suffixes: bool = False) -> Dict[Path, Path]:
        """
        Rename files for consistent numeric padding, via the backend.

        Note: This involves complex parsing and renaming rules. Consider moving
              to a dedicated 'FileNameFormatter' or similar utility later.

        Args:
            directory: Directory containing files to rename.
            parser: Parser object capable of extracting components for padding.
            width: Target width for numeric padding (e.g., site number).
            force_suffixes: Whether to enforce specific suffix presence.

        Returns:
            Dictionary mapping original filenames to new filenames (from backend).
        """
        # COMMENT: TRANSITIONAL - Delegate to backend.
        # Backend currently houses this complex renaming logic.
        logger.debug(f"FileManager delegating rename_files_with_consistent_padding for {directory}")
        return self.backend.rename_files_with_consistent_padding(directory, parser, width, force_suffixes)

    def detect_zstack_folders(self, plate_folder: Union[str, Path], pattern: Optional[str] = None) -> Tuple[bool, List[Path]]:
        """
        Detect Z-stack folders based on naming conventions, via the backend.

        Note: This is highly domain-specific (microscopy Z-stacks). Consider moving
              to a 'MicroscopyUtils' or 'ExperimentStructureAnalyzer' later.

        Args:
            plate_folder: Path to the plate folder.
            pattern: Regex pattern to identify Z-stack folders (backend-specific default?).

        Returns:
            Tuple (has_zstack: bool, z_folders: List[Path]) provided by the backend.
        """
        # COMMENT: TRANSITIONAL - Delegate to backend.
        # Backend currently implements Z-stack detection logic.
        logger.debug(f"FileManager delegating detect_zstack_folders for {plate_folder}")
        return self.backend.detect_zstack_folders(plate_folder, pattern)

    def organize_zstack_folders(self, plate_folder: Union[str, Path], filename_parser: Any) -> bool:
        """
        Organize files into Z-stack subfolders, via the backend.

        Note: Complex, domain-specific file organization logic. Consider moving
              to a dedicated 'ZStackOrganizer' or similar utility later.

        Args:
            plate_folder: Path to the plate folder containing images to organize.
            filename_parser: Parser needed to identify Z-plane from filenames.

        Returns:
            True if successful, False otherwise (determined by backend).
        """
        # COMMENT: TRANSITIONAL - Delegate to backend.
        # Backend currently implements the logic to move files into Z-stack folders.
        logger.debug(f"FileManager delegating organize_zstack_folders for {plate_folder}")
        return self.backend.organize_zstack_folders(plate_folder, filename_parser)

    def create_symlink(self, source_path: Union[str, Path], symlink_path: Union[str, Path]) -> bool:
        """
        Create a symbolic link from source_path to symlink_path.

        Args:
            source_path: Path to the source file or directory
            symlink_path: Path where the symlink should be created

        Returns:
            bool: True if successful, False otherwise
        """
        logger.debug("FileManager delegating create_symlink from %s to %s", source_path, symlink_path)
        return self.backend.create_symlink(source_path, symlink_path)

    def rename(self, old_path: Union[str, Path], new_path: Union[str, Path]) -> bool:
        """
        Rename a file or directory.

        Args:
            old_path: Path to the file or directory to rename
            new_path: New path for the file or directory

        Returns:
            bool: True if successful, False otherwise
        """
        logger.debug("FileManager delegating rename from %s to %s", old_path, new_path)
        return self.backend.rename(old_path, new_path)

    def mirror_directory_with_symlinks(self, source_dir: Union[str, Path], target_dir: Union[str, Path],
                                      recursive: bool = True, overwrite: bool = True) -> int:
        """
        Mirror a directory structure from source to target and create symlinks to all files.

        This method delegates to the backend's implementation, which typically uses
        the dedicated directory_mirror utility to implement the functionality.

        Args:
            source_dir: Path to the source directory to mirror
            target_dir: Path to the target directory where the mirrored structure will be created
            recursive: Whether to recursively mirror subdirectories. Defaults to True.
            overwrite: Whether to overwrite the target directory if it exists. Defaults to True.

        Returns:
            int: Number of symlinks created
        """
        logger.debug("FileManager delegating mirror_directory_with_symlinks from %s to %s", source_dir, target_dir)
        return self.backend.mirror_directory_with_symlinks(source_dir, target_dir, recursive=recursive, overwrite=overwrite)