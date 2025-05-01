# ezstitcher/io/storage_backend.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set

# Import type aliases and constants
from .types import ImageArray
from .constants import DEFAULT_IMAGE_EXTENSIONS

import logging
import numpy as np # Needed for type hints here
import os
import shutil
from glob import glob
from ezstitcher.core.file_system_manager import FileSystemManager # Existing static manager
import fnmatch # For pattern matching in list_files
import copy # For simulating read/write isolation


logger = logging.getLogger(__name__)

class BasicStorageBackend(ABC):
    """
    Abstract base class for basic storage operations.

    Defines the fundamental operations required for interacting with a storage system,
    independent of specific data types like microscopy images.
    """

    @abstractmethod
    def load(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load data from a file."""
        pass

    @abstractmethod
    def save(self, data: Any, output_path: Union[str, Path], **kwargs) -> bool:
        """Save data to a file."""
        pass

    @abstractmethod
    def list_files(self, directory: Union[str, Path], pattern: Optional[str] = None, extensions: Optional[Set[str]] = None, recursive: bool = False) -> List[Path]:
        """
        List files in a directory, optionally filtering by pattern and extensions.

        Args:
            directory: Directory to search.
            pattern: Optional glob pattern to match filenames.
            extensions: Optional set of file extensions to filter by (e.g., {'.tif', '.png'}).
                        Extensions should include the dot and are case-insensitive.
            recursive: Whether to search recursively.

        Returns:
            List of paths to matching files.
        """
        pass

    @abstractmethod
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """Delete a file."""
        pass

    @abstractmethod
    def copy_file(self, source_path: Union[str, Path], dest_path: Union[str, Path]) -> bool:
        """Copy a file."""
        pass

    @abstractmethod
    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """Ensure a directory exists, creating it if necessary."""
        pass

    @abstractmethod
    def find_file_recursive(self, directory: Union[str, Path], filename: str) -> Optional[Path]:
        """Find a file by name recursively."""
        pass

    @abstractmethod
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a file or directory exists."""
        pass


class MicroscopyStorageBackend(BasicStorageBackend):
    """
    Abstract base class extending basic storage with microscopy-specific operations.

    Builds upon BasicStorageBackend to add methods relevant to handling
    microscopy images, z-stacks, and specific directory structures.
    """

    @abstractmethod
    def load_image(self, file_path: Union[str, Path]) -> Optional[ImageArray]:
        """Load a microscopy image."""
        # This could potentially delegate to self.load() with specific kwargs
        pass

    @abstractmethod
    def save_image(self, image: ImageArray, output_path: Union[str, Path], metadata: Optional[Dict] = None) -> bool:
        """Save a microscopy image, potentially with metadata."""
        # This could potentially delegate to self.save() with specific kwargs
        pass

    @abstractmethod
    def list_image_files(self, directory: Union[str, Path], extensions: Optional[Set[str]] = None, recursive: bool = True) -> List[Path]:
        """
        List image files in a directory, filtering by specific extensions.

        Args:
            directory: Directory to search.
            extensions: Set of file extensions (e.g., {'.tif', '.png'}).
                        If None, uses DEFAULT_IMAGE_EXTENSIONS from ezstitcher.io.constants.
            recursive: Whether to search recursively.

        Returns:
            List of paths to image files.
        """
        # Implementation should use the 'extensions' argument, defaulting to
        # DEFAULT_IMAGE_EXTENSIONS if 'extensions' is None.
        pass

    @abstractmethod
    def find_image_directory(self, plate_folder: Union[str, Path], extensions: Optional[Set[str]] = None) -> Path:
        """
        Find the primary directory containing image files within a plate folder.

        Args:
            plate_folder: Base directory to search (e.g., the plate folder).
            extensions: Set of file extensions to look for (e.g., {'.tif', '.png'}).
                        If None, uses DEFAULT_IMAGE_EXTENSIONS from ezstitcher.io.constants.

        Returns:
            Path to the directory containing the images.
        """
        # Implementation should use the 'extensions' argument, defaulting to
        # DEFAULT_IMAGE_EXTENSIONS if 'extensions' is None.
        pass

    @abstractmethod
    def rename_files_with_consistent_padding(self, directory: Union[str, Path], parser: Any, width: int = 3, force_suffixes: bool = False) -> Dict[Path, Path]:
        """Rename microscopy image files for consistent numerical padding."""
        # Note: The 'parser' type hint should be refined.
        pass

    @abstractmethod
    def detect_zstack_folders(self, plate_folder: Union[str, Path], pattern: Optional[str] = None) -> Tuple[bool, List[Path]]:
        """Detect folders likely containing Z-stack data."""
        pass

    @abstractmethod
    def organize_zstack_folders(self, plate_folder: Union[str, Path], filename_parser: Any) -> bool:
        """Organize files within Z-stack folders based on a parser."""
        # Note: The 'filename_parser' type hint should be refined.
        pass

    @abstractmethod
    def create_symlink(self, source_path: Union[str, Path], symlink_path: Union[str, Path]) -> bool:
        """Create a symbolic link from source_path to symlink_path."""
        pass

    @abstractmethod
    def rename(self, old_path: Union[str, Path], new_path: Union[str, Path]) -> bool:
        """Rename a file or directory."""
        pass

    @abstractmethod
    def remove_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> bool:
        """
        Remove a directory and optionally all its contents.

        Args:
            directory_path: Path to the directory to remove
            recursive: Whether to remove the directory recursively (including all contents)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def mirror_directory_with_symlinks(self, source_dir: Union[str, Path], target_dir: Union[str, Path],
                                      recursive: bool = True, overwrite: bool = True) -> int:
        """
        Mirror a directory structure from source to target and create symlinks to all files.

        Args:
            source_dir: Path to the source directory to mirror
            target_dir: Path to the target directory where the mirrored structure will be created
            recursive: Whether to recursively mirror subdirectories
            overwrite: Whether to overwrite the target directory if it exists

        Returns:
            Number of symlinks created
        """
        pass

    # Consider adding load_zstack / save_zstack if direct Z-stack operations are needed
    # @abstractmethod
    # def load_zstack(self, zstack_folder: Union[str, Path]) -> Optional[ImageArray]: ...
    #
    # @abstractmethod
    # def save_zstack(self, zstack: ImageArray, output_folder: Union[str, Path], ...) -> bool: ...


class DiskStorageBackend(MicroscopyStorageBackend): # Implement the most specific interface
    """
    Storage backend that uses the local disk.

    **Transitional Implementation:** This backend currently delegates heavily
    to the static `ezstitcher.core.file_system_manager.FileSystemManager`.
    Future work involves replacing these delegations with direct Python
    file system operations (os, pathlib, shutil) to fully decouple from
    FileSystemManager.
    """

    # --- BasicStorageBackend Methods ---

    def load(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load data from disk. Currently specialized for images via FileSystemManager."""
        # TODO: Replace FileSystemManager.load_image with generic load logic
        #       based on file type or kwargs. For now, assumes image loading.
        #       Requires understanding how non-image data might be loaded.
        if 'image' in kwargs or str(file_path).endswith(tuple(DEFAULT_IMAGE_EXTENSIONS)):
             # Delegate to the specific image loader for now
             return self.load_image(file_path)
        logger.warning(f"DiskStorageBackend.load called for non-image path {file_path}, returning None (implement generic load)")
        # Placeholder for generic load - perhaps raise NotImplementedError?
        raise NotImplementedError(f"Generic load not implemented for {file_path}")


    def save(self, data: Any, output_path: Union[str, Path], **kwargs) -> bool:
        """Save data to disk. Currently specialized for images via FileSystemManager."""
        # TODO: Replace FileSystemManager.save_image with generic save logic
        #       based on data type or kwargs. For now, assumes image saving.
        #       Requires understanding how non-image data might be saved.
        if isinstance(data, np.ndarray): # Assuming ImageArray is np.ndarray
            metadata = kwargs.get('metadata') # Check if metadata kwarg exists
            # Delegate to the specific image saver for now
            return self.save_image(data, output_path, metadata)
        logger.warning(f"DiskStorageBackend.save called for non-image data type {type(data)}, returning False (implement generic save)")
        # Placeholder for generic save - perhaps raise NotImplementedError?
        raise NotImplementedError(f"Generic save not implemented for type {type(data)}")


    def list_files(self, directory: Union[str, Path], pattern: Optional[str] = None, extensions: Optional[Set[str]] = None, recursive: bool = False) -> List[Path]:
        """
        List files on disk, optionally filtering by pattern and extensions.

        Args:
            directory: Directory to search.
            pattern: Optional glob pattern to match filenames.
            extensions: Optional set of file extensions to filter by (e.g., {'.tif', '.png'}).
                        Extensions should include the dot and are case-insensitive.
            recursive: Whether to search recursively.

        Returns:
            List of paths to matching files.
        """
        directory = Path(directory)
        if not directory.is_dir():
            return []
        try:
            if recursive:
                glob_pattern = f"**/{pattern}" if pattern else "**/*"
                files = [p for p in directory.glob(glob_pattern) if p.is_file()]
            else:
                glob_pattern = pattern if pattern else "*"
                files = [p for p in directory.glob(glob_pattern) if p.is_file()]

            # Filter by extensions if provided
            if extensions:
                # Convert extensions to lowercase for case-insensitive comparison
                lowercase_extensions = {ext.lower() for ext in extensions}
                files = [f for f in files if f.suffix.lower() in lowercase_extensions]

            return files
        except Exception as e:
            logger.error(f"Error listing files in {directory} with pattern '{pattern}': {e}")
            return []


    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """Delete a file from disk."""
        # NATIVE IMPLEMENTATION (No TODO needed)
        try:
            os.remove(file_path)
            logger.debug(f"Deleted file: {file_path}")
            return True
        except FileNotFoundError:
            logger.warning(f"Attempted to delete non-existent file: {file_path}")
            return False # Or True depending on desired idempotency
        except OSError as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False

    def copy_file(self, source_path: Union[str, Path], dest_path: Union[str, Path]) -> bool:
        """Copy a file on disk."""
        # NATIVE IMPLEMENTATION (No TODO needed)
        try:
            dest_path_obj = Path(dest_path)
            self.ensure_directory(dest_path_obj.parent) # Ensure destination dir exists
            shutil.copy2(source_path, dest_path_obj) # copy2 preserves metadata
            logger.debug(f"Copied file {source_path} to {dest_path_obj}")
            return True
        except FileNotFoundError:
             logger.error(f"Source file not found for copy: {source_path}")
             return False
        except OSError as e:
            logger.error(f"OS error copying file {source_path} to {dest_path}: {e}")
            return False
        except Exception as e: # Catch other potential errors like permissions
             logger.error(f"Unexpected error copying file {source_path} to {dest_path}: {e}")
             return False


    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """Ensure a directory exists on disk."""
        # NATIVE IMPLEMENTATION (No TODO needed)
        path = Path(directory)
        try:
            path.mkdir(parents=True, exist_ok=True)
            # logger.debug(f"Ensured directory exists: {path}") # Can be noisy
            return path
        except OSError as e:
            logger.error(f"Error creating directory {path}: {e}")
            raise # Re-raise after logging, as this is often critical

    def find_file_recursive(self, directory: Union[str, Path], filename: str) -> Optional[Path]:
        """Find a file recursively on disk."""
        # NATIVE IMPLEMENTATION (No TODO needed)
        directory = Path(directory)
        if not directory.is_dir():
            return None
        try:
            # Use rglob for recursive search
            found = list(directory.rglob(filename))
            if found:
                # Prioritize exact matches if multiple files match a pattern
                exact_matches = [p for p in found if p.name == filename]
                if exact_matches:
                    logger.debug(f"Found exact match for '{filename}' at: {exact_matches[0]}")
                    return exact_matches[0]
                else:
                    logger.debug(f"Found pattern match for '{filename}' at: {found[0]}")
                    return found[0] # Return the first pattern match if no exact match
            else:
                logger.debug(f"File '{filename}' not found in {directory}")
                return None
        except Exception as e:
            logger.error(f"Error finding file '{filename}' in {directory}: {e}")
            return None

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a file or directory exists on disk."""
        # NATIVE IMPLEMENTATION (No TODO needed)
        return Path(path).exists()

    def create_symlink(self, source_path: Union[str, Path], symlink_path: Union[str, Path]) -> bool:
        """Create a symbolic link from source_path to symlink_path."""
        # NATIVE IMPLEMENTATION
        try:
            source_path_obj = Path(source_path)
            symlink_path_obj = Path(symlink_path)

            # Ensure the parent directory of the symlink exists
            self.ensure_directory(symlink_path_obj.parent)

            # Create the symlink
            if symlink_path_obj.exists():
                # Remove existing symlink if it exists
                symlink_path_obj.unlink()

            symlink_path_obj.symlink_to(source_path_obj)
            logger.debug("Created symlink from %s to %s", source_path_obj, symlink_path_obj)
            return True
        except FileNotFoundError:
            logger.error("Source file not found for symlink: %s", source_path)
            return False
        except OSError as e:
            logger.error("OS error creating symlink from %s to %s: %s", source_path, symlink_path, e)
            return False
        except Exception as e:
            logger.error("Unexpected error creating symlink from %s to %s: %s", source_path, symlink_path, e)
            return False

    def rename(self, old_path: Union[str, Path], new_path: Union[str, Path]) -> bool:
        """Rename a file or directory."""
        # NATIVE IMPLEMENTATION
        try:
            old_path_obj = Path(old_path)
            new_path_obj = Path(new_path)

            # Ensure the parent directory of the new path exists
            self.ensure_directory(new_path_obj.parent)

            # Rename the file or directory
            old_path_obj.rename(new_path_obj)
            logger.debug("Renamed %s to %s", old_path_obj, new_path_obj)
            return True
        except FileNotFoundError:
            logger.error("Source file not found for rename: %s", old_path)
            return False
        except OSError as e:
            logger.error("OS error renaming %s to %s: %s", old_path, new_path, e)
            return False
        except Exception as e:
            logger.error("Unexpected error renaming %s to %s: %s", old_path, new_path, e)
            return False

    def remove_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> bool:
        """
        Remove a directory and optionally all its contents.

        Args:
            directory_path: Path to the directory to remove
            recursive: Whether to remove the directory recursively (including all contents)

        Returns:
            True if successful, False otherwise
        """
        try:
            directory_path_obj = Path(directory_path)

            if not directory_path_obj.exists():
                logger.warning(f"Directory does not exist: {directory_path_obj}")
                return True  # Consider non-existence as success (idempotent)

            if not directory_path_obj.is_dir():
                logger.error(f"Path is not a directory: {directory_path_obj}")
                return False

            if recursive:
                shutil.rmtree(directory_path_obj)
                logger.debug(f"Recursively removed directory: {directory_path_obj}")
            else:
                directory_path_obj.rmdir()
                logger.debug(f"Removed empty directory: {directory_path_obj}")

            return True
        except OSError as e:
            logger.error(f"OS error removing directory {directory_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error removing directory {directory_path}: {e}")
            return False

    # --- MicroscopyStorageBackend Methods (Delegated) ---

    def load_image(self, file_path: Union[str, Path]) -> Optional[ImageArray]:
        """Load an image from disk using FileSystemManager."""
        # TODO: Replace FileSystemManager.load_image with native image loading (e.g., tifffile, imageio)
        logger.debug(f"Delegating load_image for {file_path} to FileSystemManager")
        try:
            # Ensure FileSystemManager is available and has the method
            if hasattr(FileSystemManager, 'load_image'):
                 return FileSystemManager.load_image(file_path)
            else:
                 logger.error("FileSystemManager.load_image not found!")
                 return None
        except Exception as e:
            logger.error(f"Error during FileSystemManager.load_image for {file_path}: {e}")
            return None


    def save_image(self, image: ImageArray, output_path: Union[str, Path], metadata: Optional[Dict] = None) -> bool:
        """Save an image to disk using FileSystemManager."""
        # TODO: Replace FileSystemManager.save_image with native image saving (e.g., tifffile, imageio)
        #       Handle metadata if the chosen library supports it.
        logger.debug(f"Delegating save_image for {output_path} to FileSystemManager")
        try:
             # Note: FileSystemManager.save_image signature might be (path, image)
             if hasattr(FileSystemManager, 'save_image'):
                 # FileSystemManager might not handle metadata, pass only required args
                 return FileSystemManager.save_image(image, output_path)
             else:
                 logger.error("FileSystemManager.save_image not found!")
                 return False
        except Exception as e:
            logger.error(f"Error during FileSystemManager.save_image for {output_path}: {e}")
            return False


    def list_image_files(self, directory: Union[str, Path], extensions: Optional[Set[str]] = None, recursive: bool = True) -> List[Path]:
        """
        List image files in a directory, filtering by specific extensions.

        Args:
            directory: Directory to search.
            extensions: Set of file extensions (e.g., {'.tif', '.png'}).
                        If None, uses DEFAULT_IMAGE_EXTENSIONS from ezstitcher.io.constants.
            recursive: Whether to search recursively.

        Returns:
            List of paths to image files.
        """
        logger.debug(f"Listing image files in {directory}")
        effective_extensions = extensions if extensions is not None else DEFAULT_IMAGE_EXTENSIONS
        # Use our native list_files implementation with extension filtering
        return self.list_files(directory, pattern=None, extensions=effective_extensions, recursive=recursive)


    def find_image_directory(self, plate_folder: Union[str, Path], extensions: Optional[Set[str]] = None) -> Path:
        """Find the image directory on disk using FileSystemManager."""
        # TODO: Replace FileSystemManager.find_image_directory with native logic
        #       (e.g., search common subdirs like 'Images', 'Data', or check root)
        #       using self.list_files and provided/default extensions.
        logger.debug(f"Delegating find_image_directory for {plate_folder} to FileSystemManager")
        effective_extensions = extensions if extensions is not None else DEFAULT_IMAGE_EXTENSIONS
        try:
            if hasattr(FileSystemManager, 'find_image_directory'):
                # FileSystemManager might expect a List, not Set
                return FileSystemManager.find_image_directory(plate_folder, list(effective_extensions))
            else:
                 logger.error("FileSystemManager.find_image_directory not found!")
                 # Need to return a Path or raise error. Raising is safer.
                 raise FileNotFoundError("FileSystemManager.find_image_directory not available")
        except Exception as e:
            logger.error(f"Error during FileSystemManager.find_image_directory for {plate_folder}: {e}")
            raise FileNotFoundError(f"Could not find image directory in {plate_folder} via FileSystemManager") from e


    def rename_files_with_consistent_padding(self, directory: Union[str, Path], parser: Any, width: int = 3, force_suffixes: bool = False) -> Dict[Path, Path]:
        """Rename files on disk using FileSystemManager."""
        # TODO: Replace FileSystemManager.rename_files_with_consistent_padding with native logic
        #       involving listing files, parsing names, generating new names, and renaming.
        logger.debug(f"Delegating rename_files_with_consistent_padding for {directory} to FileSystemManager")
        try:
            if hasattr(FileSystemManager, 'rename_files_with_consistent_padding'):
                # Assuming the return type of FSM is Dict[str, str], convert to Path
                result_str_dict = FileSystemManager.rename_files_with_consistent_padding(directory, parser, width, force_suffixes)
                return {Path(k): Path(v) for k, v in result_str_dict.items()}
            else:
                logger.error("FileSystemManager.rename_files_with_consistent_padding not found!")
                return {}
        except Exception as e:
            logger.error(f"Error during FileSystemManager.rename_files_with_consistent_padding for {directory}: {e}")
            return {}


    def detect_zstack_folders(self, plate_folder: Union[str, Path], pattern: Optional[str] = None) -> Tuple[bool, List[Path]]:
        """Detect Z-stack folders on disk using FileSystemManager."""
        # TODO: Replace FileSystemManager.detect_zstack_folders with native logic
        #       (e.g., list directories matching a pattern like 'Z[0-9]+').
        logger.debug(f"Delegating detect_zstack_folders for {plate_folder} to FileSystemManager")
        try:
            if hasattr(FileSystemManager, 'detect_zstack_folders'):
                return FileSystemManager.detect_zstack_folders(plate_folder, pattern)
            else:
                logger.error("FileSystemManager.detect_zstack_folders not found!")
                return False, []
        except Exception as e:
            logger.error(f"Error during FileSystemManager.detect_zstack_folders for {plate_folder}: {e}")
            return False, []


    def organize_zstack_folders(self, plate_folder: Union[str, Path], filename_parser: Any) -> bool:
        """Organize Z-stack folders on disk using FileSystemManager."""
        # TODO: Replace FileSystemManager.organize_zstack_folders with native logic
        #       involving listing files, parsing, creating directories, and moving files.
        logger.debug(f"Delegating organize_zstack_folders for {plate_folder} to FileSystemManager")
        try:
            if hasattr(FileSystemManager, 'organize_zstack_folders'):
                return FileSystemManager.organize_zstack_folders(plate_folder, filename_parser)
            else:
                logger.error("FileSystemManager.organize_zstack_folders not found!")
                return False
        except Exception as e:
            logger.error(f"Error during FileSystemManager.organize_zstack_folders for {plate_folder}: {e}")
            return False

    def mirror_directory_with_symlinks(self, source_dir: Union[str, Path], target_dir: Union[str, Path],
                                      recursive: bool = True, overwrite: bool = True) -> int:
        """
        Mirror a directory structure from source to target and create symlinks to all files.

        This method uses the dedicated directory_mirror utility to implement the functionality,
        converting the boolean overwrite parameter to the appropriate OverwriteStrategy.

        Args:
            source_dir: Path to the source directory to mirror
            target_dir: Path to the target directory where the mirrored structure will be created
            recursive: Whether to recursively mirror subdirectories. Defaults to True.
            overwrite: Whether to overwrite the target directory if it exists. Defaults to True.

        Returns:
            int: Number of symlinks created
        """
        from ezstitcher.io.directory_mirror import mirror_directory_with_symlinks as mirror_func
        from ezstitcher.io.directory_mirror import OverwriteStrategy

        # Convert boolean overwrite to strategy enum
        strategy = OverwriteStrategy.REPLACE if overwrite else OverwriteStrategy.MERGE

        logger.debug(f"Mirroring directory {source_dir} to {target_dir} with strategy={strategy.value}")
        return mirror_func(source_dir, target_dir, recursive=recursive, strategy=strategy, verbose=True)


class FakeStorageBackend(MicroscopyStorageBackend): # Implement the most specific interface
    """
    In-memory fake storage backend for testing purposes.

    Simulates file system operations using Python dictionaries and basic data structures.
    Tries to mimic file system behavior regarding paths and existence checks.
    """
    def __init__(self):
        # Stores file content (e.g., bytes, str, ImageArray). Use deepcopy on access/mutation.
        self._files: Dict[Path, Any] = {}
        # Stores existing directory paths. Start with root.
        self._directories: Set[Path] = {Path('.')}
        logger.debug("Initialized FakeStorageBackend")

    def _add_parents(self, path: Path):
        """Helper to ensure parent directories exist in the fake store."""
        parent = path.parent
        while parent != Path('.') and parent not in self._directories:
            self._directories.add(parent)
            logger.debug(f"FakeStorageBackend: Added parent directory {parent}")
            parent = parent.parent

    def _normalize_path(self, path: Union[str, Path]) -> Path:
        """Ensure path is normalized but maintain relative paths.

        This method normalizes paths (resolving . and ..) but maintains
        the relative/absolute nature of the original path.
        """
        # Convert to Path object if it's a string
        p = Path(path)

        # Normalize the path (resolve . and ..) but don't make it absolute
        # Use parts to rebuild the path without making it absolute
        if p.is_absolute():
            # If it's already absolute, keep it that way
            return p
        else:
            # If it's relative, keep it relative but normalize it
            return Path(*p.parts)

    # --- BasicStorageBackend Methods ---

    def load(self, file_path: Union[str, Path], **kwargs) -> Any:
        path = self._normalize_path(file_path)
        logger.debug(f"FakeStorageBackend: Attempting to load {path}")
        if path in self._files:
            # Return a deep copy to prevent modifying the stored fake data externally
            content = copy.deepcopy(self._files[path])
            logger.debug(f"FakeStorageBackend: Loaded {path}")
            return content
        logger.error(f"FakeStorageBackend: File not found for load: {path}")
        raise FileNotFoundError(f"Fake file not found: {path}")

    def save(self, data: Any, output_path: Union[str, Path], **kwargs) -> bool:
        path = self._normalize_path(output_path)
        logger.debug(f"FakeStorageBackend: Attempting to save to {path}")
        self._add_parents(path)
        # Store a deep copy to prevent external modifications affecting the fake store
        self._files[path] = copy.deepcopy(data)
        # If saving a file implicitly creates its directory entry if needed
        if path.parent not in self._directories:
             self._directories.add(path.parent)
        logger.debug(f"FakeStorageBackend: Saved data to {path}")
        return True

    def list_files(self, directory: Union[str, Path], pattern: Optional[str] = None, extensions: Optional[Set[str]] = None, recursive: bool = False ) -> List[Path]:
        """
        List files in the fake storage system, optionally filtering by pattern and extensions.

        Args:
            directory: Directory to search.
            pattern: Optional glob pattern to match filenames.
            extensions: Optional set of file extensions to filter by (e.g., {'.tif', '.png'}).
                        Extensions should include the dot and are case-insensitive.
            recursive: Whether to search recursively.

        Returns:
            List of paths to matching files.
        """
        dir_path = self._normalize_path(directory)
        logger.debug(f"FakeStorageBackend: Listing files in {dir_path} (pattern='{pattern}', extensions={extensions}, recursive={recursive})")

        # Check if directory exists explicitly or implicitly via a file within it
        dir_exists = dir_path in self._directories or any(f.parent == dir_path for f in self._files)
        if not dir_exists:
             logger.warning(f"FakeStorageBackend: Directory not found for listing: {dir_path}")
             return []

        results = []
        for file_path in self._files.keys():
            # Check if file is directly in the directory
            is_in_dir = file_path.parent == dir_path
            # Check if file is in a subdirectory (for recursive search)
            # Use path.parents which includes all ancestors
            is_in_subdir = dir_path in file_path.parents

            if (recursive and (is_in_dir or is_in_subdir)) or (not recursive and is_in_dir):
                # Apply pattern matching using fnmatch (Unix shell-style)
                if pattern is None or fnmatch.fnmatch(file_path.name, pattern):
                    # Apply extension filtering if provided
                    if extensions is None or file_path.suffix.lower() in {ext.lower() for ext in extensions}:
                        results.append(file_path)

        logger.debug(f"FakeStorageBackend: Found {len(results)} files matching criteria in {dir_path}")
        return results


    def delete_file(self, file_path: Union[str, Path]) -> bool:
        path = self._normalize_path(file_path)
        logger.debug(f"FakeStorageBackend: Attempting to delete {path}")
        if path in self._files:
            del self._files[path]
            logger.debug(f"FakeStorageBackend: Deleted file {path}")
            return True
        logger.warning(f"FakeStorageBackend: File not found for deletion: {path}")
        return False # File didn't exist

    def copy_file(self, source_path: Union[str, Path], dest_path: Union[str, Path]) -> bool:
        src = self._normalize_path(source_path)
        dest = self._normalize_path(dest_path)
        logger.debug(f"FakeStorageBackend: Attempting to copy {src} to {dest}")
        if src in self._files:
            self._add_parents(dest)
            # Simulate copy - use deepcopy for mutable data
            self._files[dest] = copy.deepcopy(self._files[src])
            logger.debug(f"FakeStorageBackend: Copied {src} to {dest}")
            return True
        logger.error(f"FakeStorageBackend: Source file not found for copy: {src}")
        return False

    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        path = self._normalize_path(directory)
        logger.debug(f"FakeStorageBackend: Ensuring directory {path}")
        self._add_parents(path)
        if path not in self._directories:
            self._directories.add(path)
            logger.debug(f"FakeStorageBackend: Created directory {path}")
        return path

    def find_file_recursive(self, directory: Union[str, Path], filename: str) -> Optional[Path]:
        dir_path = self._normalize_path(directory)
        logger.debug(f"FakeStorageBackend: Finding '{filename}' recursively in {dir_path}")
        # This is simplified; a real find might need pattern matching if filename isn't exact
        possible_matches = []
        for file_path in self._files.keys():
            # Check name match and if it's within the target directory tree
            if file_path.name == filename and (file_path.parent == dir_path or dir_path in file_path.parents):
                possible_matches.append(file_path)

        if possible_matches:
             # Maybe return the shortest path or first found? Let's return first.
             found_path = possible_matches[0]
             logger.debug(f"FakeStorageBackend: Found '{filename}' at {found_path}")
             return found_path
        logger.debug(f"FakeStorageBackend: '{filename}' not found in {dir_path}")
        return None

    def exists(self, path: Union[str, Path]) -> bool:
        p = self._normalize_path(path)
        # Check if it exists as a file OR an explicitly created directory
        exists = p in self._files or p in self._directories
        # Also consider if it exists implicitly because a file is inside it
        if not exists:
             exists = any(f.parent == p for f in self._files)
        logger.debug(f"FakeStorageBackend: Exists check for {p}: {exists}")
        return exists

    def create_symlink(self, source_path: Union[str, Path], symlink_path: Union[str, Path]) -> bool:
        """Simulate creating a symbolic link in the fake storage system."""
        src = self._normalize_path(source_path)
        dest = self._normalize_path(symlink_path)
        logger.debug(f"FakeStorageBackend: Creating symlink from {src} to {dest}")

        # Check if source exists
        if src not in self._files:
            logger.error(f"FakeStorageBackend: Source file not found for symlink: {src}")
            return False

        # Ensure parent directory exists
        self._add_parents(dest)

        # In a fake system, a symlink is essentially a copy with a reference to the original
        # For simplicity, we'll just copy the content
        self._files[dest] = self._files[src]  # Direct reference, not deepcopy
        logger.debug(f"FakeStorageBackend: Created symlink from {src} to {dest}")
        return True

    def rename(self, old_path: Union[str, Path], new_path: Union[str, Path]) -> bool:
        """Rename a file or directory in the fake storage system."""
        old = self._normalize_path(old_path)
        new = self._normalize_path(new_path)
        logger.debug(f"FakeStorageBackend: Renaming {old} to {new}")

        # Check if source exists
        if old not in self._files:
            logger.error(f"FakeStorageBackend: Source file not found for rename: {old}")
            return False

        # Ensure parent directory exists
        self._add_parents(new)

        # Move the content
        self._files[new] = self._files[old]
        del self._files[old]
        logger.debug(f"FakeStorageBackend: Renamed {old} to {new}")
        return True

    def remove_directory(self, directory_path: Union[str, Path], recursive: bool = True) -> bool:
        """
        Remove a directory and optionally all its contents in the fake storage system.

        Args:
            directory_path: Path to the directory to remove
            recursive: Whether to remove the directory recursively (including all contents)

        Returns:
            True if successful, False otherwise
        """
        dir_path = self._normalize_path(directory_path)
        logger.debug(f"FakeStorageBackend: Removing directory {dir_path} (recursive={recursive})")

        # Check if directory exists
        if dir_path not in self._directories:
            logger.warning(f"FakeStorageBackend: Directory not found for removal: {dir_path}")
            return True  # Consider non-existence as success (idempotent)

        # Check if directory is empty or if recursive removal is requested
        files_in_dir = [f for f in self._files.keys() if dir_path in f.parents or f.parent == dir_path]
        subdirs_in_dir = [d for d in self._directories if dir_path in d.parents]

        if (files_in_dir or subdirs_in_dir) and not recursive:
            logger.error(f"FakeStorageBackend: Cannot remove non-empty directory {dir_path} without recursive=True")
            return False

        # Remove all files in directory and subdirectories
        if recursive:
            for file_path in files_in_dir:
                del self._files[file_path]
                logger.debug(f"FakeStorageBackend: Removed file {file_path} during directory removal")

            # Remove all subdirectories
            for subdir in subdirs_in_dir:
                self._directories.remove(subdir)
                logger.debug(f"FakeStorageBackend: Removed subdirectory {subdir} during directory removal")

        # Remove the directory itself
        self._directories.remove(dir_path)
        logger.debug(f"FakeStorageBackend: Removed directory {dir_path}")
        return True

    # --- MicroscopyStorageBackend Methods (Simplified Fakes) ---

    def load_image(self, file_path: Union[str, Path]) -> Optional[ImageArray]:
        logger.debug(f"FakeStorageBackend: Loading image from {file_path}")
        try:
            # Delegate to basic load and assume it returns correct type if found
            data = self.load(file_path)
            # Basic check - real fake might validate type more strictly
            if isinstance(data, np.ndarray):
                logger.debug(f"FakeStorageBackend: Loaded image data from {file_path}")
                return data
            logger.warning(f"FakeStorageBackend: Loaded data from {file_path} is not np.ndarray (type: {type(data)})")
            return None # Or raise TypeError
        except FileNotFoundError:
            logger.warning(f"FakeStorageBackend: Image file not found: {file_path}")
            return None

    def save_image(self, image: ImageArray, output_path: Union[str, Path], metadata: Optional[Dict] = None) -> bool:
        logger.debug(f"FakeStorageBackend: Saving image to {output_path} (metadata ignored in fake)")
        # Fake doesn't handle metadata explicitly here, just saves the array
        # Delegate to basic save
        return self.save(image, output_path)

    def list_image_files(self, directory: Union[str, Path], extensions: Optional[Set[str]] = None, recursive: bool = True) -> List[Path]:
        """
        List image files in a directory, filtering by specific extensions.

        Args:
            directory: Directory to search.
            extensions: Set of file extensions (e.g., {'.tif', '.png'}).
                        If None, uses DEFAULT_IMAGE_EXTENSIONS from ezstitcher.io.constants.
            recursive: Whether to search recursively.

        Returns:
            List of paths to image files.
        """
        dir_path = self._normalize_path(directory)
        logger.debug(f"FakeStorageBackend: Listing image files in {dir_path} (extensions={extensions}, recursive={recursive})")
        effective_extensions = extensions if extensions is not None else DEFAULT_IMAGE_EXTENSIONS
        # Use our list_files implementation with extension filtering directly
        return self.list_files(directory=dir_path, pattern=None, extensions=effective_extensions, recursive=recursive)

    def find_image_directory(self, plate_folder: Union[str, Path], extensions: Optional[Set[str]] = None) -> Path:
        pf_path = self._normalize_path(plate_folder)
        logger.debug(f"FakeStorageBackend: Finding image directory in {pf_path}")
        # Very basic fake logic:
        # 1. Check if images exist directly in plate_folder
        if self.list_image_files(pf_path, extensions, recursive=False):
             logger.debug(f"FakeStorageBackend: Found images directly in {pf_path}")
             return pf_path
        # 2. Check immediate subdirectories for images
        subdirs = [d for d in self._directories if d.parent == pf_path]
        for subdir in subdirs:
             if self.list_image_files(subdir, extensions, recursive=False):
                 logger.debug(f"FakeStorageBackend: Found images in subdirectory {subdir}")
                 return subdir
        logger.error(f"FakeStorageBackend: Image directory not found in {pf_path} or its immediate subdirectories")
        raise FileNotFoundError(f"Fake image directory not found in {pf_path}")


    def rename_files_with_consistent_padding(self, directory: Union[str, Path], parser: Any, width: int = 3, force_suffixes: bool = False) -> Dict[Path, Path]:
        # Fake implementation: Difficult to truly fake without parser logic.
        # Could list files, attempt basic parsing if simple, and simulate rename in self._files.
        # For now, just log and return empty.
        dir_path = self._normalize_path(directory)
        logger.warning(f"FakeStorageBackend.rename_files_with_consistent_padding not fully implemented for {dir_path}. Returning empty map.")
        # To make it slightly useful, you could potentially iterate self._files in dir_path
        # and create a dummy mapping, but without parser logic it's not accurate.
        return {}

    def detect_zstack_folders(self, plate_folder: Union[str, Path], pattern: Optional[str] = None) -> Tuple[bool, List[Path]]:
        pf_path = self._normalize_path(plate_folder)
        logger.debug(f"FakeStorageBackend: Detecting Z-stack folders in {pf_path} (pattern='{pattern}')")
        # Fake implementation: find immediate subdirectories matching default 'Z' pattern or provided pattern
        z_folders = []
        default_pattern = "Z[0-9]*" # Basic pattern
        match_pattern = pattern if pattern else default_pattern

        subdirs = [d for d in self._directories if d.parent == pf_path]
        for subdir in subdirs:
             # Use fnmatch for simple pattern matching on the directory name
             if fnmatch.fnmatch(subdir.name, match_pattern):
                 z_folders.append(subdir)

        has_zstack = bool(z_folders)
        logger.debug(f"FakeStorageBackend: Detected Z-stack folders ({has_zstack}): {z_folders}")
        return has_zstack, z_folders


    def organize_zstack_folders(self, plate_folder: Union[str, Path], filename_parser: Any) -> bool:
        # Fake implementation: Difficult to fake without parser. Return True, do nothing.
        pf_path = self._normalize_path(plate_folder)
        logger.warning(f"FakeStorageBackend.organize_zstack_folders not implemented for {pf_path}. Returning True.")
        return True

    def mirror_directory_with_symlinks(self, source_dir: Union[str, Path], target_dir: Union[str, Path],
                                      recursive: bool = True, overwrite: bool = True) -> int:
        """
        Simulate mirroring a directory with symlinks in the fake storage system.

        This implementation creates a copy of the file references in the fake storage system,
        simulating the behavior of symlinks without actually creating them.

        Args:
            source_dir: Path to the source directory to mirror
            target_dir: Path to the target directory where the mirrored structure will be created
            recursive: Whether to recursively mirror subdirectories. Defaults to True.
            overwrite: Whether to overwrite the target directory if it exists. Defaults to True.

        Returns:
            int: Number of simulated symlinks created
        """
        source_dir_path = self._normalize_path(source_dir)
        target_dir_path = self._normalize_path(target_dir)

        logger.debug(f"FakeStorageBackend: Mirroring directory {source_dir_path} to {target_dir_path}")

        # Check if source directory exists in our fake filesystem
        if not self.exists(source_dir_path):
            logger.error(f"FakeStorageBackend: Source directory not found: {source_dir_path}")
            return 0

        # Handle overwrite behavior
        if self.exists(target_dir_path) and overwrite:
            logger.info(f"FakeStorageBackend: Removing existing target directory: {target_dir_path}")
            # Remove all files in target directory and subdirectories
            files_to_remove = [f for f in self._files.keys()
                              if f == target_dir_path or target_dir_path in f.parents]
            for file_path in files_to_remove:
                del self._files[file_path]

            # Remove all directories in target directory and subdirectories
            dirs_to_remove = [d for d in self._directories
                             if d == target_dir_path or target_dir_path in d.parents]
            for dir_path in dirs_to_remove:
                self._directories.remove(dir_path)

        # Ensure target directory exists
        self.ensure_directory(target_dir_path)

        # Count of simulated symlinks created
        symlinks_created = 0

        # Get all files in source directory
        source_files = self.list_files(source_dir_path, recursive=recursive)

        # Create simulated symlinks (copies in our fake filesystem)
        for source_file in source_files:
            # Calculate relative path from source_dir to the file
            try:
                rel_path = source_file.relative_to(source_dir_path)
                target_file = target_dir_path / rel_path

                # Create parent directories if needed
                self.ensure_directory(target_file.parent)

                # Copy the file (simulating a symlink)
                if source_file in self._files:
                    self._files[target_file] = self._files[source_file]  # Direct reference, not deepcopy
                    symlinks_created += 1
                    logger.debug(f"FakeStorageBackend: Created symlink from {source_file} to {target_file}")
            except ValueError:
                # This happens if source_file is not relative to source_dir_path
                logger.error(f"FakeStorageBackend: Error calculating relative path for {source_file}")

        logger.info(f"FakeStorageBackend: Created {symlinks_created} symlinks from {source_dir_path} to {target_dir_path}")
        return symlinks_created