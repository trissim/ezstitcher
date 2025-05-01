# ezstitcher/io/__init__.py
"""
I/O module for ezstitcher.

Provides interfaces and implementations for interacting with various storage systems,
along with related types and constants.
"""

# Try importing components, handle ImportError gracefully if files don't exist yet
try:
    from .types import ImageArray
except ImportError:
    ImageArray = None # Or define a placeholder type

try:
    from .constants import DEFAULT_IMAGE_EXTENSIONS
except ImportError:
    DEFAULT_IMAGE_EXTENSIONS = None # Or define a placeholder set

try:
    from .storage_backend import (
        BasicStorageBackend,
        MicroscopyStorageBackend,
        DiskStorageBackend,
        FakeStorageBackend
    )
except ImportError:
    # Define placeholder classes if the main file doesn't exist yet
    class BasicStorageBackend: pass
    class MicroscopyStorageBackend(BasicStorageBackend): pass
    class DiskStorageBackend(MicroscopyStorageBackend): pass
    class FakeStorageBackend(MicroscopyStorageBackend): pass

try:
    from .filemanager import FileManager
except ImportError:
    class FileManager: pass # Placeholder

try:
    from .directory_mirror import (
        mirror_directory_with_symlinks,
        OverwriteStrategy
    )
except ImportError:
    # Placeholder function with same signature as the real one
    def mirror_directory_with_symlinks(
        source_dir=None, target_dir=None, recursive=True,
        strategy=None, progress_callback=None, verbose=False
    ):
        """Placeholder for mirror_directory_with_symlinks."""
        return 0

    # Placeholder enum
    class OverwriteStrategy:
        """Placeholder for OverwriteStrategy enum."""
        REPLACE = "replace"
        SKIP = "skip"
        MERGE = "merge"


__all__ = [
    # Types
    'ImageArray',
    # Constants
    'DEFAULT_IMAGE_EXTENSIONS',
    # Interfaces
    'BasicStorageBackend',
    'MicroscopyStorageBackend',
    # Implementations
    'DiskStorageBackend',
    'FakeStorageBackend',
    # Manager
    'FileManager',
    # Directory mirroring utilities
    'mirror_directory_with_symlinks',
    'OverwriteStrategy',
]

# Clean up namespace if imports failed
if ImageArray is None:
    if 'ImageArray' in __all__:
        __all__.remove('ImageArray')
if DEFAULT_IMAGE_EXTENSIONS is None:
    if 'DEFAULT_IMAGE_EXTENSIONS' in __all__:
        __all__.remove('DEFAULT_IMAGE_EXTENSIONS')
# Add similar checks for classes if necessary
if 'FileManager' in __all__ and not hasattr(FileManager, '__init__'):  # Basic check for placeholder
    __all__.remove('FileManager')

# Add checks for directory_mirror components
if 'mirror_directory_with_symlinks' in __all__ and not hasattr(
        mirror_directory_with_symlinks, '__module__'):
    __all__.remove('mirror_directory_with_symlinks')
if 'OverwriteStrategy' in __all__ and not hasattr(
        OverwriteStrategy, '__module__'):
    __all__.remove('OverwriteStrategy')
