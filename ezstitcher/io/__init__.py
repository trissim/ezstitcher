# ezstitcher/io/__init__.py
"""
I/O module for ezstitcher.

Provides interfaces and implementations for interacting with various storage systems,
along with related types and constants.
"""

# Direct imports - ImportError will propagate if components are missing
from .types import ImageArray
from .constants import DEFAULT_IMAGE_EXTENSIONS
from .storage_backend import (
    BasicStorageBackend,
    MicroscopyStorageBackend,
    DiskStorageBackend,
    FakeStorageBackend
)
from .filemanager import FileManager
from .directory_mirror import (
    mirror_directory_with_symlinks,
    OverwriteStrategy
)
from .storage_adapter import (
    StorageAdapter,
    MemoryStorageAdapter,
    ZarrStorageAdapter,
    select_storage
)
from .storage_config import StorageConfig
from .overlay import OverlayMode, OverlayOperation
from .virtual_path import VirtualPath
from .virtual_path_factory import VirtualPathFactory

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
    # Storage adapters
    'StorageAdapter',
    'MemoryStorageAdapter',
    'ZarrStorageAdapter',
    'select_storage',
    'StorageConfig',
    # Overlay
    'OverlayMode',
    'OverlayOperation',
    # Virtual paths
    'VirtualPath',
    'VirtualPathFactory',
]

# No cleanup needed as ImportError will prevent module loading if imports fail
