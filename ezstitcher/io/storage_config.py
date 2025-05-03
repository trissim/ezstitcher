"""
Storage configuration for EZStitcher.

This module provides a configuration class for storage-related settings in EZStitcher.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from ezstitcher.io.overlay import OverlayMode


@dataclass
class StorageConfig:
    """
    Configuration for storage-related settings.
    
    This class encapsulates storage-related settings such as storage mode,
    overlay mode, and overlay root directory.
    """
    
    storage_mode: str = "legacy"
    overlay_mode: OverlayMode = OverlayMode.DISABLED
    overlay_root: Optional[Path] = None
