"""
Overlay architecture for filesystem-only backends.

This module provides support for temporarily writing in-memory data to disk
for components that require filesystem access, while maintaining the stateless
design pattern and clean separation of concerns.
"""

from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Literal, Callable, Protocol
import logging
import numpy as np

logger = logging.getLogger(__name__)

class OverlayMode(Enum):
    """Modes for overlay disk writes."""
    DISABLED = auto()  # No overlay writes
    ON_DEMAND = auto()  # Write to disk only when explicitly requested
    AUTO = auto()       # Automatically write to disk for filesystem-dependent operations

class ArraySupplier(Protocol):
    """Protocol for lazy array suppliers."""
    def __call__(self) -> np.ndarray: ...

class OverlayOperation:
    """Represents an overlay disk write operation with lazy data loading."""
    def __init__(
        self,
        key: str,
        data_supplier: ArraySupplier,
        disk_path: Path,
        operation_type: Literal["read", "write", "both"] = "both",
        cleanup: bool = True
    ):
        self.key = key
        self.data_supplier = data_supplier
        self.disk_path = disk_path
        self.operation_type = operation_type
        self.cleanup = cleanup
        self.executed = False
        
    def __repr__(self):
        return f"OverlayOperation(key={self.key}, path={self.disk_path}, type={self.operation_type}, cleanup={self.cleanup})"
