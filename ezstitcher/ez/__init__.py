"""
EZ module for simplified access to ezstitcher functionality.

This module provides a simplified interface for common stitching workflows,
making it easier for non-coders to use ezstitcher.
"""

from .core import EZStitcher
from .functions import stitch_plate

__all__ = [
    'EZStitcher',
    'stitch_plate',
]
