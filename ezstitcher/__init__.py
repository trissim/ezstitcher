"""EZStitcher: An easy-to-use microscopy image stitching and processing tool."""

__version__ = "1.1.0"

# Import simplified interface
from ezstitcher.ez import stitch_plate, EZStitcher

__all__ = [
    'stitch_plate',
    'EZStitcher',
]
