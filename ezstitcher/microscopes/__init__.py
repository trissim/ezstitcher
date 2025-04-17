"""
Microscope-specific implementations for ezstitcher.

This package contains modules for different microscope types, each providing
concrete implementations of FilenameParser and MetadataHandler interfaces.
"""

# Import microscope handlers for easier access
from ezstitcher.microscopes.imagexpress import ImageXpressFilenameParser, ImageXpressMetadataHandler
from ezstitcher.microscopes.opera_phenix import OperaPhenixFilenameParser, OperaPhenixMetadataHandler
