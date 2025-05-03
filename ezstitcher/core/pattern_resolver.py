"""
Pattern resolution utilities for EZStitcher.

Provides functions for resolving image patterns from microscope data.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

@runtime_checkable
class PatternDetector(Protocol):
    """Protocol compatible with MicroscopeHandler and subclasses."""

    def auto_detect_patterns(
        self,
        directory: Union[str, Path],
        well_filter: Optional[List[str]] = None,
        variable_components: Optional[List[str]] = None,
        group_by: Optional[str] = None,
        flat: bool = False
    ) -> Dict[str, Any]:
        """Detect patterns in the given directory."""
        ...


def get_patterns_for_well(
    well: str,
    directory: Union[str, Path],
    detector: PatternDetector,
    variable_components: Optional[List[str]] = None,
    recursive: bool = False
) -> List[str]:
    """
    Get flattened list of patterns for a specific well.

    Args:
        well: Well identifier (e.g., 'A01')
        directory: Directory to search for patterns
        detector: Object implementing PatternDetector (e.g., MicroscopeHandler)
        variable_components: Components that vary across files (default: ['site'])
        recursive: Whether to scan subdirectories recursively

    Returns:
        List of patterns for the well

    Example:
        patterns = get_patterns_for_well('A01', input_dir, microscope_handler)
    """
    if variable_components is None:
        variable_components = ['site']

    directory = Path(directory) if isinstance(directory, str) else directory

    try:
        # Pass recursive parameter to auto_detect_patterns if it accepts it
        try:
            patterns_by_well = detector.auto_detect_patterns(
                directory,
                well_filter=[well],
                variable_components=variable_components,
                flat=False,  # Handle flattening ourselves
                recursive=recursive  # Pass recursive parameter
            )
        except TypeError:
            # Fallback for detectors that don't support recursive parameter
            logger.debug("Detector doesn't support recursive parameter, using default implementation")
            patterns_by_well = detector.auto_detect_patterns(
                directory,
                well_filter=[well],
                variable_components=variable_components,
                flat=False  # Handle flattening ourselves
            )

        all_patterns = []
        if patterns_by_well and well in patterns_by_well:
            well_patterns = patterns_by_well[well]
            if isinstance(well_patterns, dict):
                # Grouped patterns (by channel, z-index, etc.)
                for _, patterns in well_patterns.items():
                    if isinstance(patterns, list):
                        all_patterns.extend(patterns)
            elif isinstance(well_patterns, list):
                # Flat list of patterns
                all_patterns.extend(well_patterns)

        # If recursive is True and no patterns were found, try to scan subdirectories manually
        if recursive and not all_patterns and hasattr(detector, 'parser') and hasattr(detector.parser, 'path_list_from_pattern'):
            logger.debug("No patterns found with auto_detect_patterns, trying manual recursive scan")
            try:
                # Try to find subdirectories
                if hasattr(detector, 'file_manager'):
                    file_manager = detector.file_manager
                    subdirs = [d for d in file_manager.list_files(directory, recursive=False) if d.is_dir()]

                    # For each subdirectory, try to find patterns
                    for subdir in subdirs:
                        logger.debug(f"Scanning subdirectory: {subdir}")
                        subdir_patterns = get_patterns_for_well(well, subdir, detector, variable_components, False)
                        all_patterns.extend(subdir_patterns)
            except Exception as subdir_error:
                logger.debug(f"Error during manual recursive scan: {subdir_error}")

        return all_patterns

    except Exception as e:
        logger.warning(f"Error getting patterns for well {well}: {e}")
        return []
