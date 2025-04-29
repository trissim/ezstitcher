"""
Utility functions for the EZ module.

This module provides helper functions for the EZ module.
"""

from pathlib import Path
from typing import List, Optional, Union, Dict, Any

def detect_wells(plate_path: Union[str, Path]) -> List[str]:
    """
    Detect available wells in a plate.
    
    Args:
        plate_path: Path to the plate folder
        
    Returns:
        List[str]: List of well identifiers
    """
    # Implementation using microscope handler
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    orchestrator = PipelineOrchestrator(plate_path=plate_path)
    return orchestrator._get_wells_to_process()

def suggest_channel_weights(plate_path: Union[str, Path]) -> Optional[List[float]]:
    """
    Suggest channel weights based on plate content.
    
    Args:
        plate_path: Path to the plate folder
        
    Returns:
        List[float] or None: Suggested channel weights
    """
    # Implementation using EZStitcher's detection
    from .core import EZStitcher
    stitcher = EZStitcher(plate_path)
    return stitcher.channel_weights

def create_config(input_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """
    Create a configuration dictionary for stitching based on input data.
    
    Analyzes the input data and suggests appropriate configuration.
    
    Args:
        input_path: Path to the plate folder
        **kwargs: User overrides for auto-detected settings
        
    Returns:
        dict: Configuration dictionary
    """
    # Convert to Path
    input_path = Path(input_path)
    
    # Create base configuration
    config = {
        "input_path": input_path,
        "output_path": input_path.parent / f"{input_path.name}_stitched",
        "normalize": True,
        "flatten_z": None,  # Will be auto-detected
        "z_method": "max",
        "channel_weights": None,  # Will be auto-detected
        "well_filter": None
    }
    
    # Create temporary EZStitcher to detect parameters
    from .core import EZStitcher
    temp_stitcher = EZStitcher(input_path)
    
    # Update with auto-detected values
    config["flatten_z"] = temp_stitcher.flatten_z
    config["channel_weights"] = temp_stitcher.channel_weights
    
    # Override with user-provided values
    config.update(kwargs)
    
    return config
