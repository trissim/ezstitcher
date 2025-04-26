"""Context class for sharing data between pipeline steps."""

from typing import Dict, Any, Optional, List
from pathlib import Path

class Context:
    """Context object for sharing data between pipeline steps."""
    
    def __init__(self, 
                 input_dir: Path,
                 output_dir: Path,
                 well_filter: Optional[List[str]] = None,
                 orchestrator: Optional[Any] = None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.well_filter = well_filter
        self.orchestrator = orchestrator
        self.results: Dict[str, Any] = {}