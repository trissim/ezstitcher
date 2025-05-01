"""
Core implementation of the EZ module.

This module provides the EZStitcher class, which is a simplified interface
for common stitching workflows.
"""

from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from ezstitcher.core import AutoPipelineFactory
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator


class EZStitcher:
    """
    Simplified interface for microscopy image stitching.

    This class provides an easy-to-use interface for common stitching workflows,
    hiding the complexity of pipelines and orchestrators.
    """

    def __init__(self,
                 input_path: Union[str, Path],
                 output_path: Optional[Union[str, Path]] = None,
                 normalize: bool = True,
                 flatten_z: Optional[bool] = None,
                 z_method: str = "max",
                 channel_weights: Optional[List[float]] = None,
                 well_filter: Optional[List[str]] = None):
        """
        Initialize with minimal required parameters.

        Args:
            input_path: Path to the plate folder
            output_path: Path for output (default: input_path + "_stitched")
            normalize: Whether to apply normalization
            flatten_z: Whether to flatten Z-stacks (auto-detected if None)
            z_method: Method for Z-flattening ("max", "mean", "focus", etc.)
            channel_weights: Weights for channel compositing (auto-detected if None)
            well_filter: List of wells to process (processes all if None)
        """
        self.input_path = Path(input_path)

        # Auto-generate output path if not provided
        if output_path is None:
            self.output_path = self.input_path.parent / f"{self.input_path.name}_stitched"
        else:
            self.output_path = Path(output_path)

        # Store basic configuration
        self.normalize = normalize
        self.z_method = z_method
        self.well_filter = well_filter

        # Create orchestrator
        self.orchestrator = PipelineOrchestrator(plate_path=self.input_path)

        # Set parameters (no auto-detection for now)
        # Default to False if flatten_z is None (no auto-detection)
        self.flatten_z = False if flatten_z is None else flatten_z
        self.channel_weights = channel_weights

        # Create factory with current configuration
        self._create_factory()

    def _create_factory(self):
        """Create AutoPipelineFactory with current configuration."""
        self.factory = AutoPipelineFactory(
            input_dir=self.orchestrator.workspace_path,
            output_dir=self.output_path,
            normalize=self.normalize,
            flatten_z=self.flatten_z,
            z_method=self.z_method,
            channel_weights=self.channel_weights,
            well_filter=self.well_filter
        )

    def set_options(self, **kwargs):
        """
        Update configuration options.

        Args:
            **kwargs: Configuration options to update

        Returns:
            self: For method chaining
        """
        # Update attributes
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown option: {key}")

        # Recreate factory with updated configuration
        self._create_factory()

        return self

    def stitch(self):
        """
        Run the complete stitching process with current settings.

        Returns:
            Path: Path to the output directory
        """
        # Create pipelines
        pipelines = self.factory.create_pipelines()

        # Run pipelines
        self.orchestrator.run(pipelines=pipelines)

        return self.output_path
