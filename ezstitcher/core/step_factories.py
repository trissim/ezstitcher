"""
Step factory classes for the EZStitcher pipeline architecture.

This module contains specialized step implementations that inherit from the regular Step class
and pre-configure parameters for common operations like Z-stack flattening, focus selection,
and channel compositing.
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from ezstitcher.core.steps import Step
from ezstitcher.core.image_processor import ImageProcessor as IP
from ezstitcher.core.focus_analyzer import FocusAnalyzer


class ZFlatStep(Step):
    """
    Specialized step for Z-stack flattening.

    This step performs Z-stack flattening using the specified method.
    It pre-configures variable_components=['z_index'] and group_by=None.
    """

    PROJECTION_METHODS = {
        "max": "max_projection",
        "mean": "mean_projection",
        "median": "median_projection",
        "min": "min_projection",
        "std": "std_projection",
        "sum": "sum_projection"
    }

    def __init__(
        self,
        method: str = "max",
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        well_filter: Optional[List[str]] = None,
    ):
        """
        Initialize a Z-stack flattening step.

        Args:
            method: Projection method. Options: "max", "mean", "median", "min", "std", "sum"
            input_dir: Input directory
            output_dir: Output directory
            well_filter: Wells to process
        """
        # Validate method
        if method not in self.PROJECTION_METHODS and method not in self.PROJECTION_METHODS.values():
            raise ValueError(f"Unknown projection method: {method}. "
                            f"Options are: {', '.join(self.PROJECTION_METHODS.keys())}")

        # Get the full method name if a shorthand was provided
        self.method = method
        full_method = self.PROJECTION_METHODS.get(method, method)

        # Initialize the Step with pre-configured parameters
        super().__init__(
            func=(IP.create_projection, {'method': full_method}),
            variable_components=['z_index'],
            group_by=None,
            input_dir=input_dir,
            output_dir=output_dir,
            well_filter=well_filter,
            name=f"{method.capitalize()} Projection"
        )


class FocusStep(Step):
    """
    Specialized step for focus-based Z-stack processing.

    This step finds the best focus plane in a Z-stack using a FocusAnalyzer.
    It pre-configures variable_components=['z_index'] and group_by=None.
    """

    def __init__(
        self,
        focus_options: Optional[Dict[str, Any]] = None,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        well_filter: Optional[List[str]] = None,
    ):
        """
        Initialize a focus step.

        Args:
            focus_options: Dictionary of focus analyzer options:
                - metric: Focus metric. Options: "combined", "normalized_variance",
                         "laplacian", "tenengrad", "fft" (default: "combined")
                - roi: Region of interest as (x, y, width, height)
                - weights: Weights for combined focus measure
            input_dir: Input directory
            output_dir: Output directory
            well_filter: Wells to process
        """
        # Initialize focus options
        focus_options = focus_options or {'metric': 'combined'}

        # Create focus analyzer
        focus_analyzer = FocusAnalyzer(
            metric=focus_options.get('metric', 'combined'),
            roi=focus_options.get('roi'),
            weights=focus_options.get('weights')
        )

        # Initialize the Step with pre-configured parameters
        super().__init__(
            func=(IP.create_projection, {'method': 'best_focus', 'focus_analyzer': focus_analyzer}),
            variable_components=['z_index'],
            group_by=None,
            input_dir=input_dir,
            output_dir=output_dir,
            well_filter=well_filter,
            name=f"Best Focus ({focus_options.get('metric', 'combined')})"
        )


class CompositeStep(Step):
    """
    Specialized step for creating composite images from multiple channels.

    This step creates composite images from multiple channels with specified weights.
    It pre-configures variable_components=['channel'] and group_by=None.
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        well_filter: Optional[List[str]] = None,
    ):
        """
        Initialize a channel compositing step.

        Args:
            weights: List of weights for each channel. If None, equal weights are used.
            input_dir: Input directory
            output_dir: Output directory
            well_filter: Wells to process
        """
        # Initialize the Step with pre-configured parameters
        super().__init__(
            func=(IP.create_composite, {'weights': weights}),
            variable_components=['channel'],
            group_by=None,
            input_dir=input_dir,
            output_dir=output_dir,
            well_filter=well_filter,
            name="Channel Composite"
        )
