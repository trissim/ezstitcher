"""
Integration example for the Flexible Pipeline Architecture.

This module demonstrates how to integrate the new Pipeline architecture
with the existing PipelineOrchestrator.
"""

from typing import Dict, List, Any
import logging
from pathlib import Path

from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.pipeline import step, pipeline
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.focus_analyzer import FocusAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the processing pipeline for microscopy images.

    This class coordinates the processing of microscopy images, including
    reference image processing, position generation, and final image assembly.

    Attributes:
        config: Configuration parameters
        microscope_handler: Handler for the microscope type
        image_preprocessor: Image preprocessing component
        stitcher: Image stitching component
        focus_analyzer: Focus analysis component
    """

    def __init__(self, config=None):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Configuration parameters
        """
        self.config = config or PipelineConfig()
        self.microscope_handler = None
        self.image_preprocessor = None
        self.stitcher = None
        self.focus_analyzer = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run(self, plate_folder):
        """
        Run the pipeline on a plate folder.

        Args:
            plate_folder: Path to the plate folder

        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing plate: {plate_folder}")

        # Initialize components
        self._init_components(plate_folder)

        # Initialize workspace
        workspace_path = self.microscope_handler.init_workspace(plate_folder, None)
        input_dir = self._prepare_images(workspace_path)
        dirs = self._setup_directories(workspace_path, input_dir)

        # Get wells to process
        wells = self._get_wells_to_process(dirs['input'])
        self.logger.info(f"Processing {len(wells)} wells: {wells}")

        # Process wells
        results = {}

        # Determine number of worker threads
        effective_workers = min(self.config.num_workers, len(wells))
        self.logger.info(f"Using {effective_workers} worker threads")

        if effective_workers > 1:
            # Process wells in parallel
            import concurrent.futures
            import copy

            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                # Submit all well processing tasks
                futures = {}
                for well in wells:
                    # Create a deep copy of dirs for each well to avoid shared state issues
                    well_dirs = copy.deepcopy(dirs)
                    future = executor.submit(self.process_well, well, well_dirs)
                    futures[future] = well

                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    well = futures[future]
                    try:
                        result = future.result()
                        results[well] = result
                        self.logger.info(f"Completed processing well: {well}")
                    except Exception as e:
                        self.logger.error(f"Error processing well {well}: {e}")
                        results[well] = {"error": str(e)}
        else:
            # Process wells sequentially
            for well in wells:
                try:
                    results[well] = self.process_well(well, dirs)
                    self.logger.info(f"Completed processing well: {well}")
                except Exception as e:
                    self.logger.error(f"Error processing well {well}: {e}")
                    results[well] = {"error": str(e)}

        # Clean up if needed
        if self.config.cleanup_processed:
            self._cleanup_processed(dirs['processed'])

        if self.config.cleanup_post_processed:
            self._cleanup_processed(dirs['post_processed'])

        self.logger.info(f"Completed processing plate: {plate_folder}")
        return results

    def process_well(self, well, dirs):
        """
        Process a single well.

        Args:
            well: The well to process
            dirs: Directory paths

        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing well: {well}")

        # Process reference images
        reference_results = self.process_reference_images(well, dirs)

        # Generate positions
        positions_results = self.generate_positions(well, dirs)

        # Process final images
        final_results = self.process_final_images(well, dirs)

        # Stitch images
        stitched_results = self.stitch_images(well, dirs)

        return {
            "reference": reference_results,
            "positions": positions_results,
            "final": final_results,
            "stitched": stitched_results
        }

    def process_reference_images(self, well, dirs):
        """
        Process reference images using the new Pipeline architecture.

        Args:
            well: The well to process
            dirs: Directory paths

        Returns:
            The results of reference image processing
        """
        # Create reference pipeline
        reference_pipeline = (
            pipeline(
                # Flatten Z-stacks
                step(
                    func=self.image_preprocessor.create_projection,
                    variable_components=['z_index'],
                    processing_args={
                        'method': self.config.reference_flatten,
                        'focus_analyzer': self.focus_analyzer
                    },
                    name="Z-Stack Flattening"
                ),

                # Process channels
                step(
                    func=self.config.reference_processing,
                    variable_components=['site'],
                    group_by='channel',
                    name="Channel Processing"
                ),

                # Create composites
                step(
                    func=self.image_preprocessor.create_composite,
                    variable_components=['channel'],
                    group_by='site',
                    processing_args={'weights': self.config.reference_composite_weights},
                    name="Composite Creation"
                )
            )
            .set_input(dirs['input'])
            .set_output(dirs['processed'])
            .set_well_filter([well])
        )

        return reference_pipeline.run()

    def generate_positions(self, well, dirs):
        """
        Generate positions for stitching.

        Args:
            well: The well to process
            dirs: Directory paths

        Returns:
            Tuple of (positions_file, stitch_pattern)
        """
        # Create position generation pipeline
        position_pipeline = (
            pipeline(
                step(
                    func=self.stitcher.generate_positions,
                    variable_components=['site'],
                    processing_args={
                        'grid_size_x': self.config.grid_size[0],
                        'grid_size_y': self.config.grid_size[1],
                        'reference_channels': self.config.reference_channels,
                        'reference_composite_weights': self.config.reference_composite_weights
                    },
                    name="Position Generation"
                )
            )
            .set_input(dirs['processed'])
            .set_output(dirs['positions'])
            .set_well_filter([well])
        )

        results = position_pipeline.run()

        # Extract positions_file and stitch_pattern from results
        positions_file = dirs['positions'] / f"{well}.csv"
        stitch_pattern = results.get(well, {}).get('pattern', None)

        return positions_file, stitch_pattern

    def process_final_images(self, well, dirs):
        """
        Process final images using the new Pipeline architecture.

        Args:
            well: The well to process
            dirs: Directory paths

        Returns:
            The results of final image processing
        """
        # Create final processing pipeline
        final_pipeline = (
            pipeline(
                # Flatten Z-stacks
                step(
                    func=self.image_preprocessor.create_projection,
                    variable_components=['z_index'],
                    processing_args={
                        'method': self.config.stitch_flatten,
                        'focus_analyzer': self.focus_analyzer
                    },
                    name="Z-Stack Flattening"
                ),

                # Process channels
                step(
                    func=self.config.final_processing,
                    variable_components=['site'],
                    group_by='channel',
                    name="Channel Processing"
                )
            )
            .set_input(dirs['input'])
            .set_output(dirs['post_processed'])
            .set_well_filter([well])
        )

        return final_pipeline.run()

    def stitch_images(self, well, dirs, positions_file=None, stitcher=None):
        """
        Stitch images using the new Pipeline architecture.

        Args:
            well: The well to process
            dirs: Dictionary of directories
            positions_file: Path to positions file (optional)
            stitcher: Optional Stitcher instance to use (for thread safety)

        Returns:
            The results of image stitching
        """
        # Use the provided stitcher or the default one
        stitcher_to_use = stitcher or self.stitcher

        # If positions_file is not provided, construct it from the positions directory
        if positions_file is None:
            positions_file = dirs['positions'] / f"{well}.csv"

        # Create stitching pipeline
        stitching_pipeline = (
            pipeline(
                step(
                    func=stitcher_to_use.assemble_image,
                    variable_components=['site'],
                    processing_args={
                        'positions_file': positions_file,
                        'channels': self.config.stitch_channels
                    },
                    name="Image Assembly"
                )
            )
            .set_input(dirs['post_processed'])
            .set_output(dirs['stitched'])
            .set_well_filter([well])
        )

        return stitching_pipeline.run()

    # Helper methods (unchanged from original implementation)
    def _init_components(self, plate_folder):
        # Implementation unchanged
        pass

    def _prepare_images(self, workspace_path):
        # Implementation unchanged
        pass

    def _setup_directories(self, workspace_path, input_dir):
        # Implementation unchanged
        pass

    def _get_wells_to_process(self, input_dir):
        # Implementation unchanged
        pass

    def _cleanup_processed(self, directory):
        # Implementation unchanged
        pass


# Example usage

def main():
    """
    Example usage of the PipelineOrchestrator with the new Pipeline architecture.
    """
    # Create configuration
    config = PipelineConfig(
        reference_channels=["1", "2"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        reference_composite_weights={"1": 0.7, "2": 0.3},
        num_workers=1,
        cleanup_processed=False,
        cleanup_post_processed=False
    )

    # Set stitch_channels as an attribute
    config.stitch_channels = ["1", "2"]

    # Create reference processing functions
    config.reference_processing = {
        "1": ImagePreprocessor.percentile_normalize,
        "2": ImagePreprocessor.tophat
    }

    # Create final processing functions
    config.final_processing = {
        "1": [
            ImagePreprocessor.percentile_normalize,
            ImagePreprocessor.tophat
        ],
        "2": [
            ImagePreprocessor.percentile_normalize,
            ImagePreprocessor.normalize
        ]
    }

    # Create pipeline orchestrator
    orchestrator = PipelineOrchestrator(config)

    # Run pipeline
    plate_folder = "/path/to/plate"
    results = orchestrator.run(plate_folder)

    # Print results
    print(f"Processed {len(results)} wells")
    for well, well_results in results.items():
        print(f"Well {well}: {len(well_results)} results")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run example
    main()
