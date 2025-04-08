import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from ezstitcher.core.config import PlateProcessorConfig
from ezstitcher.core.zstack_processor import ZStackProcessor
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.utils import ensure_directory, path_list_from_pattern

logger = logging.getLogger(__name__)

class PlateProcessor:
    """
    High-level orchestrator for processing a microscopy plate.
    Coordinates Z-stack handling, stitching, and output management.
    """
    def __init__(self, config: PlateProcessorConfig):
        self.config = config
        self.zstack_processor = ZStackProcessor(config.z_stack_processor)
        self.focus_analyzer = FocusAnalyzer(config.focus_analyzer)
        self.image_preprocessor = ImagePreprocessor(config.image_preprocessor)
        self.stitcher = Stitcher(config.stitcher)

    def find_HTD_file(self, plate_path):
        """
        Find the HTD file for a plate.

        Args:
            plate_path (Path): Path to the plate folder

        Returns:
            Path or None: Path to the HTD file, or None if not found
        """
        # Look in plate directory
        htd_files = list(plate_path.glob("*.HTD"))
        if htd_files:
            for htd_file in htd_files:
                if 'plate' in htd_file.name.lower():
                    return htd_file
            return htd_files[0]

        # Look in parent directory
        parent_dir = plate_path.parent
        htd_files = list(parent_dir.glob("*.HTD"))
        if htd_files:
            for htd_file in htd_files:
                if 'plate' in htd_file.name.lower():
                    return htd_file
            return htd_files[0]

        return None

    def parse_HTD_file(self, htd_path):
        """
        Parse an HTD file to extract grid dimensions.

        Args:
            htd_path (Path): Path to the HTD file

        Returns:
            tuple: (grid_size_x, grid_size_y) or None if parsing fails
        """
        try:
            with open(htd_path, 'r') as f:
                htd_content = f.read()

            # Extract grid dimensions - try multiple formats
            # First try the new format with "XSites" and "YSites"
            cols_match = re.search(r'"XSites", (\d+)', htd_content)
            rows_match = re.search(r'"YSites", (\d+)', htd_content)

            # If not found, try the old format with SiteColumns and SiteRows
            if not (cols_match and rows_match):
                cols_match = re.search(r'SiteColumns=(\d+)', htd_content)
                rows_match = re.search(r'SiteRows=(\d+)', htd_content)

            # If still not found, try looking for GridSizeX and GridSizeY
            if not (cols_match and rows_match):
                cols_match = re.search(r'GridSizeX,(\d+)', htd_content)
                rows_match = re.search(r'GridSizeY,(\d+)', htd_content)

            if cols_match and rows_match:
                grid_size_x = int(cols_match.group(1))
                grid_size_y = int(rows_match.group(1))
                return grid_size_x, grid_size_y

        except Exception as e:
            logger.error(f"Error parsing HTD file: {e}")

        return None

    def run(self, plate_folder, reference_channels=['1'], preprocessing_funcs=None, margin_ratio=0.1, composite_weights=None, well_filter=None, tile_overlap=6.5, tile_overlap_x=None, tile_overlap_y=None, max_shift=50, focus_detect=False, focus_method="combined", create_projections=False, stitch_z_reference='best_focus', save_projections=True, stitch_all_z_planes=False, use_reference_positions=False):
        """
        Instance method version of process_plate_folder.
        """
        try:
            plate_path = Path(plate_folder)
            parent_dir = plate_path.parent
            plate_name = plate_path.name

            # Create directory paths
            processed_dir = parent_dir / f"{plate_name}_processed" / "TimePoint_1"
            positions_dir = parent_dir / f"{plate_name}_positions"
            stitched_dir = parent_dir / f"{plate_name}_stitched" / "TimePoint_1"

            # Ensure directories exist
            processed_dir.mkdir(parents=True, exist_ok=True)
            positions_dir.mkdir(parents=True, exist_ok=True)
            stitched_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Created processed directory: {processed_dir}")
            logger.info(f"Created positions directory: {positions_dir}")
            logger.info(f"Created stitched directory: {stitched_dir}")

            dirs = {
                'input': plate_path / "TimePoint_1",
                'processed': processed_dir,
                'positions': positions_dir,
                'stitched': stitched_dir
            }

            if not dirs['input'].exists():
                logger.error(f"Input directory does not exist: {dirs['input']}")
                return False

            # 1. Detect and handle Z-stacks
            has_zstack = self.zstack_processor.detect_z_stacks(plate_folder)

            best_focus_dir = None
            projections_dir = None

            if has_zstack:
                logger.info(f"Z-stack detected in {plate_folder}")

                if focus_detect:
                    best_focus_dir = parent_dir / f"{plate_name}_best_focus"
                    ensure_directory(best_focus_dir)

                    logger.info(f"Finding best focused images using method: {focus_method}")
                    success, _ = self.zstack_processor.create_best_focus_images(
                        dirs['input'],
                        best_focus_dir,
                        focus_method=focus_method,
                        focus_wavelength=reference_channels[0]
                    )

                    # Ensure TimePoint_1 directory exists in best_focus_dir
                    timepoint_dir = best_focus_dir / "TimePoint_1"
                    if not timepoint_dir.exists():
                        logger.error(f"TimePoint_1 directory not created in {best_focus_dir}")
                        return False
                    if not success:
                        logger.warning("No best focus images created")

                if create_projections:
                    projections_dir = parent_dir / f"{plate_name}_{stitch_z_reference}" / "TimePoint_1"
                    ensure_directory(projections_dir)

                    logger.info(f"Creating projection: {stitch_z_reference}")
                    success, _ = self.zstack_processor.create_zstack_projections(
                        dirs['input'],
                        projections_dir,
                        projection_types=[stitch_z_reference]
                    )
                    if not success:
                        logger.warning("No projections created")

                stitch_source = plate_folder
                if stitch_z_reference == 'best_focus' and best_focus_dir:
                    stitch_source = best_focus_dir.parent
                    logger.info(f"Using best focus images for stitching from {best_focus_dir}")
                elif stitch_z_reference in ['max', 'mean'] and projections_dir:
                    stitch_source = projections_dir.parent
                    logger.info(f"Using {stitch_z_reference} projections for stitching from {projections_dir}")
                elif stitch_z_reference in ['max', 'mean', 'best_focus'] and stitch_all_z_planes:
                    logger.info(f"Stitching all Z-planes using {stitch_z_reference} as reference")
                    success = self.zstack_processor.stitch_across_z(
                        plate_folder,
                        reference_z=stitch_z_reference,
                        stitch_all_z_planes=True,
                        process_plate_folder=self.run,
                        reference_channels=reference_channels,
                        preprocessing_funcs=preprocessing_funcs,
                        margin_ratio=margin_ratio,
                        composite_weights=composite_weights,
                        well_filter=well_filter,
                        tile_overlap=tile_overlap,
                        tile_overlap_x=tile_overlap_x,
                        tile_overlap_y=tile_overlap_y,
                        max_shift=max_shift
                    )
                    return success
            else:
                stitch_source = plate_folder
                logger.info(f"No Z-stack detected in {plate_folder}, using standard stitching")

            if stitch_source != plate_folder:
                dirs['input'] = Path(stitch_source) / "TimePoint_1"

            # 2. Find HTD file to get grid dimensions
            htd_file = self.find_HTD_file(plate_path)
            grid_size_x, grid_size_y = 2, 2  # Default grid size for tests
            if htd_file:
                parsed = self.parse_HTD_file(htd_file)
                if parsed:
                    grid_size_x, grid_size_y = parsed
                    logger.info(f"Parsed grid dimensions from HTD file: {grid_size_x}x{grid_size_y}")
                else:
                    logger.warning("Could not parse grid dimensions from HTD file, using default 2x2")
            else:
                logger.warning("No HTD file found, using default grid size 2x2")

            grid_dims = (grid_size_x, grid_size_y)
            logger.info(f"Using grid dimensions: {grid_size_x}x{grid_size_y}")

            # 3. Auto-detect patterns
            patterns_by_well = self.stitcher.auto_detect_patterns(dirs['input'], well_filter)
            if not patterns_by_well:
                logger.error(f"No image patterns detected in {dirs['input']}")
                return False

            logger.info(f"Detected {len(patterns_by_well)} wells with images")

            # 4. Process each well
            for well, wavelength_patterns in patterns_by_well.items():
                logger.info(f"\nProcessing well {well} with {len(wavelength_patterns)} wavelength(s)")

                # Check if we need to use multiple reference channels
                if len(reference_channels) > 1:
                    logger.info(f"Using multiple reference channels: {reference_channels}")
                    # Create composite reference channel
                    ref_channel, ref_pattern, ref_dir, updated_patterns = self.stitcher.prepare_reference_channel(
                        well, wavelength_patterns, dirs, reference_channels, preprocessing_funcs, composite_weights
                    )
                else:
                    # Use single reference channel
                    ref_channel, ref_pattern, ref_dir, updated_patterns = self.stitcher.prepare_reference_channel(
                        well, wavelength_patterns, dirs, reference_channels, preprocessing_funcs, composite_weights
                    )

                if ref_channel is None:
                    logger.error(f"Failed to prepare reference channel for well {well}")
                    continue

                if use_reference_positions:
                    stitched_name = self.stitcher.compute_stitched_name(ref_pattern)
                    positions_path = dirs['positions'] / f"{Path(stitched_name).stem}.csv"

                    if not positions_path.exists():
                        logger.error(f"Reference positions file not found: {positions_path}")
                        logger.error("Cannot stitch using reference positions")
                        return False

                    logger.info(f"Using existing reference positions from {positions_path}")

                    success = self.stitcher.process_well_wavelengths(
                        well, updated_patterns, dirs, grid_dims,
                        ref_channel, ref_pattern, ref_dir,
                        margin_ratio=margin_ratio,
                        tile_overlap=tile_overlap,
                        tile_overlap_x=tile_overlap_x,
                        tile_overlap_y=tile_overlap_y,
                        max_shift=max_shift,
                        use_existing_positions=True
                    )
                else:
                    success = self.stitcher.process_well_wavelengths(
                        well, updated_patterns, dirs, grid_dims,
                        ref_channel, ref_pattern, ref_dir,
                        margin_ratio=margin_ratio,
                        tile_overlap=tile_overlap,
                        tile_overlap_x=tile_overlap_x,
                        tile_overlap_y=tile_overlap_y,
                        max_shift=max_shift
                    )

                if success:
                    logger.info(f"Completed processing well {well}")
                else:
                    logger.error(f"Failed to process well {well}")

            return True

        except Exception as e:
            logger.error(f"Error in PlateProcessor.run: {e}", exc_info=True)
            return False

    def _process_regular_plate(self, plate_folder: Path) -> bool:
        """
        Process a regular (non-Z-stack) plate.

        Args:
            plate_folder (Path): Path to the plate folder

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Processing regular plate: {plate_folder}")

            # 1. Create output directories
            processed_dir = plate_folder.parent / f"{plate_folder.name}_processed"
            positions_dir = plate_folder.parent / f"{plate_folder.name}_positions"
            stitched_dir = plate_folder.parent / f"{plate_folder.name}_stitched"

            ensure_directory(processed_dir)
            ensure_directory(positions_dir)
            ensure_directory(stitched_dir)

            # 2. Process each well
            timepoint_dir = plate_folder / "TimePoint_1"
            if not timepoint_dir.exists():
                logger.error(f"TimePoint_1 directory not found in {plate_folder}")
                return False

            # 3. Find all wells and filter if needed
            wells = self._find_wells(timepoint_dir)
            if self.config.well_filter:
                wells = [w for w in wells if w in self.config.well_filter]

            if not wells:
                logger.error(f"No wells found in {timepoint_dir}")
                return False

            logger.info(f"Processing wells: {wells}")

            # 4. Process each well
            for well in wells:
                success = self._process_well(timepoint_dir, well, processed_dir, positions_dir, stitched_dir)
                if not success:
                    logger.error(f"Failed to process well: {well}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error in _process_regular_plate: {e}", exc_info=True)
            return False

    def _process_zstack_plate(self, plate_folder: Path) -> bool:
        """
        Process a Z-stack plate.

        Args:
            plate_folder (Path): Path to the plate folder

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Processing Z-stack plate: {plate_folder}")

            # Get Z-stack configuration
            z_config = self.config.z_stack_processor

            # 1. Create projections if needed
            if z_config.create_projections:
                logger.info(f"Creating projections for Z-stack plate: {plate_folder}")
                # TODO: Implement projection creation

            # 2. Select reference Z-plane for stitching
            reference_z = z_config.stitch_z_reference
            logger.info(f"Using reference Z-plane: {reference_z}")

            # 3. Process the reference Z-plane
            if reference_z == "best_focus" and z_config.focus_detect:
                # TODO: Implement best focus selection
                pass
            elif reference_z in ["max", "mean", "std"]:
                # Use projection as reference
                projection_dir = plate_folder.parent / f"{plate_folder.name}_Projections"
                if not projection_dir.exists():
                    logger.error(f"Projection directory not found: {projection_dir}")
                    return False

                # Process the projection directory as a regular plate
                return self._process_regular_plate(projection_dir)
            else:
                # Use specific Z-plane as reference
                # TODO: Implement specific Z-plane processing
                pass

            # 4. Stitch all Z-planes if needed
            if z_config.stitch_all_z_planes:
                logger.info(f"Stitching all Z-planes for plate: {plate_folder}")
                # TODO: Implement stitching all Z-planes

            return True

        except Exception as e:
            logger.error(f"Error in _process_zstack_plate: {e}", exc_info=True)
            return False

    def _find_wells(self, timepoint_dir: Path) -> List[str]:
        """
        Find all wells in the timepoint directory.

        Args:
            timepoint_dir (Path): Path to the TimePoint_1 directory

        Returns:
            list: List of well names (e.g., ['A01', 'A02', ...])
        """
        well_pattern = re.compile(r'([A-Z]\d{2})_')
        wells = set()

        for file in timepoint_dir.glob("*.tif"):
            match = well_pattern.search(file.name)
            if match:
                wells.add(match.group(1))

        return sorted(list(wells))

    def _process_well(self, timepoint_dir: Path, well: str,
                      processed_dir: Path, positions_dir: Path,
                      stitched_dir: Path) -> bool:
        """
        Process a single well.

        Args:
            timepoint_dir (Path): Path to the TimePoint_1 directory
            well (str): Well name (e.g., 'A01')
            processed_dir (Path): Path to the processed directory
            positions_dir (Path): Path to the positions directory
            stitched_dir (Path): Path to the stitched directory

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Processing well: {well}")

            # 1. Create output directories
            processed_timepoint_dir = processed_dir / "TimePoint_1"
            positions_timepoint_dir = positions_dir / "TimePoint_1"
            stitched_timepoint_dir = stitched_dir / "TimePoint_1"

            ensure_directory(processed_timepoint_dir)
            ensure_directory(positions_timepoint_dir)
            ensure_directory(stitched_timepoint_dir)

            # 2. Process reference channels
            for channel in self.config.reference_channels:
                # TODO: Implement reference channel processing
                pass

            # 3. Generate positions
            # TODO: Implement position generation

            # 4. Stitch images
            # TODO: Implement stitching

            return True

        except Exception as e:
            logger.error(f"Error in _process_well: {e}", exc_info=True)
            return False