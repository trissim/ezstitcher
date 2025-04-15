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
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.filename_parser import FilenameParser, detect_parser, create_parser
from ezstitcher.core.image_locator import ImageLocator


logger = logging.getLogger(__name__)

class PlateProcessor:
    """
    High-level orchestrator for processing a microscopy plate.
    Modular pipeline: flatten/rename, preprocessing, position generation, stitching, z-stack handling.
    All orchestration is driven by a unified configuration model (PlateProcessorConfig).
    """

    def __init__(self, config: PlateProcessorConfig):
        """
        Initialize PlateProcessor with a unified configuration model.

        Args:
            config (PlateProcessorConfig): Unified configuration object (Pydantic or dataclass).
        """
        self.config = config
        self.fs_manager = FileSystemManager()
        self.zstack_processor = ZStackProcessor(config.z_stack_processor)
        self.focus_analyzer = FocusAnalyzer(config.focus_analyzer)
        self.image_preprocessor = ImagePreprocessor(config.image_preprocessor)
        self._current_plate_folder = None
        self.filename_parser = None

    def run(self, plate_folder):
        """
        Entry point: orchestrate the full plate processing pipeline.

        Args:
            plate_folder (str or Path): Path to the plate folder.

        Returns:
            bool: True if successful, False otherwise.
        """
        self._current_plate_folder = plate_folder
        plate_path = Path(plate_folder)
        config = self.config
        self.filename_parser = create_parser(config.microscope_type, plate_folder=plate_path)
        config.stitcher.filename_parser = self.filename_parser
        #self.stitcher.pattern_matcher = PatternMatcher(self.filename_parser)    
        self.stitcher = Stitcher(config.stitcher,filename_parser=self.filename_parser)
        try:
            # Phase 1: Flatten and rename files (non-destructive, outputs to new dir if needed)

            images_dir = ImageLocator.find_image_directory(plate_path)
            self._flatten_and_rename(images_dir)
            dirs = self.fs_manager.create_output_directories(
                plate_path,
                {
                    'processed': config.output_dir_suffix,
                    'positions': config.positions_dir_suffix,
                    'stitched': config.stitched_dir_suffix
                }
            )
            dirs['input'] = ImageLocator.find_image_directory(plate_path)

            # Phase 2: Preprocessing (if any)
            # (Handled within stitching phase as needed, see below)

            # Phase 3: Z-stack handling (detection, best focus, projections)
            zstack_result = self._handle_zstack(plate_path)
#            if zstack_result is False:
#                return False
#            input_dir = zstack_result

            # Phase 4: Position generation and stitching
            return self._stitch_plate(plate_path)

        except Exception as e:
            logger.error(f"Error in PlateProcessor.run: {e}", exc_info=True)
            return False

    def _flatten_and_rename(self, plate_path: Path):
        """
        Flatten Z-stack folders and rename files for consistent site number padding if configured.

        Args:
            plate_path (Path): Path to the plate folder.
        """
        config = self.config
        if config.rename_files:
            # Use FileSystemManager's rename utility (non-destructive if implemented as such)
            self.fs_manager.rename_files_with_consistent_padding(
                plate_path,
                parser=self.filename_parser,
                width=config.padding_width
            )

    def _handle_zstack(self, plate_path: Path):
        """
        Detect and process Z-stacks, create best focus/projection dirs as needed.

        Args:
            plate_path (Path): Path to the plate folder.

        Returns:
            tuple: input_dir for downstream processing, or False on error.
        """
        config = self.config
        parent_dir = plate_path.parent
        plate_name = plate_path.name
        z_config = config.z_stack_processor

        # Create output directories
        dirs = self.fs_manager.create_output_directories(
            plate_path,
            {
                'processed': config.output_dir_suffix,
                'positions': config.positions_dir_suffix,
                'stitched': config.stitched_dir_suffix
            }
        )
        dirs['input'] = ImageLocator.find_image_directory(plate_path)
        if not dirs['input'].exists():
            logger.error(f"Input directory does not exist: {dirs['input']}")
            return False

        has_zstack = self.zstack_processor.detect_z_stacks(str(plate_path))
        best_focus_dir = None
        projections_dir = None

        if has_zstack:
            # Best focus images
            if getattr(z_config, "focus_detect", False):
                best_focus_dir = parent_dir / f"{plate_name}{config.best_focus_dir_suffix}"
                self.fs_manager.ensure_directory(best_focus_dir)
                logger.info(f"Finding best focused images using method: {z_config.focus_method}")
                self.zstack_processor.focus_manager.create_best_focus_images(dirs['input'],
                                                                             best_focus_dir,
                                                                             z_config.focus_method,
                                                                             config.reference_channels[0],
                                                                             )

                success, _ = self.zstack_processor.create_best_focus_images(
                    dirs['input'],
                    best_focus_dir,
                    focus_method=z_config.focus_method,
                    focus_wavelength=config.reference_channels[0]
                )
                if not success:
                    logger.warning("No best focus images created")

            # Projections
            if getattr(z_config, "create_projections", False):
                projections_dir = parent_dir / f"{plate_name}{config.projections_dir_suffix}"
                logger.info(f"Creating projection: {getattr(z_config, 'stitch_z_reference', 'max')}")
                success, _ = self.zstack_processor.create_zstack_projections(
                    dirs['input'],
                    projections_dir,
                    projection_types=[getattr(z_config, "stitch_z_reference", "max")]
                )
                if not success:
                    logger.warning("No projections created")

    def _stitch_plate(self, input_dir: Path):
        """
        Detect patterns, generate positions, and perform stitching for all wells.

        Args:
            input_dir (Path): Directory to plate folder.

        Returns:
            bool: True if successful, False otherwise.
        """
        config = self.config
        plate_path = Path(self._current_plate_folder)
        parent_dir = plate_path.parent
        plate_name =plate_path.name

        # Output directories
        processed_dir = parent_dir / f"{plate_name}{config.output_dir_suffix}"
        positions_dir = parent_dir / f"{plate_name}{config.positions_dir_suffix}"
        stitched_dir = parent_dir / f"{plate_name}{config.stitched_dir_suffix}"

        # Find HTD file for grid size
        htd_file = self.fs_manager.find_htd_file(plate_path)
        grid_size_x, grid_size_y = 2, 2
        if htd_file:
            parsed = self.fs_manager.parse_htd_file(htd_file)
            if parsed:
                grid_size_x, grid_size_y = parsed
                logger.info(f"Parsed grid dimensions from HTD file: {grid_size_x}x{grid_size_y}")
            else:
                logger.warning("Could not parse grid dimensions from HTD file, using default 2x2")
        else:
            logger.warning("No HTD file found, using default grid size 2x2")
        grid_dims = (grid_size_x, grid_size_y)
        logger.info(f"Using grid dimensions: {grid_size_x}x{grid_size_y}")

        # Detect patterns
        images_dir = ImageLocator.find_image_directory(plate_path)
        patterns_by_well = self.stitcher.pattern_matcher.auto_detect_patterns(images_dir, config.well_filter)
        if not patterns_by_well:
            logger.error(f"No image patterns detected in {input_dir}")
            return False
        logger.info(f"Detected {len(patterns_by_well)} wells with images")

        # Process each well
        for well, wavelength_patterns in patterns_by_well.items():
            logger.info(f"Processing well {well} with {len(wavelength_patterns)} wavelength(s)")
            ref_channel, ref_pattern, ref_dir, updated_patterns = self.stitcher.prepare_reference_channel(
                well, wavelength_patterns,
                {
                    'input': input_dir,
                    'images': images_dir,
                    'processed': processed_dir,
                    'positions': positions_dir,
                    'stitched': stitched_dir
                },
                config.reference_channels,
                config.preprocessing_funcs,
                config.composite_weights
            )

            if config.use_reference_positions:
                stitched_name = self.stitcher.compute_stitched_name(ref_pattern)
                positions_path = positions_dir / f"{Path(stitched_name).stem}.csv"
                if not positions_path.exists():
                    logger.error(f"Reference positions file not found: {positions_path}")
                    logger.error("Cannot stitch using reference positions")
                    return False
                logger.info(f"Using existing reference positions from {positions_path}")
                success = self.stitcher.stitch_well_wavelengths(
                    well, updated_patterns,
                    {
                        'input': input_dir,
                        'processed': processed_dir,
                        'positions': positions_dir,
                        'stitched': stitched_dir
                    },
                    grid_dims,
                    ref_channel, ref_pattern,
                    use_existing_positions=True
                )
            else:
                images_dir = ImageLocator.find_image_directory(plate_path)
                success = self.stitcher.stitch_well_wavelengths(
                    well, updated_patterns,
                    {
                        'input': input_dir,
                        'images': images_dir, 
                        'processed': processed_dir,
                        'positions': positions_dir,
                        'stitched': stitched_dir
                    },
                    grid_dims,
                    ref_channel, ref_pattern
                )
            if success:
                logger.info(f"Completed processing well {well}")
            else:
                logger.error(f"Failed to process well {well}")

        # Clean up temporary folders if needed
        #self.fs_manager.clean_temp_folders(parent_dir, plate_name, keep_suffixes=['_stitched'])
        return True

    def run(self, plate_folder):
        """
        Run the pipeline on the specified plate folder.
        
        Args:
            plate_folder: Path to the plate folder
            
        Returns:
            True if successful, False otherwise
        """
        plate_path = Path(plate_folder)
        config = self.config
        
        # Create output directories
        dirs = self.fs_manager.create_output_directories(
            plate_path,
            {
                'processed': config.processed_dir_suffix,
                'post_processed': config.post_processed_dir_suffix,
                'positions': config.positions_dir_suffix,
                'stitched': config.stitched_dir_suffix
            }
        )
        
        # Find image directory
        dirs['input'] = ImageLocator.find_image_directory(plate_path)
        
        # Phase 1: Reference Processing
        logger.info("Phase 1: Reference Processing")
        reference_patterns = self.process_images(
            dirs['input'],
            dirs['processed'],
            config.reference_processing,
            well_filter=config.well_filter
        )
        
        # Phase 2: Generate Stitching Positions
        logger.info("Phase 2: Generate Stitching Positions")
        positions = self._generate_positions(
            dirs['processed'],
            dirs['positions'],
            reference_patterns,
            config.stitcher
        )
        
        # Phase 3: Final Processing
        logger.info("Phase 3: Final Processing")
        final_patterns = self.process_images(
            dirs['input'],
            dirs['post_processed'],
            config.final_processing,
            well_filter=config.well_filter
        )
        
        # Phase 4: Stitch Final Images
        logger.info("Phase 4: Stitch Final Images")
        success = self._stitch_images(
            dirs['post_processed'],
            dirs['stitched'],
            dirs['positions'],
            final_patterns,
            config.stitcher
        )
        
        return success