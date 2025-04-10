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
from ezstitcher.core.filename_parser import FilenameParser, detect_parser

logger = logging.getLogger(__name__)

class PlateProcessor:
    """
    High-level orchestrator for processing a microscopy plate.
    Coordinates Z-stack handling, stitching, and output management.
    """
    def __init__(self, config: PlateProcessorConfig):
        self.config = config
        self.filename_parser = None  # Will be initialized in run() based on microscope_type
        self.fs_manager = FileSystemManager()
        self.zstack_processor = ZStackProcessor(config.z_stack_processor)  # Will be updated in run()
        self.focus_analyzer = FocusAnalyzer(config.focus_analyzer)
        self.image_preprocessor = ImagePreprocessor(config.image_preprocessor)
        self.stitcher = Stitcher(config.stitcher)  # Will be updated in run()

    # Methods removed as they're now in FileSystemManager

    def run(self, plate_folder):
        """
        Process a plate folder with microscopy images.

        Args:
            plate_folder (str or Path): Path to the plate folder

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            plate_path = Path(plate_folder)
            parent_dir = plate_path.parent
            plate_name = plate_path.name

            # Get configuration parameters
            config = self.config
            reference_channels = config.reference_channels
            well_filter = config.well_filter

            # Initialize the filename parser based on microscope_type
            if config.microscope_type.lower() == 'auto':
                # Auto-detect the microscope type from the filenames
                # Get a sample of filenames from the plate folder
                sample_files = self.fs_manager.list_image_files(plate_path, extensions=['.tif', '.tiff', '.TIF', '.TIFF'])[:10]
                sample_files = [Path(f).name for f in sample_files]

                if not sample_files:
                    logger.warning(f"No image files found in {plate_path}. Using default ImageXpress parser.")
                    from ezstitcher.core.filename_parser import ImageXpressFilenameParser
                    self.filename_parser = ImageXpressFilenameParser()
                else:
                    # Auto-detect the parser based on the filenames
                    self.filename_parser = detect_parser(sample_files)
                    logger.info(f"Auto-detected microscope type: {self.filename_parser.__class__.__name__}")

                    # If Opera Phenix is detected, rename files to ImageXpress format
                    if self.filename_parser.__class__.__name__ == 'OperaPhenixFilenameParser':
                        logger.info(f"Converting Opera Phenix files to ImageXpress format...")

                        # Check if this is a directory with an Images subdirectory (Opera Phenix format)
                        images_dir = plate_path / "Images"
                        if images_dir.exists() and images_dir.is_dir():
                            logger.info(f"Found Images directory, using it for Opera Phenix files")
                            # Rename files in the Images directory
                            success = self.filename_parser.rename_all_files_in_directory(images_dir)
                        else:
                            # Rename files in place
                            success = self.filename_parser.rename_all_files_in_directory(plate_path)

                        if success:
                            logger.info(f"Successfully converted Opera Phenix files to ImageXpress format")
                            # Use ImageXpress parser for the converted files
                            from ezstitcher.core.filename_parser import ImageXpressFilenameParser
                            self.filename_parser = ImageXpressFilenameParser()
                        else:
                            logger.warning(f"Failed to convert Opera Phenix files to ImageXpress format. Using original files.")
            else:
                # Use the specified microscope type
                from ezstitcher.core.filename_parser import create_parser
                self.filename_parser = create_parser(config.microscope_type)
                logger.info(f"Using specified microscope type: {config.microscope_type}")

                # If Opera Phenix is specified, rename files to ImageXpress format
                if config.microscope_type.lower() == 'operaphenix':
                    logger.info(f"Converting Opera Phenix files to ImageXpress format...")

                    # Check if this is a directory with an Images subdirectory (Opera Phenix format)
                    images_dir = plate_path / "Images"
                    if images_dir.exists() and images_dir.is_dir():
                        logger.info(f"Found Images directory, using it for Opera Phenix files")
                        # Rename files in the Images directory
                        success = self.filename_parser.rename_all_files_in_directory(images_dir)
                    else:
                        # Rename files in place
                        success = self.filename_parser.rename_all_files_in_directory(plate_path)

                    if success:
                        logger.info(f"Successfully converted Opera Phenix files to ImageXpress format")
                        # Use ImageXpress parser for the converted files
                        from ezstitcher.core.filename_parser import ImageXpressFilenameParser
                        self.filename_parser = ImageXpressFilenameParser()
                    else:
                        logger.warning(f"Failed to convert Opera Phenix files to ImageXpress format. Using original files.")

            # Update components with the filename parser
            self.fs_manager = FileSystemManager(filename_parser=self.filename_parser)
            self.stitcher = Stitcher(self.config.stitcher, self.filename_parser)
            self.zstack_processor = ZStackProcessor(self.config.z_stack_processor, self.filename_parser)
            use_reference_positions = config.use_reference_positions
            preprocessing_funcs = config.preprocessing_funcs
            composite_weights = config.composite_weights

            # Get stitcher configuration
            stitcher_config = config.stitcher
            margin_ratio = stitcher_config.margin_ratio
            tile_overlap = stitcher_config.tile_overlap
            tile_overlap_x = stitcher_config.tile_overlap_x
            tile_overlap_y = stitcher_config.tile_overlap_y
            max_shift = stitcher_config.max_shift

            # Get Z-stack configuration
            z_config = config.z_stack_processor
            focus_detect = z_config.focus_detect
            focus_method = z_config.focus_method
            create_projections = z_config.create_projections
            stitch_z_reference = z_config.stitch_z_reference
            save_projections = z_config.save_projections
            stitch_all_z_planes = z_config.stitch_all_z_planes

            # Create directory paths using config parameters
            processed_dir = parent_dir / f"{plate_name}{config.output_dir_suffix}"
            positions_dir = parent_dir / f"{plate_name}{config.positions_dir_suffix}"
            stitched_dir = parent_dir / f"{plate_name}{config.stitched_dir_suffix}"
            timepoint_dir = config.timepoint_dir_name

            # Ensure directories exist using FileSystemManager
            processed_dir = self.fs_manager.ensure_directory(processed_dir)
            positions_dir = self.fs_manager.ensure_directory(positions_dir)
            stitched_dir = self.fs_manager.ensure_directory(stitched_dir)

            # Create TimePoint_1 directories
            processed_timepoint_dir = self.fs_manager.ensure_directory(processed_dir / timepoint_dir)
            stitched_timepoint_dir = self.fs_manager.ensure_directory(stitched_dir / timepoint_dir)

            logger.info(f"Created processed directory: {processed_dir}")
            logger.info(f"Created positions directory: {positions_dir}")
            logger.info(f"Created stitched directory: {stitched_dir}")

            # Initialize directory structure manager
            dir_structure = self.fs_manager.initialize_dir_structure(plate_path)
            logger.info(f"Detected directory structure: {dir_structure.structure_type}")

            # Get input directory based on the detected structure
            timepoint_dir_path = dir_structure.get_timepoint_dir()

            if timepoint_dir_path:
                logger.info(f"Using timepoint directory: {timepoint_dir_path}")
                input_dir = timepoint_dir_path
            elif "images" in dir_structure.image_locations:
                # Images are in the Images directory
                images_dir = plate_path / "Images"
                logger.info(f"Using Images directory: {images_dir}")
                input_dir = images_dir
            elif "plate" in dir_structure.image_locations:
                # Images are directly in the plate folder
                logger.info(f"Using plate directory directly: {plate_path}")
                input_dir = plate_path
            else:
                # No images found
                raise FileNotFoundError(f"No image files found in {plate_path} or its subdirectories")

            dirs = {
                'input': input_dir,
                'processed': processed_timepoint_dir,
                'positions': positions_dir,
                'stitched': stitched_timepoint_dir
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
                    best_focus_dir = parent_dir / f"{plate_name}{config.best_focus_dir_suffix}"
                    self.fs_manager.ensure_directory(best_focus_dir)

                    logger.info(f"Finding best focused images using method: {focus_method}")
                    success, _ = self.zstack_processor.create_best_focus_images(
                        dirs['input'],
                        best_focus_dir,
                        focus_method=focus_method,
                        focus_wavelength=reference_channels[0]
                    )

                    # Ensure TimePoint_1 directory exists in best_focus_dir
                    best_focus_timepoint_dir = best_focus_dir / timepoint_dir
                    if not best_focus_timepoint_dir.exists():
                        logger.error(f"{timepoint_dir} directory not created in {best_focus_dir}")
                        return False
                    if not success:
                        logger.warning("No best focus images created")

                if create_projections:
                    projections_dir = parent_dir / f"{plate_name}{config.projections_dir_suffix}"
                    projections_timepoint_dir = self.fs_manager.ensure_directory(projections_dir / timepoint_dir)

                    logger.info(f"Creating projection: {stitch_z_reference}")
                    success, _ = self.zstack_processor.create_zstack_projections(
                        dirs['input'],
                        projections_timepoint_dir,
                        projection_types=[stitch_z_reference]
                    )
                    if not success:
                        logger.warning("No projections created")

                stitch_source = plate_folder
                if stitch_z_reference == 'best_focus' and best_focus_dir:
                    stitch_source = best_focus_dir
                    logger.info(f"Using best focus images for stitching from {best_focus_dir}")
                elif stitch_z_reference in ['max', 'mean'] and projections_dir:
                    stitch_source = projections_dir
                    logger.info(f"Using {stitch_z_reference} projections for stitching from {projections_dir}")
                elif (isinstance(stitch_z_reference, str) and stitch_z_reference in ['max', 'mean', 'best_focus']) or callable(stitch_z_reference):
                    if callable(stitch_z_reference):
                        logger.info(f"Stitching all Z-planes using custom function as reference")
                    else:
                        logger.info(f"Stitching all Z-planes using {stitch_z_reference} as reference")
                    # Create a new config for Z-plane stitching
                    z_plane_config = PlateProcessorConfig(
                        reference_channels=reference_channels,
                        well_filter=well_filter,
                        preprocessing_funcs=preprocessing_funcs,
                        composite_weights=composite_weights,
                        use_reference_positions=use_reference_positions,
                        stitcher=stitcher_config,
                        z_stack_processor=z_config
                    )

                    # Create a new processor with this config
                    z_processor = PlateProcessor(z_plane_config)

                    success = self.zstack_processor.stitch_across_z(
                        plate_folder,
                        reference_z=stitch_z_reference,
                        stitch_all_z_planes=True,
                        processor=z_processor
                    )
                    return success
            else:
                stitch_source = plate_folder
                logger.info(f"No Z-stack detected in {plate_folder}, using standard stitching")

            if stitch_source != plate_folder:
                dirs['input'] = Path(stitch_source) / timepoint_dir

            # 2. Find HTD file to get grid dimensions
            htd_file = self.fs_manager.find_htd_file(plate_path)
            grid_size_x, grid_size_y = 2, 2  # Default grid size for tests
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

                    # Use the stitcher configuration directly
                    success = self.stitcher.process_well_wavelengths(
                        well, updated_patterns, dirs, grid_dims,
                        ref_channel, ref_pattern, ref_dir,
                        use_existing_positions=True
                    )
                else:
                    # Use the stitcher configuration directly
                    success = self.stitcher.process_well_wavelengths(
                        well, updated_patterns, dirs, grid_dims,
                        ref_channel, ref_pattern, ref_dir
                    )

                if success:
                    logger.info(f"Completed processing well {well}")
                else:
                    logger.error(f"Failed to process well {well}")

            # Clean up temporary folders if needed
            self.fs_manager.clean_temp_folders(parent_dir, plate_name, keep_suffixes=['_stitched', '_positions'])

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

            # 1. Create output directories using config parameters
            config = self.config
            parent_dir = plate_folder.parent
            plate_name = plate_folder.name

            processed_dir = parent_dir / f"{plate_name}{config.output_dir_suffix}"
            positions_dir = parent_dir / f"{plate_name}{config.positions_dir_suffix}"
            stitched_dir = parent_dir / f"{plate_name}{config.stitched_dir_suffix}"
            timepoint_dir = config.timepoint_dir_name

            self.fs_manager.ensure_directory(processed_dir)
            self.fs_manager.ensure_directory(positions_dir)
            self.fs_manager.ensure_directory(stitched_dir)

            # 2. Initialize directory structure manager
            dir_structure = self.fs_manager.initialize_dir_structure(plate_folder)
            logger.info(f"Detected directory structure: {dir_structure.structure_type}")

            # Get input directory based on the detected structure
            timepoint_path = dir_structure.get_timepoint_dir()

            if timepoint_path:
                logger.info(f"Using timepoint directory: {timepoint_path}")
            elif "images" in dir_structure.image_locations:
                # Images are in the Images directory
                images_dir = plate_folder / "Images"
                logger.info(f"Using Images directory: {images_dir}")
                timepoint_path = images_dir
            elif "plate" in dir_structure.image_locations:
                # Images are directly in the plate folder
                logger.info(f"Using plate directory directly: {plate_folder}")
                timepoint_path = plate_folder
            else:
                # No images found
                logger.error(f"No image files found in {plate_folder} or its subdirectories")
                return False

            # 3. Find all wells and filter if needed
            wells = dir_structure.get_wells()
            if self.config.well_filter:
                wells = [w for w in wells if w in self.config.well_filter]

            if not wells:
                logger.error(f"No wells found in {timepoint_path}")
                return False

            logger.info(f"Processing wells: {wells}")

            # 4. Process each well
            for well in wells:
                success = self._process_well(timepoint_path, well, processed_dir, positions_dir, stitched_dir)
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

            # Get configuration parameters
            config = self.config
            z_config = config.z_stack_processor
            parent_dir = plate_folder.parent
            plate_name = plate_folder.name
            timepoint_dir = config.timepoint_dir_name

            # 1. Create projections if needed
            if z_config.create_projections:
                logger.info(f"Creating projections for Z-stack plate: {plate_folder}")
                projections_dir = parent_dir / f"{plate_name}{config.projections_dir_suffix}"
                projections_timepoint_dir = self.fs_manager.ensure_directory(projections_dir / timepoint_dir)

                # Initialize directory structure manager
                dir_structure = self.fs_manager.initialize_dir_structure(plate_folder)
                logger.info(f"Detected directory structure: {dir_structure.structure_type}")

                # Get input directory based on the detected structure
                timepoint_path = dir_structure.get_timepoint_dir()

                if timepoint_path:
                    logger.info(f"Using timepoint directory: {timepoint_path}")
                elif "images" in dir_structure.image_locations:
                    # Images are in the Images directory
                    images_dir = plate_folder / "Images"
                    logger.info(f"Using Images directory: {images_dir}")
                    timepoint_path = images_dir
                elif "plate" in dir_structure.image_locations:
                    # Images are directly in the plate folder
                    logger.info(f"Using plate directory directly: {plate_folder}")
                    timepoint_path = plate_folder
                else:
                    # No images found
                    logger.error(f"No image files found in {plate_folder} or its subdirectories")
                    return False

                # Create projections using ZStackProcessor
                success, _ = self.zstack_processor.create_zstack_projections(
                    timepoint_path,
                    projections_timepoint_dir,
                    projection_types=z_config.projection_types
                )

                if not success:
                    logger.warning("Failed to create projections")

            # 2. Select reference method for stitching
            # Convert from new reference_method format to old reference_z format for backward compatibility
            if isinstance(z_config.reference_method, str):
                if z_config.reference_method == 'max_projection':
                    reference_z = 'max'
                elif z_config.reference_method == 'mean_projection':
                    reference_z = 'mean'
                elif z_config.reference_method == 'best_focus':
                    reference_z = 'best_focus'
                else:
                    logger.error(f"Invalid reference_method: {z_config.reference_method}")
                    return False
            else:
                # If it's a callable, use it directly
                reference_z = z_config.reference_method

            logger.info(f"Using reference method: {z_config.reference_method} (reference_z: {reference_z})")

            # 3. Process the reference Z-plane
            if reference_z == "best_focus":
                # Create best focus directory
                best_focus_dir = parent_dir / f"{plate_name}{config.best_focus_dir_suffix}"
                self.fs_manager.ensure_directory(best_focus_dir)

                # Check if TimePoint_1 directory exists
                timepoint_path = plate_folder / timepoint_dir
                if not timepoint_path.exists():
                    # Check if there are image files directly in the plate folder
                    image_files = []
                    for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                        image_files.extend(list(plate_folder.glob(f"*{ext}")))

                    if image_files:
                        logger.info(f"No {timepoint_dir} directory found, but found {len(image_files)} image files directly in the plate folder.")
                        timepoint_path = plate_folder
                    else:
                        logger.error(f"{timepoint_dir} directory not found in {plate_folder} and no image files found directly in the plate folder")
                        return False

                # Create best focus images
                success, _ = self.zstack_processor.create_best_focus_images(
                    timepoint_path,
                    best_focus_dir,
                    focus_method=z_config.focus_method,
                    focus_wavelength=config.reference_channels[0]
                )

                if success:
                    # Process the best focus directory as a regular plate
                    return self._process_regular_plate(best_focus_dir)
                else:
                    logger.error("Failed to create best focus images")
                    return False

            elif reference_z in ["max", "mean", "std"]:
                # Use projection as reference
                projection_dir = parent_dir / f"{plate_name}{config.projections_dir_suffix}"
                if not projection_dir.exists():
                    logger.error(f"Projection directory not found: {projection_dir}")
                    return False

                # Process the projection directory as a regular plate
                return self._process_regular_plate(projection_dir)
            else:
                # Use specific Z-plane as reference
                logger.info(f"Using specific Z-plane {reference_z} as reference")
                # TODO: Implement specific Z-plane processing
                pass

            # 4. Stitch all Z-planes if needed
            if z_config.stitch_all_z_planes:
                logger.info(f"Stitching all Z-planes for plate: {plate_folder}")
                # Create a new processor with the same config
                z_processor = PlateProcessor(self.config)

                # Stitch across Z using ZStackProcessor
                # Pass None for reference_z to use the reference_method from the config
                success = self.zstack_processor.stitch_across_z(
                    plate_folder,
                    reference_z=None,  # Use reference_method from config
                    stitch_all_z_planes=True,
                    processor=z_processor
                )

                return success

            return True

        except Exception as e:
            logger.error(f"Error in _process_zstack_plate: {e}", exc_info=True)
            return False

    # Method removed as it's now in FileSystemManager

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
            config = self.config
            timepoint_dir_name = config.timepoint_dir_name

            # 1. Create output directories
            processed_timepoint_dir = self.fs_manager.ensure_directory(processed_dir / timepoint_dir_name)
            positions_timepoint_dir = self.fs_manager.ensure_directory(positions_dir / timepoint_dir_name)
            stitched_timepoint_dir = self.fs_manager.ensure_directory(stitched_dir / timepoint_dir_name)

            # 2. Auto-detect patterns for this well
            patterns = self.stitcher.auto_detect_patterns(timepoint_dir, [well])
            if not patterns or well not in patterns:
                logger.error(f"No image patterns detected for well {well}")
                return False

            wavelength_patterns = patterns[well]
            logger.info(f"Detected {len(wavelength_patterns)} wavelengths for well {well}")

            # 3. Prepare directories for stitching
            dirs = {
                'input': timepoint_dir,
                'processed': processed_timepoint_dir,
                'positions': positions_dir,
                'stitched': stitched_timepoint_dir
            }

            # 4. Find HTD file to get grid dimensions
            htd_file = self.fs_manager.find_htd_file(timepoint_dir.parent)
            grid_size_x, grid_size_y = 2, 2  # Default grid size
            if htd_file:
                parsed = self.fs_manager.parse_htd_file(htd_file)
                if parsed:
                    grid_size_x, grid_size_y = parsed

            grid_dims = (grid_size_x, grid_size_y)

            # 5. Prepare reference channel
            reference_channels = config.reference_channels
            preprocessing_funcs = config.preprocessing_funcs
            composite_weights = config.composite_weights

            ref_channel, ref_pattern, ref_dir, updated_patterns = self.stitcher.prepare_reference_channel(
                well, wavelength_patterns, dirs, reference_channels, preprocessing_funcs, composite_weights
            )

            if ref_channel is None:
                logger.error(f"Failed to prepare reference channel for well {well}")
                return False

            # 6. Process wavelengths
            success = self.stitcher.process_well_wavelengths(
                well, updated_patterns, dirs, grid_dims,
                ref_channel, ref_pattern, ref_dir
            )

            if success:
                logger.info(f"Successfully processed well {well}")
            else:
                logger.error(f"Failed to process well {well}")

            return success

        except Exception as e:
            logger.error(f"Error in _process_well: {e}", exc_info=True)
            return False