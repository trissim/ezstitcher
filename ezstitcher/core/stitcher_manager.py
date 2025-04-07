"""
Stitcher management module for ezstitcher.

This module contains the StitcherManager class for handling image stitching operations.
"""

import os
import re
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile

from ashlar import fileseries, reg

from ezstitcher.core.utils import (
    ensure_directory, load_image, save_image, path_list_from_pattern
)
from ezstitcher.core.image_processor import ImageProcessor
from ezstitcher.core.z_stack_manager import ZStackManager


def generate_positions_df(image_dir, image_pattern, positions, grid_size_x, grid_size_y):
    """
    Given an image_dir, an image_pattern (with '{iii}' or similar placeholder)
    and a list of (x, y) tuples 'positions', build a DataFrame with lines like:

      file: <filename>; position: (x, y); grid: (col, row);
    """
    all_files = path_list_from_pattern(image_dir, image_pattern)
    if len(all_files) != len(positions):
        raise ValueError(
            f"Number of matched files ({len(all_files)}) != number of positions ({len(positions)})"
        )

    # Generate a list of (x, y) grid positions following a raster pattern
    positions_grid = [(x, y) for y in range(grid_size_y) for x in range(grid_size_x)]
    data_rows = []

    for i, fname in enumerate(all_files):
        x, y = positions[i]
        row, col = positions_grid[i]

        data_rows.append({
            "file": "file: " + fname,
            "grid": " grid: " + "("+str(row)+", "+str(col)+")",
            "position": " position: " + "("+str(x)+", "+str(y)+")",
        })

    df = pd.DataFrame(data_rows)
    return df

logger = logging.getLogger(__name__)

class StitcherManager:
    """
    Class for handling image stitching operations.
    """

    @staticmethod
    def find_HTD_file(plate_folder):
        """
        Find the HTD file in a plate folder.

        Args:
            plate_folder (str or Path): Path to the plate folder

        Returns:
            Path: Path to the HTD file or None if not found
        """
        plate_path = Path(plate_folder)

        # Look for HTD files in the plate folder
        htd_files = list(plate_path.glob("*.HTD"))
        if htd_files:
            # If multiple HTD files, prefer the one with 'plate' in the name
            for htd_file in htd_files:
                if 'plate' in htd_file.name.lower():
                    return htd_file
            # Otherwise return the first one
            return htd_files[0]

        # Look in TimePoint_1 folder
        timepoint_dir = plate_path / "TimePoint_1"
        if timepoint_dir.exists():
            htd_files = list(timepoint_dir.glob("*.HTD"))
            if htd_files:
                for htd_file in htd_files:
                    if 'plate' in htd_file.name.lower():
                        return htd_file
                return htd_files[0]

        # Look in MetaData folder
        metadata_dir = plate_path / "MetaData"
        if metadata_dir.exists():
            htd_files = list(metadata_dir.glob("*.HTD"))
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

    @staticmethod
    def parse_HTD_file(htd_path):
        """
        Parse an HTD file to extract grid dimensions.

        Args:
            htd_path (str or Path): Path to the HTD file

        Returns:
            tuple: (grid_size_x, grid_size_y) or (None, None) if parsing fails
        """
        htd_path = Path(htd_path)

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
                logger.info(f"Parsed HTD file: grid size {grid_size_x}x{grid_size_y}")
                return grid_size_x, grid_size_y
            else:
                # If all else fails, try to find SiteSelection rows and count them
                site_selection_rows = []
                for line in htd_content.split('\n'):
                    if '"SiteSelection' in line:
                        parts = line.split(',')
                        if len(parts) > 1:
                            site_selection_rows.append(parts[1:])

                if site_selection_rows:
                    grid_size_y = len(site_selection_rows)
                    grid_size_x = len(site_selection_rows[0])
                    logger.info(f"Parsed HTD file from SiteSelection rows: grid size {grid_size_x}x{grid_size_y}")
                    return grid_size_x, grid_size_y
                else:
                    logger.warning(f"Could not extract grid dimensions from HTD file: {htd_path}")
                    return None, None
        except Exception as e:
            logger.error(f"Error parsing HTD file {htd_path}: {e}")
            return None, None

    @staticmethod
    def auto_detect_patterns(folder_path, well_filter=None):
        """
        Automatically detect image patterns in a folder.

        Args:
            folder_path (str or Path): Path to the folder
            well_filter (list): Optional list of wells to include

        Returns:
            dict: Dictionary mapping wells to wavelength patterns
        """
        folder_path = Path(folder_path)

        # Get all image files
        image_files = []
        for ext in ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            image_files.extend([f.name for f in folder_path.glob(f"*{ext}")])

        # Group by well and wavelength
        patterns_by_well = {}

        # Pattern to extract well, site, wavelength from filename
        # Example: A01_s001_w1.tif or A01_s001_w1_z001.tif
        filename_pattern = re.compile(r'([A-Z]\d+)_s(\d+)_w(\d+)(?:_z\d+)?\..*')

        for filename in image_files:
            match = filename_pattern.match(filename)
            if match:
                well = match.group(1)
                site_str = match.group(2)
                site = int(site_str)
                wavelength = int(match.group(3))

                # Skip if well is not in filter
                if well_filter and well not in well_filter:
                    continue

                # Add to patterns dictionary
                if well not in patterns_by_well:
                    patterns_by_well[well] = {}

                # Create pattern for this wavelength
                # First, ensure the site number is padded using the utility function
                padded_filename = ZStackManager.pad_site_number(filename)

                # Now replace the padded site with the placeholder
                pattern = re.sub(r'_s\d{3}', '_s{iii}', padded_filename)

                # Remove z-index if present
                pattern = re.sub(r'_z\d+', '', pattern)

                # Add to patterns
                patterns_by_well[well][str(wavelength)] = pattern

        return patterns_by_well

    @staticmethod
    def prepare_reference_channel(well, wavelength_patterns, dirs, reference_channels, preprocessing_funcs=None, composite_weights=None):
        """
        Prepare the reference channel for stitching.

        Args:
            well (str): Well ID
            wavelength_patterns (dict): Dictionary mapping wavelengths to patterns
            dirs (dict): Dictionary of directories
            reference_channels (list): List of reference channels
            preprocessing_funcs (dict): Dictionary mapping wavelengths to preprocessing functions
            composite_weights (dict): Dictionary mapping wavelengths to weights for composite

        Returns:
            tuple: (ref_channel, ref_pattern, ref_dir, updated_patterns)
        """
        # Create processed directory if needed
        processed_dir = ensure_directory(dirs['processed'])

        # Make a copy of patterns that we can modify
        updated_patterns = wavelength_patterns.copy()

        # Determine reference channel
        if len(reference_channels) == 1:
            # Single reference channel
            ref_channel = reference_channels[0]

            if ref_channel in wavelength_patterns:
                ref_pattern = wavelength_patterns[ref_channel]
                ref_dir = dirs['input']

                # Apply preprocessing if needed
                if preprocessing_funcs and ref_channel in preprocessing_funcs:
                    logger.info(f"Applying preprocessing to reference channel {ref_channel}")

                    # Process images
                    StitcherManager.process_imgs_from_pattern(
                        dirs['input'],
                        ref_pattern,
                        preprocessing_funcs[ref_channel],
                        processed_dir
                    )

                    # Use processed directory for reference
                    ref_dir = processed_dir

                logger.info(f"Using single reference channel {ref_channel} with pattern {ref_pattern}")
                return ref_channel, ref_pattern, ref_dir, updated_patterns
            else:
                logger.error(f"Reference channel {ref_channel} not found in patterns: {wavelength_patterns}")
                return None, None, None, updated_patterns
        else:
            # Multiple reference channels - create composite
            logger.info(f"Creating composite from channels: {reference_channels}")

            # Check if all reference channels exist
            for channel in reference_channels:
                if channel not in wavelength_patterns:
                    logger.error(f"Reference channel {channel} not found in patterns: {wavelength_patterns}")
                    return None, None, None, updated_patterns

            # Create composite pattern
            composite_pattern = StitcherManager.create_composite_reference(
                well,
                dirs['input'],
                processed_dir,
                reference_channels,
                {channel: wavelength_patterns[channel] for channel in reference_channels},
                preprocessing_funcs,
                composite_weights
            )

            # Add composite to patterns
            updated_patterns['composite'] = composite_pattern

            logger.info(f"Created composite reference with pattern {composite_pattern}")
            return 'composite', composite_pattern, processed_dir, updated_patterns

    @staticmethod
    def create_composite_reference(well, input_dir, processed_dir, reference_channels, channel_patterns, preprocessing_funcs=None, channel_weights=None):
        """
        Create a composite reference image from multiple channels.

        Args:
            well (str): Well ID
            input_dir (str or Path): Input directory
            processed_dir (str or Path): Output directory for processed images
            reference_channels (list): List of reference channels
            channel_patterns (dict): Dictionary mapping channels to patterns
            preprocessing_funcs (dict): Dictionary mapping channels to preprocessing functions
            channel_weights (dict): Dictionary mapping channels to weights

        Returns:
            str: Pattern for the composite images
        """
        input_dir = Path(input_dir)
        processed_dir = ensure_directory(processed_dir)

        # Create composite pattern
        first_channel = reference_channels[0]
        first_pattern = channel_patterns[first_channel]

        # Create composite pattern with the expected format
        composite_pattern = f"composite_{well}_s{{iii}}_{well}_w1.tif"

        # Get files for each channel
        channel_files = {}
        for channel in reference_channels:
            pattern = channel_patterns[channel]
            files = path_list_from_pattern(input_dir, pattern)
            channel_files[channel] = files

        # Check if all channels have the same number of files
        file_counts = [len(files) for files in channel_files.values()]
        if len(set(file_counts)) > 1:
            logger.error(f"Channels have different numbers of files: {file_counts}")
            return None

        # Process each site
        for i in range(file_counts[0]):
            # Load images for each channel
            position_images = {}

            for channel in reference_channels:
                # Get file for this channel and site
                filename = channel_files[channel][i]
                file_path = input_dir / filename

                # Load image
                img = load_image(file_path)
                if img is None:
                    logger.error(f"Failed to load image: {file_path}")
                    continue

                # Apply preprocessing if needed
                if preprocessing_funcs and channel in preprocessing_funcs:
                    img = preprocessing_funcs[channel]([img])[0]

                # Add to position images
                position_images[channel] = img

            # Create composite using the image processing function
            composite = ImageProcessor.create_weighted_composite(position_images, channel_weights)

            # Extract site number from filename using existing pattern
            site_num = None
            filename = channel_files[first_channel][i]
            match = re.search(r's(\d+)', filename)
            if match:
                # Ensure site number is padded to 3 digits
                site_num = match.group(1).zfill(3)
            else:
                site_num = f"{i:03d}"

            # Generate output filename with the correct site number
            out_filename = composite_pattern.replace("{iii}", site_num)
            out_path = processed_dir / out_filename

            # Save composite reference to processed directory
            save_image(out_path, composite)

        return composite_pattern

    @staticmethod
    def process_imgs_from_pattern(image_dir, image_pattern, function, out_dir):
        """
        Process all images matching a pattern with the given function.

        Args:
            image_dir (str or Path): Directory containing images
            image_pattern (str): Pattern to match
            function (callable): Function to apply to images
            out_dir (str or Path): Output directory

        Returns:
            int: Number of processed images
        """
        image_dir = Path(image_dir)
        out_dir = ensure_directory(out_dir)

        # Get matching files
        image_names = path_list_from_pattern(image_dir, image_pattern)
        if not image_names:
            logger.warning(f"No images found matching pattern {image_pattern} in {image_dir}")
            return 0

        # Load images
        images = []
        for name in image_names:
            img = load_image(image_dir / name)
            if img is not None:
                images.append(img)

        # Process images
        processed = function(images)

        # Save processed images
        for img, name in zip(processed, image_names):
            save_image(out_dir / name, img)

        return len(processed)

    @staticmethod
    def ashlar_stitch_v2(image_dir, image_pattern, positions_path, grid_size_x, grid_size_y, tile_overlap=10, tile_overlap_x=None, tile_overlap_y=None, max_shift=20, pixel_size=1):
        """
        Stitch images using the Ashlar library.

        Args:
            image_dir (str or Path): Directory containing images
            image_pattern (str): Pattern with '{iii}' placeholder
            positions_path (str or Path): Path to save positions CSV
            grid_size_x (int): Number of tiles horizontally
            grid_size_y (int): Number of tiles vertically
            tile_overlap (float): Overlap percentage (used for both x and y)
            tile_overlap_x (float): Deprecated - use tile_overlap instead
            tile_overlap_y (float): Deprecated - use tile_overlap instead
            max_shift (int): Maximum allowed error in microns
            pixel_size (float): Size of pixel in microns

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            image_dir = Path(image_dir)
            positions_path = Path(positions_path)

            # Log warning if deprecated parameters are used
            if tile_overlap_x is not None or tile_overlap_y is not None:
                logger.warning("tile_overlap_x and tile_overlap_y are deprecated. Using tile_overlap for both directions.")

            # Convert overlap from percentage to fraction
            overlap = tile_overlap / 100.0

            # Replace {iii} with {series} for Ashlar
            ashlar_pattern = image_pattern.replace("{iii}", "{series}")
            logger.info(f"Using pattern: {ashlar_pattern} for ashlar")

            # Create a FileSeriesReader for the images
            fs_reader = fileseries.FileSeriesReader(
                path=str(image_dir),
                pattern=ashlar_pattern,
                overlap=overlap,  # Using single overlap value for now
                width=grid_size_x,
                height=grid_size_y,
                layout="raster",
                direction="horizontal",
                pixel_size=pixel_size,
            )

            # Align the tiles using EdgeAligner
            aligner = reg.EdgeAligner(
                fs_reader,
                channel=0,  # If multi-channel, pick the channel to align on
                filter_sigma=0,  # adjust if needed
                verbose=True,
                max_shift=max_shift
            )
            aligner.run()

            # Build a Mosaic from the alignment
            mosaic_args = {
                'verbose': True,
                'flip_mosaic_y': False  # if your final mosaic needs flipping
            }
            mosaic = reg.Mosaic(
                aligner,
                aligner.mosaic_shape,
                **mosaic_args
            )

            # Extract positions and generate CSV
            positions = [(y, x) for x, y in mosaic.aligner.positions]

            # Use the original pattern (with {iii} instead of {series})
            original_pattern = image_pattern.replace("{series}", "{iii}")

            # Generate positions DataFrame
            positions_df = generate_positions_df(str(image_dir), original_pattern, positions, grid_size_x, grid_size_y)

            # Save to CSV
            positions_path.parent.mkdir(parents=True, exist_ok=True)
            positions_df.to_csv(positions_path, index=False, sep=";", header=False)

            logger.info(f"Saved positions to {positions_path}")
            return True

        except Exception as e:
            logger.error(f"Error in ashlar_stitch_v2: {e}")
            return False

    @staticmethod
    def process_well_wavelengths(well, wavelength_patterns, dirs, grid_dims, ref_channel, ref_pattern, ref_dir, margin_ratio=0.1, tile_overlap=10, tile_overlap_x=None, tile_overlap_y=None, max_shift=50):
        """
        Process all wavelengths for a well.

        Args:
            well (str): Well ID
            wavelength_patterns (dict): Dictionary mapping wavelengths to patterns
            dirs (dict): Dictionary of directories
            grid_dims (tuple): Grid dimensions (grid_size_x, grid_size_y)
            ref_channel (str): Reference channel
            ref_pattern (str): Reference pattern
            ref_dir (str or Path): Reference directory
            margin_ratio (float): Blending margin ratio
            tile_overlap (float): Overlap percentage
            tile_overlap_x (float): Horizontal overlap percentage
            tile_overlap_y (float): Vertical overlap percentage
            max_shift (int): Maximum allowed error in microns

        Returns:
            bool: True if successful, False otherwise
        """
        grid_size_x, grid_size_y = grid_dims

        # Generate positions using Ashlar
        stitched_name = StitcherManager.compute_stitched_name(ref_pattern)
        positions_path = dirs['positions'] / f"{Path(stitched_name).stem}.csv"

        logger.info(f"Generating positions using Ashlar with pattern: {ref_pattern}")
        logger.info(f"Reading reference images from: {ref_dir}")

        # Run Ashlar to generate positions
        success = StitcherManager.ashlar_stitch_v2(
            image_dir=ref_dir,
            image_pattern=ref_pattern,
            positions_path=positions_path,
            grid_size_x=grid_size_x,
            grid_size_y=grid_size_y,
            tile_overlap=tile_overlap,
            tile_overlap_x=tile_overlap_x,
            tile_overlap_y=tile_overlap_y,
            max_shift=max_shift
        )

        if not success:
            logger.error(f"Failed to generate positions for {well}")
            return False

        # Process each wavelength
        for wavelength, pattern in wavelength_patterns.items():
            # Skip the reference channel if it's the composite
            if wavelength == 'composite' and ref_channel == 'composite':
                logger.info(f"Skipping assembly of composite channel (used only for alignment)")
                continue

            # Use original image directory for assembly
            img_dir = dirs['input']

            # Get files for this wavelength to override the composite filenames
            override_names = path_list_from_pattern(img_dir, pattern)

            # Assemble final image
            stitched_name = StitcherManager.compute_stitched_name(pattern)
            output_path = dirs['stitched'] / stitched_name

            logger.info(f"Assembling wavelength {wavelength} from {img_dir} to {output_path}")

            # Use the image processor to assemble the image
            success = ImageProcessor.assemble_image_subpixel(
                positions_path=positions_path,
                images_dir=img_dir,
                output_path=output_path,
                margin_ratio=margin_ratio,
                override_names=override_names
            )

            if not success:
                logger.error(f"Failed to assemble wavelength {wavelength} for {well}")

        logger.info(f"Completed processing well {well}")
        return True

    @staticmethod
    def compute_stitched_name(pattern):
        """
        Remove the 's{iii}_' or 's{iii}' portion from the pattern,
        returning the rest as the final stitched filename.

        Examples:
          pattern = "mfd-ctb_A05_s{iii}_w1.tif" -> "mfd-ctb_A05_w1.tif"
          pattern = "mfd-ctb_B06_s{iii}w1.tif"  -> "mfd-ctb_B06_w1.tif"
        """
        # Handle dictionary patterns if needed
        if isinstance(pattern, dict):
            pattern = pattern.get('pattern', '')

        pattern = re.sub(r"\{.*?\}", f"{{{'iii'}}}", pattern)
        if "s{iii}_" in pattern:
            stitched_name = pattern.replace("s{iii}_", "")
        else:
            stitched_name = pattern.replace("s{iii}", "")
        return stitched_name

    @staticmethod
    def process_plate_folder(plate_folder, reference_channels=['1'], preprocessing_funcs=None, margin_ratio=0.1, composite_weights=None, well_filter=None, tile_overlap=6.5, tile_overlap_x=None, tile_overlap_y=None, max_shift=50, focus_detect=False, focus_method="combined", create_projections=False, stitch_z_reference='best_focus', save_projections=True, stitch_all_z_planes=False, use_reference_positions=False):
        """
        Process an entire plate folder with microscopy images.

        Args:
            plate_folder (str): Path to the plate folder
            reference_channels (list): List of channels to use for reference
            preprocessing_funcs (list): List of preprocessing functions
            margin_ratio (float): Margin ratio for blending
            composite_weights (dict): Weights for composite images
            well_filter (list): List of wells to process
            tile_overlap (float): Percentage of overlap between tiles
            tile_overlap_x (float): Horizontal overlap percentage
            tile_overlap_y (float): Vertical overlap percentage
            max_shift (int): Maximum shift allowed between tiles in microns
            focus_detect (bool): Whether to detect best focus for Z-stacks
            focus_method (str): Method for focus detection
            create_projections (bool): Whether to create projections for Z-stacks
            stitch_z_reference (str): Reference to use for Z-stack stitching ('best_focus', 'max', 'mean')
            save_projections (bool): Whether to save projections
            stitch_all_z_planes (bool): Whether to stitch all Z-planes using reference positions

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            plate_path = Path(plate_folder)

            # Create output directories
            parent_dir = plate_path.parent
            plate_name = plate_path.name

            dirs = {
                'input': plate_path / "TimePoint_1",
                'processed': ensure_directory(parent_dir / f"{plate_name}_processed" / "TimePoint_1"),
                'positions': ensure_directory(parent_dir / f"{plate_name}_positions"),
                'stitched': ensure_directory(parent_dir / f"{plate_name}_stitched" / "TimePoint_1")
            }

            # Check if input directory exists
            if not dirs['input'].exists():
                logger.error(f"Input directory does not exist: {dirs['input']}")
                return False

            # 1. Detect and handle Z-stacks
            has_zstack, z_info = ZStackManager.preprocess_plate_folder(plate_folder)

            # Handle Z-stack processing if needed
            best_focus_dir = None
            projections_dir = None

            if has_zstack:
                logger.info(f"Z-stack detected in {plate_folder}")

                if focus_detect:
                    # Create best focus directory with correct capitalization
                    best_focus_dir = parent_dir / f"{plate_name}_BestFocus" / "TimePoint_1"
                    ensure_directory(best_focus_dir)

                    # Find best focused images
                    logger.info(f"Finding best focused images using method: {focus_method}")
                    success, best_focus_results_dir = ZStackManager.select_best_focus_zstack(
                        dirs['input'],
                        best_focus_dir,
                        focus_method=focus_method,
                        focus_wavelength=reference_channels[0]
                    )

                    if not success:
                        logger.warning("No best focus images created")

                if create_projections:
                    # Create projections directory with projection type in the name
                    projections_dir = parent_dir / f"{plate_name}_{stitch_z_reference}" / "TimePoint_1"
                    ensure_directory(projections_dir)

                    # Create projections
                    logger.info(f"Creating projection: {stitch_z_reference}")
                    success, projections_results_dir = ZStackManager.create_zstack_projections(
                        dirs['input'],
                        projections_dir,
                        projection_types=[stitch_z_reference]  # Pass as a list with single item
                    )

                    if not success:
                        logger.warning("No projections created")

                # Determine which directory to use for stitching
                stitch_source = plate_folder
                if stitch_z_reference == 'best_focus' and best_focus_dir:
                    stitch_source = best_focus_dir.parent  # Use the parent directory (synthetic_plate_BestFocus)
                    logger.info(f"Using best focus images for stitching from {best_focus_dir}")
                elif stitch_z_reference in ['max', 'mean'] and projections_dir:
                    # Use the specified projection type
                    stitch_source = projections_dir.parent  # Use the parent directory (synthetic_plate_max)
                    logger.info(f"Using {stitch_z_reference} projections for stitching from {projections_dir}")
                elif stitch_z_reference in ['max', 'mean', 'best_focus'] and stitch_all_z_planes:
                    # Handle 3D stitching using reference for alignment
                    logger.info(f"Stitching all Z-planes using {stitch_z_reference} as reference")

                    # Prepare for Z-stack stitching
                    success = ZStackManager.stitch_across_z(
                        plate_folder,
                        reference_z=stitch_z_reference,
                        stitch_all_z_planes=True,
                        **{
                            'reference_channels': reference_channels,
                            'preprocessing_funcs': preprocessing_funcs,
                            'margin_ratio': margin_ratio,
                            'composite_weights': composite_weights,
                            'well_filter': well_filter,
                            'tile_overlap': tile_overlap,
                            'tile_overlap_x': tile_overlap_x,
                            'tile_overlap_y': tile_overlap_y,
                            'max_shift': max_shift
                        }
                    )

                    # Return early with the result from stitch_across_z
                    return success
            else:
                # No Z-stack detected, use original folder
                stitch_source = plate_folder
                logger.info(f"No Z-stack detected in {plate_folder}, using standard stitching")

            # Update input directory if using a different source
            if stitch_source != plate_folder:
                dirs['input'] = Path(stitch_source) / "TimePoint_1"

            # 2. Find HTD file to get grid dimensions
            htd_file = StitcherManager.find_HTD_file(plate_folder)
            if htd_file:
                grid_size_x, grid_size_y = StitcherManager.parse_HTD_file(htd_file)
                if grid_size_x is None or grid_size_y is None:
                    logger.warning("Could not parse grid dimensions from HTD file, using default 3x3")
                    grid_size_x, grid_size_y = 3, 3
            else:
                logger.warning("No HTD file found, using default grid size 3x3")
                grid_size_x, grid_size_y = 3, 3

            grid_dims = (grid_size_x, grid_size_y)
            logger.info(f"Using grid dimensions: {grid_size_x}x{grid_size_y}")

            # 3. Auto-detect patterns
            # Use the stitch_source/TimePoint_1 directory for pattern detection
            patterns_by_well = StitcherManager.auto_detect_patterns(dirs['input'], well_filter)
            if not patterns_by_well:
                logger.error(f"No image patterns detected in {dirs['input']}")
                return False

            logger.info(f"Detected {len(patterns_by_well)} wells with images")

            # 4. Process each well
            for well, wavelength_patterns in patterns_by_well.items():
                logger.info(f"\nProcessing well {well} with {len(wavelength_patterns)} wavelength(s)")

                # Prepare reference channel
                ref_channel, ref_pattern, ref_dir, updated_patterns = StitcherManager.prepare_reference_channel(
                    well, wavelength_patterns, dirs, reference_channels, preprocessing_funcs,
                    composite_weights
                )

                if ref_channel is None:
                    logger.error(f"Failed to prepare reference channel for well {well}")
                    continue

                # Process all wavelengths using the reference
                success = StitcherManager.process_well_wavelengths(
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
            logger.error(f"Error in process_plate_folder: {e}")
            import traceback
            traceback.print_exc()
            return False
