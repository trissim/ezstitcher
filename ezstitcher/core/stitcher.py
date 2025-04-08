"""
Stitcher module for ezstitcher.

This module contains the Stitcher class for handling image stitching operations.
"""

import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.ndimage import shift as subpixel_shift

from ashlar import fileseries, reg

from ezstitcher.core.config import StitcherConfig
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.utils import create_linear_weight_mask

logger = logging.getLogger(__name__)





class Stitcher:
    """
    Class for handling image stitching operations.
    """

    def __init__(self, config: Optional[StitcherConfig] = None):
        """
        Initialize the Stitcher.

        Args:
            config (StitcherConfig): Configuration for stitching
        """
        self.config = config or StitcherConfig()
        self.fs_manager = FileSystemManager()

    def generate_positions_df(self, image_dir, image_pattern, positions, grid_size_x, grid_size_y):
        """
        Given an image_dir, an image_pattern (with '{iii}' or similar placeholder)
        and a list of (x, y) tuples 'positions', build a DataFrame with lines like:

          file: <filename>; position: (x, y); grid: (col, row);
        """
        all_files = self.fs_manager.path_list_from_pattern(image_dir, image_pattern)
        if len(all_files) != len(positions):
            logger.warning(
                f"Number of matched files ({len(all_files)}) != number of positions ({len(positions)}). "
                f"Adjusting grid size to match file count."
            )

        # Adjust grid size if needed
        total_grid_size = grid_size_x * grid_size_y
        if total_grid_size < len(all_files):
            # If grid is too small, increase it to fit all files
            new_size = int(np.ceil(np.sqrt(len(all_files))))
            logger.warning(f"Grid size {grid_size_x}x{grid_size_y} is too small for {len(all_files)} files. "
                          f"Adjusting to {new_size}x{new_size}.")
            grid_size_x = grid_size_y = new_size

        # Generate a list of (x, y) grid positions following a raster pattern
        positions_grid = [(x, y) for y in range(grid_size_y) for x in range(grid_size_x)]

        # Ensure we don't try to access beyond the available positions
        num_positions = min(len(all_files), len(positions), len(positions_grid))
        data_rows = []

        for i in range(num_positions):
            fname = all_files[i]
            x, y = positions[i]
            row, col = positions_grid[i]

            data_rows.append({
                "file": "file: " + fname,
                "grid": " grid: " + "("+str(row)+", "+str(col)+")",
                "position": " position: " + "("+str(x)+", "+str(y)+")",
            })

        df = pd.DataFrame(data_rows)
        return df

    def auto_detect_patterns(self, folder_path, well_filter=None):
        """
        Automatically detect image patterns in a folder.

        Args:
            folder_path (str or Path): Path to the folder
            well_filter (list): Optional list of wells to include

        Returns:
            dict: Dictionary mapping wells to wavelength patterns
        """
        folder_path = Path(folder_path)

        # Get all image files using FileSystemManager
        image_files = [f.name for f in self.fs_manager.list_image_files(folder_path)]

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
                # First, ensure the site number is padded to 3 digits
                padded_site = f"{site:03d}"
                padded_filename = filename.replace(f"_s{site_str}", f"_s{padded_site}")

                # Now replace the padded site with the placeholder
                pattern = re.sub(r'_s\d{3}', '_s{iii}', padded_filename)

                # Remove z-index if present
                pattern = re.sub(r'_z\d+', '', pattern)

                # Add to patterns
                patterns_by_well[well][str(wavelength)] = pattern

        return patterns_by_well

    def prepare_reference_channel(self, well, wavelength_patterns, dirs, reference_channels, preprocessing_funcs=None, composite_weights=None):
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
        processed_dir = self.fs_manager.ensure_directory(dirs['processed'])
        logger.info(f"Using processed directory: {processed_dir}")

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
                    self.process_imgs_from_pattern(
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
            logger.info(f"Creating composite reference with input_dir={dirs['input']}, processed_dir={processed_dir}")
            composite_pattern = self.create_composite_reference(
                well,
                dirs['input'],
                processed_dir,
                reference_channels,
                {channel: wavelength_patterns[channel] for channel in reference_channels},
                preprocessing_funcs,
                composite_weights
            )

            if composite_pattern is None:
                logger.error(f"Failed to create composite reference for well {well}")
                return None, None, None, updated_patterns

            # Add composite to patterns
            updated_patterns['composite'] = composite_pattern

            # Check if composite files were created
            composite_files = self.fs_manager.path_list_from_pattern(processed_dir, composite_pattern)
            logger.info(f"Created {len(composite_files)} composite files: {composite_files}")

            logger.info(f"Created composite reference with pattern {composite_pattern}")
            return 'composite', composite_pattern, processed_dir, updated_patterns

    def create_composite_reference(self, well, input_dir, processed_dir, reference_channels, channel_patterns, preprocessing_funcs=None, channel_weights=None):
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
        processed_dir = self.fs_manager.ensure_directory(processed_dir)

        # Create composite pattern
        first_channel = reference_channels[0]
        first_pattern = channel_patterns[first_channel]

        # Create composite pattern with the expected format
        composite_pattern = f"composite_{well}_s{{iii}}_{well}_w1.tif"

        # Get files for each channel
        channel_files = {}
        for channel in reference_channels:
            pattern = channel_patterns[channel]
            files = self.fs_manager.path_list_from_pattern(input_dir, pattern)
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
                img = self.fs_manager.load_image(file_path)
                if img is None:
                    logger.error(f"Failed to load image: {file_path}")
                    continue

                # Apply preprocessing if needed
                if preprocessing_funcs and channel in preprocessing_funcs:
                    img = preprocessing_funcs[channel]([img])[0]

                # Add to position images
                position_images[channel] = img

            # Create composite using weighted average
            composite = self.create_weighted_composite(position_images, channel_weights)

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

            # Ensure the processed directory exists
            self.fs_manager.ensure_directory(processed_dir)

            # Save composite reference to processed directory
            out_path = processed_dir / out_filename
            logger.info(f"Saving composite image to {out_path}")
            self.fs_manager.save_image(out_path, composite)

        return composite_pattern

    def process_imgs_from_pattern(self, image_dir, image_pattern, function, out_dir):
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
        out_dir = self.fs_manager.ensure_directory(out_dir)

        # Get matching files
        image_names = self.fs_manager.path_list_from_pattern(image_dir, image_pattern)
        if not image_names:
            logger.warning(f"No images found matching pattern {image_pattern} in {image_dir}")
            return 0

        # Load images
        images = []
        for name in image_names:
            img = self.fs_manager.load_image(image_dir / name)
            if img is not None:
                images.append(img)

        # Process images
        processed = function(images)

        # Save processed images
        for img, name in zip(processed, image_names):
            self.fs_manager.save_image(out_dir / name, img)

        return len(processed)

    def create_weighted_composite(self, channel_images, weights=None):
        """
        Create a weighted composite from multiple channel images.

        Args:
            channel_images (dict): Dictionary mapping channels to images
            weights (dict): Dictionary mapping channels to weights

        Returns:
            numpy.ndarray: Composite image
        """
        if not channel_images:
            return None

        # Get first image to determine shape and dtype
        first_channel = next(iter(channel_images))
        first_image = channel_images[first_channel]
        shape = first_image.shape
        dtype = first_image.dtype

        # Initialize composite as float32 for calculations
        composite = np.zeros(shape, dtype=np.float32)
        total_weight = 0.0

        # Add each channel with appropriate weight
        for channel, image in channel_images.items():
            # Skip if image shape doesn't match
            if image.shape != shape:
                logger.warning(f"Skipping channel {channel} due to shape mismatch: {image.shape} vs {shape}")
                continue

            # Get weight for this channel (default to 1.0)
            weight = 1.0
            if weights and channel in weights:
                weight = weights[channel]

            # Add to composite
            composite += image.astype(np.float32) * weight
            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            composite /= total_weight

        # Convert back to original dtype
        if np.issubdtype(dtype, np.integer):
            composite = np.clip(composite, 0, np.iinfo(dtype).max).astype(dtype)
        else:
            composite = composite.astype(dtype)

        return composite

    def compute_stitched_name(self, pattern: str) -> str:
        """
        Remove the 's{iii}_' or 's{iii}' portion from the pattern,
        returning the rest as the final stitched filename.

        Examples:
          pattern = "mfd-ctb_A05_s{iii}_w1.tif" -> "mfd-ctb_A05_w1.tif"
          pattern = "mfd-ctb_B06_s{iii}w1.tif"  -> "mfd-ctb_B06_w1.tif"

        Args:
            pattern (str): Pattern with {iii} placeholder

        Returns:
            str: Stitched filename
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

    def generate_positions(self, image_dir: Union[str, Path],
                          image_pattern: str,
                          positions_path: Union[str, Path],
                          grid_size_x: int,
                          grid_size_y: int) -> bool:
        """
        Generate positions for stitching using Ashlar.

        Args:
            image_dir (str or Path): Directory containing images
            image_pattern (str): Pattern with '{iii}' placeholder
            positions_path (str or Path): Path to save positions CSV
            grid_size_x (int): Number of tiles horizontally
            grid_size_y (int): Number of tiles vertically

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            image_dir = Path(image_dir)
            positions_path = Path(positions_path)

            # Get tile overlap from config
            tile_overlap = self.config.tile_overlap
            max_shift = self.config.max_shift
            pixel_size = self.config.pixel_size

            # Log warning if deprecated parameters are used
            if self.config.tile_overlap_x is not None or self.config.tile_overlap_y is not None:
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
            positions_df = self.generate_positions_df(str(image_dir), original_pattern, positions, grid_size_x, grid_size_y)

            # Save to CSV
            self.fs_manager.ensure_directory(positions_path.parent)
            positions_df.to_csv(positions_path, index=False, sep=";", header=False)

            logger.info(f"Saved positions to {positions_path}")
            return True

        except Exception as e:
            logger.error(f"Error in generate_positions: {e}")
            return False

    def assemble_image(self, positions_path: Union[str, Path],
                      images_dir: Union[str, Path],
                      output_path: Union[str, Path],
                      override_names: Optional[List[str]] = None) -> bool:
        """
        Assemble a stitched image using subpixel positions from a CSV file.

        Args:
            positions_path (str or Path): Path to the CSV with subpixel positions
            images_dir (str or Path): Directory containing image tiles
            output_path (str or Path): Path to save final stitched image
            override_names (list): Optional list of filenames to use instead of those in CSV

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get margin ratio from config
            margin_ratio = self.config.margin_ratio

            # Ensure output directory exists
            output_path = Path(output_path)
            self.fs_manager.ensure_directory(output_path.parent)

            # Parse CSV file
            pos_entries = self.fs_manager.parse_positions_csv(positions_path)
            if not pos_entries:
                logger.error(f"No valid entries found in {positions_path}")
                return False

            # Override filenames if provided
            if override_names is not None:
                if len(override_names) != len(pos_entries):
                    logger.error(f"Number of override_names ({len(override_names)}) doesn't match positions ({len(pos_entries)})")
                    return False

                pos_entries = [(override_names[i], x, y) for i, (_, x, y) in enumerate(pos_entries)]

            # Check tile existence
            images_dir = Path(images_dir)
            for (fname, _, _) in pos_entries:
                if not (images_dir / fname).exists():
                    logger.error(f"Missing image: {fname} in {images_dir}")
                    return False

            # Read the first tile to get shape, dtype
            first_tile = self.fs_manager.load_image(images_dir / pos_entries[0][0])
            if first_tile is None:
                logger.error(f"Failed to load first tile: {pos_entries[0][0]}")
                return False

            tile_h, tile_w = first_tile.shape
            dtype = first_tile.dtype

            # Compute bounding box
            x_vals = [x_f for _, x_f, _ in pos_entries]
            y_vals = [y_f for _, _, y_f in pos_entries]

            min_x = min(x_vals)
            max_x = max(x_vals) + tile_w
            min_y = min(y_vals)
            max_y = max(y_vals) + tile_h

            # Final canvas size
            final_w = int(np.ceil(max_x - min_x))
            final_h = int(np.ceil(max_y - min_y))
            logger.info(f"Final canvas size: {final_h} x {final_w}")

            # Prepare accumulators
            acc = np.zeros((final_h, final_w), dtype=np.float32)
            weight_acc = np.zeros((final_h, final_w), dtype=np.float32)

            # Prepare the tile mask
            base_mask = create_linear_weight_mask(tile_h, tile_w, margin_ratio=margin_ratio)

            # Process each tile
            for i, (fname, x_f, y_f) in enumerate(pos_entries):
                logger.info(f"Placing tile {i+1}/{len(pos_entries)}: {fname} at ({x_f}, {y_f})")

                # Load tile
                tile_img = self.fs_manager.load_image(images_dir / fname)
                if tile_img is None:
                    logger.error(f"Failed to load tile: {fname}")
                    continue

                # Check shape and dtype
                if tile_img.shape != (tile_h, tile_w):
                    logger.error(f"Tile shape mismatch: {tile_img.shape} vs {tile_h}x{tile_w}")
                    continue

                if tile_img.dtype != dtype:
                    logger.error(f"Tile dtype mismatch: {tile_img.dtype} vs {dtype}")
                    continue

                # Apply weight mask
                tile_float = tile_img.astype(np.float32)
                weighted_tile = tile_float * base_mask

                # Separate offset into integer + fractional
                shift_x = x_f - min_x
                shift_y = y_f - min_y
                int_x = int(np.floor(shift_x))
                int_y = int(np.floor(shift_y))
                frac_x = shift_x - int_x
                frac_y = shift_y - int_y

                # Shift by fractional portion
                shifted_tile = subpixel_shift(
                    weighted_tile,
                    shift=(frac_y, frac_x),
                    order=1,
                    mode='constant',
                    cval=0
                )

                shifted_mask = subpixel_shift(
                    base_mask,
                    shift=(frac_y, frac_x),
                    order=1,
                    mode='constant',
                    cval=0
                )

                # Place at integer offset
                y_start = int_y
                x_start = int_x
                y_end = y_start + tile_h
                x_end = x_start + tile_w

                # Accumulate
                acc[y_start:y_end, x_start:x_end] += shifted_tile
                weight_acc[y_start:y_end, x_start:x_end] += shifted_mask

            # Final blend
            safe_weight = np.where(weight_acc == 0, 1, weight_acc)
            blended = acc / safe_weight

            # Clip to original dtype
            if np.issubdtype(dtype, np.integer):
                max_val = np.iinfo(dtype).max
            else:
                max_val = np.finfo(dtype).max

            blended = np.clip(blended, 0, max_val).astype(dtype)

            # Save stitched image
            logger.info(f"Saving stitched image to {output_path}")
            self.fs_manager.save_image(output_path, blended)

            return True

        except Exception as e:
            logger.error(f"Error in assemble_image: {e}")
            return False

    def process_well_wavelengths(self, well, wavelength_patterns, dirs, grid_dims, ref_channel, ref_pattern, ref_dir, use_existing_positions=False):
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
            use_existing_positions (bool): Whether to use existing positions file instead of generating new ones

        Returns:
            bool: True if successful, False otherwise
        """
        grid_size_x, grid_size_y = grid_dims

        # Generate or use existing positions
        stitched_name = self.compute_stitched_name(ref_pattern)
        positions_path = dirs['positions'] / f"{Path(stitched_name).stem}.csv"

        if use_existing_positions:
            # Check if positions file exists
            if not positions_path.exists():
                logger.error(f"Positions file not found: {positions_path}")
                return False

            logger.info(f"Using existing positions from: {positions_path}")
        else:
            # Generate positions using Ashlar
            logger.info(f"Generating positions using pattern: {ref_pattern}")
            logger.info(f"Reading reference images from: {ref_dir}")

            # Run generate_positions to generate positions
            success = self.generate_positions(
                image_dir=ref_dir,
                image_pattern=ref_pattern,
                positions_path=positions_path,
                grid_size_x=grid_size_x,
                grid_size_y=grid_size_y
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
            override_names = self.fs_manager.path_list_from_pattern(img_dir, pattern)

            # Assemble final image
            stitched_name = self.compute_stitched_name(pattern)
            output_path = dirs['stitched'] / stitched_name

            logger.info(f"Assembling wavelength {wavelength} from {img_dir} to {output_path}")

            # Use the assemble_image method to assemble the image
            success = self.assemble_image(
                positions_path=positions_path,
                images_dir=img_dir,
                output_path=output_path,
                override_names=override_names
            )

            if not success:
                logger.error(f"Failed to assemble wavelength {wavelength} for {well}")

        logger.info(f"Completed processing well {well}")
        return True

    def stitch(self, image_dir: Union[str, Path], output_dir: Union[str, Path], pattern: str,
               positions_dir: Optional[Union[str, Path]] = None, grid_size_x: int = 3, grid_size_y: int = 3) -> bool:
        """
        Perform stitching on images in the given directory.

        Args:
            image_dir (str or Path): Directory containing images to stitch
            output_dir (str or Path): Directory to save stitched images
            pattern (str): Image pattern with '{iii}' placeholder
            positions_dir (str or Path): Directory to save/load positions
            grid_size_x (int): Number of tiles horizontally
            grid_size_y (int): Number of tiles vertically

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            image_dir = Path(image_dir)
            output_dir = Path(output_dir)

            # Create output directory
            self.fs_manager.ensure_directory(output_dir.parent)

            # Create positions directory if needed
            if positions_dir is None:
                positions_dir = image_dir.parent / f"{image_dir.name}_positions"
            positions_dir = Path(positions_dir)
            self.fs_manager.ensure_directory(positions_dir)

            # Generate positions
            positions_path = positions_dir / f"{pattern.replace('{iii}', 'positions')}.csv"

            if not positions_path.exists():
                logger.info(f"Generating positions for {pattern}")
                success = self.generate_positions(image_dir, pattern, positions_path, grid_size_x, grid_size_y)
                if not success:
                    logger.error(f"Failed to generate positions for {pattern}")
                    return False

            # Compute stitched filename
            stitched_name = self.compute_stitched_name(pattern)
            output_path = output_dir / stitched_name

            # Assemble image
            logger.info(f"Assembling stitched image for {pattern}")
            success = self.assemble_image(positions_path, image_dir, output_path)
            if not success:
                logger.error(f"Failed to assemble image for {pattern}")
                return False

            logger.info(f"Successfully stitched {pattern} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error in stitch: {e}")
            return False
