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
from ezstitcher.core.image_preprocessor import create_linear_weight_mask
from ezstitcher.core.microscope_interfaces import FilenameParser

logger = logging.getLogger(__name__)





class Stitcher:
    """
    Class for handling image stitching operations.
    """

    def __init__(self, config: Optional[StitcherConfig] = None, filename_parser: Optional[FilenameParser] = None):
        """
        Initialize the Stitcher.

        Args:
            config (StitcherConfig): Configuration for stitching
            filename_parser (FilenameParser): Parser for microscopy filenames
        """
        self.config = config or StitcherConfig()
        self.fs_manager = FileSystemManager()
        self.filename_parser = filename_parser

    def generate_positions_df(self, image_dir, image_pattern, positions, grid_size_x, grid_size_y):
        """
        Given an image_dir, an image_pattern (with '{iii}' or similar placeholder)
        and a list of (x, y) tuples 'positions', build a DataFrame with lines like:

          file: <filename>; position: (x, y); grid: (col, row);
        """
        all_files = self.filename_parser.path_list_from_pattern(image_dir, image_pattern)
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
        return self._generate_positions_ashlar(image_dir, image_pattern, positions_path, grid_size_x, grid_size_y)

    def _generate_positions_ashlar(self, image_dir: Union[str, Path],
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

            # Check if the pattern has .tif extension, but files have .tiff extension
            if (image_pattern.endswith('.tif') and
                not self.filename_parser.path_list_from_pattern(image_dir, image_pattern)):
                # Try with .tiff extension
                tiff_pattern = image_pattern[:-4] + '.tiff'
                if self.filename_parser.path_list_from_pattern(image_dir, tiff_pattern):
                    image_pattern = tiff_pattern
                    ashlar_pattern = image_pattern.replace("{iii}", "{series}")
                    logger.info(f"Updated pattern to: {ashlar_pattern} for ashlar")

            # Check if there are enough files for the grid size
            files = self.filename_parser.path_list_from_pattern(image_dir, image_pattern)


            if len(files) < grid_size_x * grid_size_y:
                logger.error(f"Not enough files for grid size {grid_size_x}x{grid_size_y}. Found {len(files)} files.")
                return False

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
            logger.error(f"Error in generate_positions_ashlar: {e}")
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
            from ezstitcher.core.csv_handler import CSVHandler
            pos_entries = CSVHandler.parse_positions_csv(positions_path)
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