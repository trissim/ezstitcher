"""
Stitcher module for ezstitcher.

This module contains the Stitcher class for handling image stitching operations.
"""

import re
import os
import logging
from pathlib import Path
from typing import List, Optional, Union
from scipy.ndimage import shift as subpixel_shift

import numpy as np
import pandas as pd
from ashlar import fileseries, reg

from ezstitcher.core.config import StitcherConfig
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.image_processor import create_linear_weight_mask
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
            raise ValueError(f"File/position count mismatch: {len(all_files)}≠{len(positions)}")

        # Check if grid size matches the number of files
        total_grid_size = grid_size_x * grid_size_y
        if total_grid_size != len(all_files):
            # Raise an error if the grid size doesn't match the number of files
            raise ValueError(f"Grid size mismatch: {grid_size_x}×{grid_size_y}≠{len(all_files)}")

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

            # Deprecated code removed - we now only use tile_overlap

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


            if len(files) != grid_size_x * grid_size_y:
                raise ValueError(f"Grid size mismatch: {grid_size_x}×{grid_size_y}≠{len(files)}")

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
                'flip_mosaic_y': False,  # if your final mosaic needs flipping
                # 'num_workers': 1  # This parameter is not supported by Ashlar's Mosaic class
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
            self.save_positions_df(positions_df, positions_path)

            logger.info("Saved positions to %s", positions_path)
            return True

        except Exception as e:
            logger.error("Error in generate_positions_ashlar: %s", e)
            return False

    @staticmethod
    def parse_positions_csv(csv_path):
        """
        Parse a CSV file with lines of the form:
          file: <filename>; grid: (col, row); position: (x, y)

        Args:
            csv_path (str or Path): Path to the CSV file

        Returns:
            list: List of tuples (filename, x_float, y_float)
        """
        entries = []
        with open(csv_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # Example line:
                # file: some_image.tif; grid: (0, 0); position: (123.45, 67.89)
                file_match = re.search(r'file:\s*([^;]+);', line)
                pos_match = re.search(r'position:\s*\(([^,]+),\s*([^)]+)\)', line)
                if file_match and pos_match:
                    fname = file_match.group(1).strip()
                    x_val = float(pos_match.group(1).strip())
                    y_val = float(pos_match.group(2).strip())
                    entries.append((fname, x_val, y_val))
        return entries

    @staticmethod
    def save_positions_df(df, positions_path):
        """
        Save a positions DataFrame to CSV.

        Args:
            df (pandas.DataFrame): DataFrame to save
            positions_path (str or Path): Path to save the CSV file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            Path(positions_path).parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            df.to_csv(positions_path, index=False, sep=";", header=False)
            return True
        except Exception as e:
            logger.error("Error saving positions CSV: %s", e)
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
            output_dir = output_path.parent
            self.fs_manager.ensure_directory(output_dir)
            logger.info("Ensured output directory exists: %s", output_dir)

            # Parse CSV file
            pos_entries = self.parse_positions_csv(positions_path)
            if not pos_entries:
                logger.error("No valid entries found in %s", positions_path)
                return False

            # Override filenames if provided
            if override_names is not None:
                if len(override_names) != len(pos_entries):
                    raise ValueError(f"Override names/positions mismatch: {len(override_names)}≠{len(pos_entries)}")

                pos_entries = [(override_names[i], x, y) for i, (_, x, y) in enumerate(pos_entries)]

            # Check tile existence
            images_dir = Path(images_dir)
            for (fname, _, _) in pos_entries:
                if not (images_dir / fname).exists():
                    logger.error("Missing image: %s in %s", fname, images_dir)
                    return False

            # Read the first tile to get shape, dtype
            first_tile = self.fs_manager.load_image(images_dir / pos_entries[0][0])
            if first_tile is None:
                logger.error("Failed to load first tile: %s", pos_entries[0][0])
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
            logger.info("Final canvas size: %d x %d", final_h, final_w)

            # Prepare accumulators
            acc = np.zeros((final_h, final_w), dtype=np.float32)
            weight_acc = np.zeros((final_h, final_w), dtype=np.float32)

            # Prepare the tile mask
            base_mask = create_linear_weight_mask(tile_h, tile_w, margin_ratio=margin_ratio)

            # Process each tile
            for i, (fname, x_f, y_f) in enumerate(pos_entries):
                logger.info("Placing tile %d/%d: %s at (%.2f, %.2f)", i+1, len(pos_entries), fname, x_f, y_f)

                # Load tile
                tile_img = self.fs_manager.load_image(images_dir / fname)
                if tile_img is None:
                    logger.error("Failed to load tile: %s", fname)
                    continue

                # Check shape and dtype
                if tile_img.shape != (tile_h, tile_w):
                    logger.error("Tile shape mismatch: %s vs %dx%d", tile_img.shape, tile_h, tile_w)
                    continue

                if tile_img.dtype != dtype:
                    logger.error("Tile dtype mismatch: %s vs %s", tile_img.dtype, dtype)
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
            logger.info("Saving stitched image to %s", output_path)
            self.fs_manager.save_image(output_path, blended)

            return True

        except Exception as e:
            logger.error("Error in assemble_image: %s", e)
            return False
