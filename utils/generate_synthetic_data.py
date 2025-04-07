#!/usr/bin/env python3
"""
Generate synthetic microscopy images for testing ezstitcher.

This script generates synthetic microscopy images with the following features:
- Multiple wavelengths (channels)
- Z-stack support with varying focus levels
- Cell-like structures (circular particles with varying eccentricity)
- Proper tiling with configurable overlap
- Realistic stage positioning errors
- HTD file generation for metadata

Usage:
    python generate_synthetic_data.py output_dir --grid-size 3 3 --wavelengths 2 --z-stack 3
"""

import os
import sys
import argparse
import random
import numpy as np
import tifffile
from pathlib import Path
from skimage import draw, filters, transform, util
from datetime import datetime


class SyntheticMicroscopyGenerator:
    """Generate synthetic microscopy images for testing."""

    def __init__(self,
                 output_dir,
                 grid_size=(3, 3),
                 image_size=(1024, 1024),
                 tile_size=(512, 512),
                 overlap_percent=10,
                 stage_error_px=5,
                 wavelengths=2,
                 z_stack_levels=1,
                 z_step_size=1.0,
                 num_cells=100,
                 cell_size_range=(10, 30),
                 cell_eccentricity_range=(0.1, 0.5),
                 cell_intensity_range=(5000, 20000),
                 background_intensity=500,
                 noise_level=100,
                 wavelength_params=None,
                 shared_cell_fraction=0.95,  # Fraction of cells shared between wavelengths
                 wavelength_intensities=None,  # Fixed intensities for each wavelength
                 wavelength_backgrounds=None,  # Background intensities for each wavelength
                 wells=['A01'],  # List of wells to generate
                 random_seed=None):
        """
        Initialize the synthetic microscopy generator.

        Args:
            output_dir: Directory to save generated images
            grid_size: Tuple of (rows, cols) for the grid of tiles
            image_size: Size of the full image before tiling
            tile_size: Size of each tile
            overlap_percent: Percentage of overlap between tiles
            stage_error_px: Random error in stage positioning (pixels)
            wavelengths: Number of wavelength channels to generate
            z_stack_levels: Number of Z-stack levels to generate
            z_step_size: Spacing between Z-steps in microns
            num_cells: Number of cells to generate
            cell_size_range: Range of cell sizes (min, max)
            cell_eccentricity_range: Range of cell eccentricity (min, max)
            cell_intensity_range: Range of cell intensity (min, max)
            background_intensity: Background intensity level
            noise_level: Amount of noise to add
            wavelength_params: Optional dictionary of parameters for each wavelength
                Example: {
                    1: {
                        'num_cells': 100,
                        'cell_size_range': (10, 30),
                        'cell_eccentricity_range': (0.1, 0.5),
                        'cell_intensity_range': (5000, 20000),
                        'background_intensity': 500
                    },
                    2: {
                        'num_cells': 50,
                        'cell_size_range': (5, 15),
                        'cell_eccentricity_range': (0.3, 0.8),
                        'cell_intensity_range': (3000, 12000),
                        'background_intensity': 300
                    }
                }
            shared_cell_fraction: Fraction of cells shared between wavelengths (0.0-1.0)
                0.0 means all cells are unique to each wavelength
                1.0 means all cells are shared between wavelengths
                Default is 0.95 (95% shared)
            wavelength_intensities: Dictionary mapping wavelength indices to fixed intensities
                Example: {1: 20000, 2: 10000}
            wavelength_backgrounds: Dictionary mapping wavelength indices to background intensities
                Example: {1: 800, 2: 400}
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.grid_size = grid_size
        self.image_size = image_size
        self.tile_size = tile_size
        self.overlap_percent = overlap_percent
        self.stage_error_px = stage_error_px
        self.wavelengths = wavelengths
        self.z_stack_levels = z_stack_levels
        self.z_step_size = z_step_size
        self.num_cells = num_cells
        self.cell_size_range = cell_size_range
        self.cell_eccentricity_range = cell_eccentricity_range
        self.cell_intensity_range = cell_intensity_range
        self.background_intensity = background_intensity
        self.noise_level = noise_level
        self.wavelength_params = wavelength_params or {}
        self.shared_cell_fraction = shared_cell_fraction

        # Set default wavelength intensities if not provided
        if wavelength_intensities is None:
            self.wavelength_intensities = {1: 20000, 2: 10000}
            # Add defaults for additional wavelengths if needed
            for w in range(3, wavelengths + 1):
                self.wavelength_intensities[w] = 15000
        else:
            self.wavelength_intensities = wavelength_intensities

        # Set default wavelength backgrounds if not provided
        if wavelength_backgrounds is None:
            self.wavelength_backgrounds = {1: 800, 2: 400}
            # Add defaults for additional wavelengths if needed
            for w in range(3, wavelengths + 1):
                self.wavelength_backgrounds[w] = 600
        else:
            self.wavelength_backgrounds = wavelength_backgrounds

        # Store the wells to generate
        self.wells = wells

        # Store the base random seed
        self.base_random_seed = random_seed

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Create output directory structure
        self.timepoint_dir = self.output_dir / "TimePoint_1"
        self.timepoint_dir.mkdir(parents=True, exist_ok=True)

        # Calculate effective step size with overlap
        self.step_x = int(tile_size[0] * (1 - overlap_percent / 100))
        self.step_y = int(tile_size[1] * (1 - overlap_percent / 100))

        # First, generate a common set of base cells for all wavelengths
        # This ensures consistent patterns across wavelengths for better registration
        base_cells = []
        max_num_cells = max([
            self.wavelength_params.get(w+1, {}).get('num_cells', self.num_cells)
            for w in range(wavelengths)
        ])

        # Generate shared cell positions and common attributes
        for i in range(max_num_cells):
            # Generate cell in a way that ensures good features in overlap regions
            # Bias cell positions to increase density in overlap regions for better registration

            # Calculate overlap regions (where tiles will overlap)
            overlap_x = int(tile_size[0] * (overlap_percent / 100))
            overlap_y = int(tile_size[1] * (overlap_percent / 100))

            # Very strongly favor overlap regions with 80% probability
            # to ensure very high density of features in the 10% overlap region for reliable registration
            if np.random.random() < 0.8:
                # Position in an overlap region between tiles
                col = np.random.randint(0, grid_size[1])
                row = np.random.randint(0, grid_size[0])

                # Calculate base tile position
                base_x = col * self.step_x
                base_y = row * self.step_y

                # Position cells in right/bottom overlapping regions
                if np.random.random() < 0.5:
                    # Right overlap region
                    x = base_x + tile_size[0] - overlap_x + np.random.randint(0, overlap_x)
                    y = base_y + np.random.randint(0, tile_size[1])
                else:
                    # Bottom overlap region
                    x = base_x + np.random.randint(0, tile_size[0])
                    y = base_y + tile_size[1] - overlap_y + np.random.randint(0, overlap_y)

                # Ensure we're within image bounds
                x = min(x, image_size[0] - 1)
                y = min(y, image_size[1] - 1)
            else:
                # Random position anywhere in the image
                x = np.random.randint(0, image_size[0])
                y = np.random.randint(0, image_size[1])

            # Common cell attributes
            size = np.random.uniform(*self.cell_size_range)
            eccentricity = np.random.uniform(*self.cell_eccentricity_range)
            rotation = np.random.uniform(0, 2*np.pi)

            base_cells.append({
                'x': x,
                'y': y,
                'size': size,
                'eccentricity': eccentricity,
                'rotation': rotation
            })

        # We'll generate cell parameters for each well and wavelength on demand
        # This is just a placeholder initialization
        self.cell_params = {}

        # Store wavelength-specific parameters for later use
        self.wavelength_specific_params = []
        for w in range(wavelengths):
            wavelength_idx = w + 1  # 1-based wavelength index

            # Get wavelength-specific parameters or use defaults
            w_params = self.wavelength_params.get(wavelength_idx, {})
            w_num_cells = w_params.get('num_cells', self.num_cells)
            w_cell_size_range = w_params.get('cell_size_range', self.cell_size_range)
            w_cell_intensity_range = w_params.get('cell_intensity_range', self.cell_intensity_range)

            self.wavelength_specific_params.append({
                'wavelength_idx': wavelength_idx,
                'num_cells': w_num_cells,
                'cell_size_range': w_cell_size_range,
                'cell_intensity_range': w_cell_intensity_range
            })

            # We'll generate cells on demand in generate_cell_image

    def generate_cell_image(self, wavelength, z_level, well=None):
        """
        Generate a full image with cells for a specific wavelength and Z level.

        Args:
            wavelength: Wavelength channel index
            z_level: Z-stack level index
            well: Well identifier (e.g., 'A01')

        Returns:
            Full image with cells
        """
        # Generate a unique key for this well and wavelength
        key = f"{well}_{wavelength}" if well else f"default_{wavelength}"

        # Get wavelength-specific parameters
        wavelength_idx = wavelength + 1  # Convert to 1-based index for params lookup
        w_params = self.wavelength_params.get(wavelength_idx, {})

        # Generate cells for this well and wavelength if not already generated
        if key not in self.cell_params:
            # Get parameters for cell generation
            w_num_cells = w_params.get('num_cells', self.num_cells)
            w_cell_size_range = w_params.get('cell_size_range', self.cell_size_range)
            w_cell_intensity_range = w_params.get('cell_intensity_range', self.cell_intensity_range)

            # Generate cells for this wavelength
            cells = []
            for i in range(w_num_cells):
                # Generate random position for this wavelength
                x = np.random.randint(0, self.image_size[0])
                y = np.random.randint(0, self.image_size[1])

                # Generate random cell properties
                size = np.random.uniform(w_cell_size_range[0], w_cell_size_range[1])
                eccentricity = np.random.uniform(self.cell_eccentricity_range[0], self.cell_eccentricity_range[1])
                rotation = np.random.uniform(0, 2*np.pi)

                # Set very different intensities for each wavelength to make them easily distinguishable
                if wavelength_idx == 1:
                    # First wavelength: very high intensity
                    intensity = 25000
                elif wavelength_idx == 2:
                    # Second wavelength: medium intensity
                    intensity = 10000
                else:
                    # Other wavelengths: lower intensity
                    intensity = 5000 + (wavelength_idx * 1000)  # Increase slightly for each additional wavelength

                cells.append({
                    'x': x,
                    'y': y,
                    'size': size,
                    'eccentricity': eccentricity,
                    'rotation': rotation,
                    'intensity': intensity
                })

            # Store cells for this well and wavelength
            self.cell_params[key] = cells

        # Get cells for this well and wavelength
        cells = self.cell_params[key]

        # Get background intensity from wavelength_backgrounds or use default
        w_background = self.wavelength_backgrounds.get(wavelength_idx, self.background_intensity)

        # Create empty image with wavelength-specific background intensity
        # Ensure image is 2D (not 3D) to avoid shape mismatch in ashlar
        image = np.ones(self.image_size, dtype=np.uint16) * w_background

        # Get cell parameters for this well and wavelength
        cells = self.cell_params[key]

        # Calculate Z-focus factor (1.0 at center Z, decreasing toward edges)
        if self.z_stack_levels > 1:
            z_center = (self.z_stack_levels - 1) / 2
            z_distance = abs(z_level - z_center)
            z_factor = 1.0 - (z_distance / z_center) if z_center > 0 else 1.0
        else:
            z_factor = 1.0

        # Draw each cell
        for cell in cells:
            # Adjust intensity based on Z level (cells are brightest at focus)
            intensity = cell['intensity'] * z_factor

            # Calculate ellipse parameters
            a = cell['size']
            b = cell['size'] * (1 - cell['eccentricity'])

            # Generate ellipse coordinates
            rr, cc = draw.ellipse(
                cell['y'], cell['x'],
                b, a,
                rotation=cell['rotation'],
                shape=self.image_size
            )

            # Add cell to image
            image[rr, cc] = intensity

        # Add noise
        # Use wavelength-specific noise level if provided
        w_noise_level = w_params.get('noise_level', self.noise_level)
        noise = np.random.normal(0, w_noise_level, self.image_size)
        image = image + noise

        # Apply blur based on Z distance from focus
        if self.z_stack_levels > 1:
            # More blur for Z levels further from center
            # Scale blur by z_step_size to create more realistic Z-stack effect
            # z_step_size controls the amount of blur between Z-steps
            # Reduce blur by at least 4-fold
            blur_sigma = (self.z_step_size / 4.0) * (1.0 + 2.0 * (1.0 - z_factor))
            print(f"  Z-level {z_level}: blur_sigma={blur_sigma:.2f} (z_factor={z_factor:.2f}, z_step_size={self.z_step_size})")
            image = filters.gaussian(image, sigma=blur_sigma, preserve_range=True)

        # Ensure valid pixel values
        image = np.clip(image, 0, 65535).astype(np.uint16)

        return image

    # We've replaced the generate_tiles method with position pre-generation in generate_dataset

    def generate_htd_file(self):
        """Generate HTD file with metadata in the format expected by ezstitcher."""
        # Derive plate name from output directory name
        plate_name = self.output_dir.name
        htd_filename = f"{plate_name}.HTD"

        # Generate the main HTD file in the plate dir
        htd_path = self.output_dir / htd_filename

        # Basic HTD file content matching the format of real HTD files
        htd_content = f""""HTSInfoFile", Version 1.0
"Description", "Synthetic microscopy data for testing"
"PlateType", 6
"TimePoints", 1
"ZSeries", {"TRUE" if self.z_stack_levels > 1 else "FALSE"}
"ZSteps", {self.z_stack_levels}
"ZProjection", FALSE
"XWells", 4
"YWells", 3"""

        # Add wells selection (only the wells we're using are TRUE)
        for y in range(3):  # 3 rows (A, B, C)
            row_wells = []
            for x in range(4):  # 4 columns (1, 2, 3, 4)
                well = f"{chr(65+y)}{x+1:02d}"  # A01, A02, etc.
                row_wells.append("TRUE" if well in self.wells else "FALSE")
            htd_content += f"\n\"WellsSelection{y+1}\", {', '.join(row_wells)}"

        # Add sites information
        htd_content += f"\n\"Sites\", TRUE"
        htd_content += f"\n\"XSites\", {self.grid_size[1]}"
        htd_content += f"\n\"YSites\", {self.grid_size[0]}"

        # Add site selection rows (all set to FALSE except the ones we're using)
        for y in range(self.grid_size[0]):
            row = []
            for x in range(self.grid_size[1]):
                row.append("TRUE")  # All sites are used in our synthetic data
            htd_content += f"\n\"SiteSelection{y+1}\", {', '.join(row)}"

        # Add wavelength information
        htd_content += f"\n\"Waves\", TRUE"
        htd_content += f"\n\"NWavelengths\", {self.wavelengths}"

        # Add wavelength names and collection flags
        for w in range(self.wavelengths):
            htd_content += f"\n\"WaveName{w+1}\", \"W{w+1}\""
            htd_content += f"\n\"WaveCollect{w+1}\", 1"

        # Add unique identifier and end file marker
        htd_content += f"\n\"UniquePlateIdentifier\", \"{plate_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}\""
        htd_content += "\n\"EndFile\""

        # Write HTD file in plate root directory
        with open(htd_path, 'w') as f:
            f.write(htd_content)

        # Also create a copy in the TimePoint directory
        timepoint_htd_path = self.timepoint_dir / htd_filename
        with open(timepoint_htd_path, 'w') as f:
            f.write(htd_content)

        return htd_path

    def generate_dataset(self):
        """Generate the complete dataset."""
        print(f"Generating synthetic microscopy dataset in {self.output_dir}")
        print(f"Grid size: {self.grid_size[0]}x{self.grid_size[1]}")
        print(f"Wavelengths: {self.wavelengths}")
        print(f"Z-stack levels: {self.z_stack_levels}")
        print(f"Wells: {', '.join(self.wells)}")

        # Generate HTD file
        htd_path = self.generate_htd_file()
        print(f"Generated HTD file: {htd_path}")

        # Process each well
        for well_index, well in enumerate(self.wells):
            print(f"\nGenerating data for well {well}...")

            # Use a different random seed for each well if base seed is provided
            if self.base_random_seed is not None:
                well_seed = self.base_random_seed + well_index
                np.random.seed(well_seed)
                random.seed(well_seed)
                print(f"Using random seed {well_seed} for well {well}")

            # Pre-generate the positions for each site to ensure consistency across Z-levels
            # This creates a mapping of site_index -> (base_x_pos, base_y_pos)
            site_positions = {}
            site_index = 1
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    # Calculate base position
                    x = col * self.step_x
                    y = row * self.step_y

                    # Add random stage positioning error
                    # We apply this error to the base position, it will be constant across Z-steps
                    x_error = np.random.randint(-self.stage_error_px, self.stage_error_px)
                    y_error = np.random.randint(-self.stage_error_px, self.stage_error_px)

                    x_pos = x + x_error
                    y_pos = y + y_error

                    # Ensure we don't go out of bounds
                    x_pos = max(0, min(x_pos, self.image_size[0] - self.tile_size[0]))
                    y_pos = max(0, min(y_pos, self.image_size[1] - self.tile_size[1]))

                    site_positions[site_index] = (x_pos, y_pos)
                    site_index += 1

            # For multiple Z-stack levels, create proper ZStep folders
            if self.z_stack_levels > 1:
                # Make sure all ZStep folders are created first
                for z in range(self.z_stack_levels):
                    z_level = z + 1  # 1-based Z level index
                    zstep_dir = self.timepoint_dir / f"ZStep_{z_level}"
                    zstep_dir.mkdir(exist_ok=True)
                    print(f"Created ZStep folder: {zstep_dir}")

                # Now generate images for each Z-level
                for z in range(self.z_stack_levels):
                    z_level = z + 1  # 1-based Z level index
                    zstep_dir = self.timepoint_dir / f"ZStep_{z_level}"

                    # Generate images for each wavelength at this Z level
                    for w in range(self.wavelengths):
                        wavelength = w + 1  # 1-based wavelength index

                        # Generate full image
                        print(f"Generating full image for wavelength {wavelength}, Z level {z_level}...")
                        full_image = self.generate_cell_image(w, z, well=well)

                        # Save tiles for this Z level using the pre-generated positions
                        site_index = 1
                        for row in range(self.grid_size[0]):
                            for col in range(self.grid_size[1]):
                                # Get the pre-generated position
                                x_pos, y_pos = site_positions[site_index]

                                # Extract tile
                                tile = full_image[
                                    y_pos:y_pos + self.tile_size[1],
                                    x_pos:x_pos + self.tile_size[0]
                                ]

                                # Create filename without Z-index and without zero-padding site indices
                                # This tests the padding functionality in the stitcher
                                filename = f"{well}_s{site_index}_w{wavelength}.tif"
                                filepath = zstep_dir / filename

                                # Save image without compression
                                tifffile.imwrite(filepath, tile, compression=None)

                                # Print progress with full path for debugging
                                print(f"  Saved tile: {zstep_dir.name}/{filename} (position: {x_pos}, {y_pos})")
                                print(f"  Full path: {filepath.resolve()}")
                                site_index += 1
            else:
                # For single Z level (no Z-stack), just save files directly in TimePoint_1
                for w in range(self.wavelengths):
                    wavelength = w + 1  # 1-based wavelength index

                    # Generate full image for the single Z level
                    print(f"Generating full image for wavelength {wavelength} (no Z-stack)...")
                    full_image = self.generate_cell_image(w, 0, well=well)

                    # Save tiles without Z-stack index
                    site_index = 1
                    for row in range(self.grid_size[0]):
                        for col in range(self.grid_size[1]):
                            # Get the pre-generated position
                            x_pos, y_pos = site_positions[site_index]

                            # Extract tile
                            tile = full_image[
                                y_pos:y_pos + self.tile_size[1],
                                x_pos:x_pos + self.tile_size[0]
                            ]

                            # Create filename without Z-index and without zero-padding site indices
                            filename = f"{well}_s{site_index}_w{wavelength}.tif"
                            filepath = self.timepoint_dir / filename

                            # Save image without compression
                            tifffile.imwrite(filepath, tile, compression=None)

                            # Print progress
                            print(f"  Saved tile: {filename} (position: {x_pos}, {y_pos})")
                            site_index += 1

        print("Dataset generation complete!")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic microscopy images for testing")

    parser.add_argument("output_dir", help="Directory to save generated images")
    parser.add_argument("--grid-size", type=int, nargs=2, default=[3, 3], help="Grid size (rows cols)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[2048, 2048], help="Full image size")
    parser.add_argument("--tile-size", type=int, nargs=2, default=[1024, 1024], help="Tile size")
    parser.add_argument("--overlap", type=float, default=10.0, help="Percentage of overlap between tiles")
    parser.add_argument("--stage-error", type=int, default=5, help="Random error in stage positioning (pixels)")
    parser.add_argument("--wavelengths", type=int, default=2, help="Number of wavelength channels")
    parser.add_argument("--z-stack", type=int, default=1, help="Number of Z-stack levels")
    parser.add_argument("--z-step-size", type=float, default=1.0, help="Spacing between Z-steps in microns")
    parser.add_argument("--num-cells", type=int, default=300, help="Number of cells to generate (higher density improves registration)")
    parser.add_argument("--cell-size", type=float, nargs=2, default=[10, 30], help="Cell size range (min max)")
    parser.add_argument("--cell-eccentricity", type=float, nargs=2, default=[0.1, 0.5],
                        help="Cell eccentricity range (min max)")
    parser.add_argument("--cell-intensity", type=int, nargs=2, default=[8000, 30000],
                        help="Cell intensity range (min max) - higher values create more contrast for better registration")
    parser.add_argument("--background", type=int, default=500, help="Background intensity")
    parser.add_argument("--noise", type=int, default=100, help="Noise level")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # Add wavelength-specific parameter groups
    wavelength_group = parser.add_argument_group('wavelength-specific',
                                                'Wavelength-specific parameters (overrides global parameters)')
    wavelength_group.add_argument("--w1-cells", type=int,
                                 help="Number of cells for wavelength 1")
    wavelength_group.add_argument("--w1-size", type=float, nargs=2,
                                 help="Cell size range for wavelength 1 (min max)")
    wavelength_group.add_argument("--w1-eccentricity", type=float, nargs=2,
                                 help="Cell eccentricity range for wavelength 1 (min max)")
    wavelength_group.add_argument("--w1-intensity", type=int, nargs=2,
                                 help="Cell intensity range for wavelength 1 (min max)")
    wavelength_group.add_argument("--w1-background", type=int,
                                 help="Background intensity for wavelength 1")
    wavelength_group.add_argument("--w1-noise", type=int,
                                 help="Noise level for wavelength 1")

    wavelength_group.add_argument("--w2-cells", type=int,
                                 help="Number of cells for wavelength 2")
    wavelength_group.add_argument("--w2-size", type=float, nargs=2,
                                 help="Cell size range for wavelength 2 (min max)")
    wavelength_group.add_argument("--w2-eccentricity", type=float, nargs=2,
                                 help="Cell eccentricity range for wavelength 2 (min max)")
    wavelength_group.add_argument("--w2-intensity", type=int, nargs=2,
                                 help="Cell intensity range for wavelength 2 (min max)")
    wavelength_group.add_argument("--w2-background", type=int,
                                 help="Background intensity for wavelength 2")
    wavelength_group.add_argument("--w2-noise", type=int,
                                 help="Noise level for wavelength 2")

    args = parser.parse_args()

    # Build wavelength-specific parameters
    wavelength_params = {}

    # Wavelength 1 parameters
    w1_params = {}
    if args.w1_cells is not None:
        w1_params['num_cells'] = args.w1_cells
    if args.w1_size is not None:
        w1_params['cell_size_range'] = tuple(args.w1_size)
    if args.w1_eccentricity is not None:
        w1_params['cell_eccentricity_range'] = tuple(args.w1_eccentricity)
    if args.w1_intensity is not None:
        w1_params['cell_intensity_range'] = tuple(args.w1_intensity)
    if args.w1_background is not None:
        w1_params['background_intensity'] = args.w1_background
    if args.w1_noise is not None:
        w1_params['noise_level'] = args.w1_noise

    if w1_params:
        wavelength_params[1] = w1_params

    # Wavelength 2 parameters
    w2_params = {}
    if args.w2_cells is not None:
        w2_params['num_cells'] = args.w2_cells
    if args.w2_size is not None:
        w2_params['cell_size_range'] = tuple(args.w2_size)
    if args.w2_eccentricity is not None:
        w2_params['cell_eccentricity_range'] = tuple(args.w2_eccentricity)
    if args.w2_intensity is not None:
        w2_params['cell_intensity_range'] = tuple(args.w2_intensity)
    if args.w2_background is not None:
        w2_params['background_intensity'] = args.w2_background
    if args.w2_noise is not None:
        w2_params['noise_level'] = args.w2_noise

    if w2_params:
        wavelength_params[2] = w2_params

    # Create generator
    generator = SyntheticMicroscopyGenerator(
        output_dir=args.output_dir,
        grid_size=tuple(args.grid_size),
        image_size=tuple(args.image_size),
        tile_size=tuple(args.tile_size),
        overlap_percent=args.overlap,
        stage_error_px=args.stage_error,
        wavelengths=args.wavelengths,
        z_stack_levels=args.z_stack,
        z_step_size=args.z_step_size,
        num_cells=args.num_cells,
        cell_size_range=tuple(args.cell_size),
        cell_eccentricity_range=tuple(args.cell_eccentricity),
        cell_intensity_range=tuple(args.cell_intensity),
        background_intensity=args.background,
        noise_level=args.noise,
        wavelength_params=wavelength_params,
        random_seed=args.seed
    )

    # Generate dataset
    generator.generate_dataset()


if __name__ == "__main__":
    main()
