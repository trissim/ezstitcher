import os
import re
import numpy as np
import pandas as pd
import imageio
import tifffile
import sys
import traceback
import shutil
from pathlib import Path

# Internal imports
from ezstitcher.core.z_stack_handler import (
    organize_zstack_folders, preprocess_plate_folder,
    select_best_focus_zstack, create_zstack_projections
)
from ezstitcher.core.image_process import (
    assemble_image_subpixel, tophat, create_weighted_composite, process_bf
)

# Skimage - only the necessary import
from skimage import io

# Ashlar
from ashlar import fileseries, reg

############################
# FILE AND PATH HANDLING
############################

def parse_filename(filename):
    """
    Extract the well, site, wavelength, and Z-step info from a filename, rewriting if needed.
    Examples:
      - ABC12_s001_w2.TIF -> well=ABC12, site=001, wavelength=2, z_step=None
      - D05_w1.TIF       -> well=D05, site=None, wavelength=1, z_step=None
      - ABC12_s001_w2_z003.TIF -> well=ABC12, site=001, wavelength=2, z_step=003
    """
    match1 = re.search(r'([A-Z]\d+)_s(\d+)\.*', filename)
    match2 = re.search(r'([A-Z]\d+)_s(\d+)_w(\d).*', filename)
    match3 = re.search(r'([A-Z]\d+)_w(\d).*', filename)

    # Pattern for Z-step - matches 1-3 digits
    z_match = re.search(r'_z(\d{1,3})', filename)
    z_step = z_match.group(1) if z_match else None

    if match3:
        # e.g. D05_w1
        well = match3.group(1)
        wavelength = match3.group(2)
        return well, None, wavelength, z_step, filename

    if match2:
        # e.g. D05_s002_w3
        well = match2.group(1)
        site = match2.group(2)
        wavelength = match2.group(3)
        if wavelength == '0':
            new_filename = filename.replace("_w0", "_w1")
            os.rename(filename, new_filename)
            filename = new_filename
        return well, site, wavelength, z_step, filename

    if match1:
        # e.g. D05_s001   -> rename to w1
        well = match1.group(1)
        site = match1.group(2)
        wavelength = '1'
        new_filename = (
            filename[:match1.span()[1]-1]
            + "_w" + wavelength
            + filename[match1.span()[1]-1:]
        )
        os.rename(filename, new_filename)
        filename = new_filename
        return well, site, wavelength, z_step, filename

    # If no match, just return None
    return None, None, None, None, filename

def get_pattern_string(pattern_entry):
    """
    Extract the pattern string from a potentially nested structure.
    This handles both string patterns and dictionary objects from auto_detect_patterns.

    Args:
        pattern_entry: Either a string pattern or a dict with "pattern" key

    Returns:
        str: The extracted pattern string
    """
    # Check if pattern is a dict with "pattern" key (from auto_detect_patterns)
    if isinstance(pattern_entry, dict) and "pattern" in pattern_entry:
        return pattern_entry["pattern"]
    # Otherwise assume it's a string pattern directly
    return pattern_entry

def path_list_from_pattern(image_dir, image_pattern, z_step=None):
    """
    Match files in image_dir using patterns with placeholders.
    Supports multiple pattern styles:
    1. {iii} placeholder for site numbers (replaced by \d{3})
    2. {zzz} placeholder for z-step (replaced by specific z-step or \d{3})
    3. Glob-style patterns with * wildcards (converted to regex)

    Args:
        image_dir: Directory to search in
        image_pattern: Pattern with {iii} for site, {zzz} for z-step, or * wildcards
        z_step: If provided, match only files with this specific z-step

    Returns:
        list: Sorted list of matching filenames
    """
    # Handle dictionary patterns from auto_detect_patterns
    image_pattern = get_pattern_string(image_pattern)
    # Handle substitution of {series} if present (from Ashlar)
    if "{series}" in image_pattern:
        print(f"WARNING: path_list_from_pattern detected {{series}} in pattern: {image_pattern}")
        print(f"Converting {{series}} to {{iii}} for consistency")
        image_pattern = image_pattern.replace("{series}", "{iii}")

    # Check if this is a glob-style pattern with * wildcards
    if "*" in image_pattern:
        # Convert glob pattern to regex pattern
        # Escape special regex characters except for *
        special_chars = [".", "^", "$", "+", "?", "(", ")", "[", "]", "{", "}", "|", "\\"]
        file_pattern = image_pattern
        for char in special_chars:
            file_pattern = file_pattern.replace(char, f"\\{char}")
        # Convert * to regex equivalent
        file_pattern = file_pattern.replace("*", ".*")
    else:
        # Handle {iii} placeholder style
        file_pattern = image_pattern.replace("{iii}", r"\d{3}")

        if "{zzz}" in file_pattern:
            if z_step is not None:
                # Replace {zzz} with the specific z_step
                file_pattern = file_pattern.replace("{zzz}", f"{int(z_step):03d}")
            else:
                # Replace {zzz} with any 3 digits
                file_pattern = file_pattern.replace("{zzz}", r"\d{3}")

    print(f"path_list_from_pattern: Using regex pattern: '{file_pattern}' to match files in {image_dir}")

    pattern = re.compile(f'^{file_pattern}$', re.IGNORECASE)
    matches = [f for f in os.listdir(image_dir) if pattern.match(f)]
    print(f"path_list_from_pattern: Found {len(matches)} matching files")
    if len(matches) > 0:
        print(f"  First few matches: {matches[:min(5, len(matches))]}")

    return sorted(matches)

def compute_stitched_name(file_pattern):
    """
    Remove the 's{iii}_' or 's{iii}' portion from the pattern,
    returning the rest as the final stitched filename.

    Examples:
      pattern = "mfd-ctb_A05_s{iii}_w1.tif" -> "mfd-ctb_A05_w1.tif"
      pattern = "mfd-ctb_B06_s{iii}w1.tif"  -> "mfd-ctb_B06_w1.tif"
    """
    # Handle dictionary patterns from auto_detect_patterns
    file_pattern = get_pattern_string(file_pattern)

    file_pattern = re.sub(r"\{.*?\}", f"{{{'iii'}}}", file_pattern)
    if "s{iii}_" in file_pattern:
        stitched_name = file_pattern.replace("s{iii}_", "")
    else:
        stitched_name = file_pattern.replace("s{iii}", "")
    return stitched_name

def clean_filename(filepath):
    """Renames a file from e.g. D05_s2_w1.TIF to a standard format with zero-padded site."""
    if os.path.isfile(filepath):
        # First check for site pattern directly to handle cases parse_filename might miss
        # Use more comprehensive regex to match any site pattern
        site_match = re.search(r'_s(\d{1,3})(?=_|\.)', os.path.basename(filepath))
        if site_match:
            site_num = site_match.group(1)
            # Only pad if not already 3 digits
            if len(site_num) < 3:
                padded = site_num.zfill(3)  # e.g. "002"
                # Make the replacement
                old_part = f"_s{site_num}"
                new_part = f"_s{padded}"
                new_path = filepath.replace(old_part, new_part)
                os.rename(filepath, new_path)
                filepath = new_path  # Update path for further processing

        # Also check for z pattern
        z_match = re.search(r'_z(\d{1,3})(?=_|\.)', os.path.basename(filepath))
        if z_match:
            z_num = z_match.group(1)
            # Only pad if not already 3 digits
            if len(z_num) < 3:
                padded = z_num.zfill(3)  # e.g. "002"
                # Make the replacement
                old_part = f"_z{z_num}"
                new_part = f"_z{padded}"
                new_path = filepath.replace(old_part, new_part)
                os.rename(filepath, new_path)
                filepath = new_path  # Update path for further processing

        # Now run the original logic to catch any other issues
        well, site, wavelength, z_step, newpath = parse_filename(filepath)
        if site is not None:
            padded = site.zfill(3)  # e.g. "002"
            new_filename = newpath.replace("_s"+site, "_s"+padded)

            # Remove extra junk between site/wavelength
            if wavelength is not None:
                wave_idx = new_filename.index("_w"+wavelength)
                ext = new_filename.split(".")[-1]
                new_filename = new_filename[:wave_idx] + f"_w{wavelength}." + ext

            # Only rename if the filename actually changed
            if new_filename != filepath:
                os.rename(filepath, new_filename)

def clean_folder(folder):
    """Cleans up the filenames in a folder, removing thumbs, rewriting site numbers, etc."""
    df = folder_to_df(folder)
    for fp in df["filepath"]:
        if "thumb" in fp:
            os.remove(fp)
        else:
            clean_filename(fp)
    return folder_to_df(folder)

def remove_not_tif(folder_path):
    """Removes .HTD files from a folder."""
    for fname in os.listdir(folder_path):
        if fname.endswith(".HTD"):
            os.remove(os.path.join(folder_path, fname))

############################
# DATAFRAME OPERATIONS
############################

def add_filepath_to_df(filepath, df):
    """Adds filepath (with parsed well/site/wavelength/z_step) to a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    if os.path.isfile(filepath) and filepath.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tif', 'gif')):
        well, site, wavelength, z_step, filepath = parse_filename(filepath)
        row = pd.DataFrame({
            "filepath": filepath,
            "well": well,
            "site": site,
            "wavelength": [wavelength],
            "z_step": [z_step]
        }, index=[len(df)])
        df = pd.concat([df, row], axis=0)

    return df

def filepaths_to_df(filepaths):
    """Convert a list of filepaths to a DataFrame with well/site/wavelength columns."""
    df = None
    for fp in filepaths:
        df = add_filepath_to_df(fp, df)
    return df

def folder_to_df(folder_path):
    """Get a DataFrame describing all TIFF-like files in a folder."""
    fps = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)]
    return filepaths_to_df(fps)

def unique_wells_wavelengths(filepaths):
    """Find unique wells and wavelengths in a given folder or file list."""
    if isinstance(filepaths, str):
        df = folder_to_df(filepaths)
    elif isinstance(filepaths, list):
        df = filepaths_to_df(filepaths)
    else:
        return None, None

    wells = df['well'].unique()
    wavelengths = df['wavelength'].unique()
    return wells, wavelengths

############################
# PATTERN DETECTION
############################

def auto_detect_patterns(folder, placeholder="{iii}", z_placeholder="{zzz}"):
    """
    Auto-detect image patterns organized by well, wavelength, and z-step.

    Returns:
      dict: {
        well: {
          wavelength: {
            "pattern": pattern_string,
            "has_z": boolean
          }
        }
      }
      Example:
        {
          "E04": {
            "1": {
              "pattern": "mfd-ctb-test_E04_s{iii}_w1.TIF",
              "has_z": False
            },
            "2": {
              "pattern": "mfd-ctb-test_E04_s{iii}_w2_z{zzz}.TIF",
              "has_z": True
            }
          }
        }
    """
    df = folder_to_df(folder)
    patterns_by_well = {}

    # Check if Z-steps exist in the dataset
    has_zsteps = 'z_step' in df.columns and df['z_step'].notna().any()

    # Loop over each well
    for well in df["well"].unique():
        subdf_well = df[df["well"] == well]
        if subdf_well.empty:
            continue

        # Initialize dictionary for this well
        patterns_by_well[well] = {}

        # For each wavelength in that well
        for wave in subdf_well["wavelength"].unique():
            subdf_ww = subdf_well[subdf_well["wavelength"] == wave]
            if subdf_ww.empty:
                continue

            # Check if this wavelength has Z-stack data
            wave_has_z = False
            if has_zsteps:
                wave_has_z = subdf_ww['z_step'].notna().any()

            # We'll pick the first tile for that well/wave to build the pattern
            first_fp = subdf_ww["filepath"].iloc[0]
            base_name = os.path.basename(first_fp)

            # Parse the file to extract site, etc.
            w_extracted, site, w_detected, z_step, _ = parse_filename(base_name)
            # If we can't parse a well from the name, skip
            if not w_extracted:
                continue

            # Build the pattern
            pattern_candidate = base_name

            # Replace numeric site (e.g. "s003") with "s{iii}"
            if site:
                old_snippet = f"s{site}"
                new_snippet = f"s{placeholder}"
                pattern_candidate = pattern_candidate.replace(old_snippet, new_snippet)

            # Replace Z-step if present (e.g. "_z001") with "_z{zzz}"
            if z_step and wave_has_z:
                old_z_snippet = f"_z{z_step}"
                new_z_snippet = f"_z{z_placeholder}"
                pattern_candidate = pattern_candidate.replace(old_z_snippet, new_z_snippet)

            # Store in the dictionary keyed by wavelength
            patterns_by_well[well][wave] = {
                "pattern": pattern_candidate,
                "has_z": wave_has_z
            }

    return patterns_by_well

def generate_composite_reference_pattern(well, wavelength_patterns):
    """
    Generate a pattern name for a composite reference channel based on existing patterns.

    Args:
        well: Well identifier
        wavelength_patterns: Dict of wavelength to image pattern

    Returns:
        str: Composite reference pattern name
    """
    # Use the first pattern as a template
    first_pattern = next(iter(wavelength_patterns.values()))
    # Get the pattern string if it's nested
    template_pattern = get_pattern_string(first_pattern)

    base_name = compute_stitched_name(template_pattern)
    return f"composite_{well}_s{{iii}}_{base_name}"

############################
# GRID AND LAYOUT DETECTION
############################

def find_dimensions_selection(file_path, selection_name="SiteSelection"):
    """
    Reads a CSV-like HTD file, looks for lines with 'SiteSelection',
    returns (height, width) or something similar.
    """
    matrix = []
    with open(file_path, 'r', errors='ignore') as f:
        next(f, None)  # possibly skip header
        for line in f:
            if selection_name in line:
                row_data = line.strip().split(',')
                # parse the line, ignoring first token if needed
                row_bools = [
                    1 if val.strip().lower() == 'true' else 0
                    for val in row_data[1:]
                ]
                matrix.append(row_bools)

    # minimal approach: compute max rectangle or just count
    # the largest dimension. For now, assume a rectangular selection.
    # Each row in 'matrix' is a row of booleans (1/0).
    # let's guess #rows = tile_y, #columns = tile_x for a quick approach
    height = len(matrix)
    width = len(matrix[0]) if matrix else 0
    return (height, width)

def find_HTD_file(folder):
    """
    Looks for an .HTD file in 'folder', reads tile_x,tile_y from it,
    returns a dict or (tile_x, tile_y).
    """
    for fn in os.listdir(folder):
        if fn.endswith(".HTD"):
            path = os.path.join(folder, fn)
            tile_y, tile_x = find_dimensions_selection(path, "SiteSelection")
            return (tile_x, tile_y)
    return (None, None)

############################
# METADATA EXTRACTION
############################

def get_pixel_size_from_tiff(image_path):
    """
    Extract pixel size from TIFF metadata.

    Looks for spatial-calibration-x in the ImageDescription tag.

    Args:
        image_path: Path to a TIFF image

    Returns:
        float: Pixel size in microns (default 1.0 if not found)
    """
    try:
        # Read TIFF tags
        with tifffile.TiffFile(image_path) as tif:
            # Try to get ImageDescription tag
            if tif.pages[0].tags.get('ImageDescription'):
                desc = tif.pages[0].tags['ImageDescription'].value
                # Look for spatial calibration using regex
                match = re.search(r'id="spatial-calibration-x"[^>]*value="([0-9.]+)"', desc)

                if match:
                    print(f"Found pixel size metadata {str(float(match.group(1)))} in {image_path}")
                    return float(match.group(1))

                # Alternative pattern for some formats
                match = re.search(r'Spatial Calibration: ([0-9.]+) [uµ]m', desc)
                if match:

                    print(f"Found pixel size metadata {str(float(match.group(1)))} in {image_path}")
                    return float(match.group(1))

        print(f"Could not find pixel size metadata in {image_path}, using default")
    except Exception as e:
        print(f"Error reading metadata from {image_path}: {e}")

    # Default value if metadata not found
    return 1.0

############################
# IMAGE PROCESSING
############################

def process_imgs_from_pattern(image_dir, image_pattern, function, out_dir):
    """Process all images matching a pattern with the given function."""
    image_names = path_list_from_pattern(image_dir, image_pattern)
    # Use tifffile directly to avoid imagecodecs dependency
    images = np.array([tifffile.imread(os.path.join(image_dir, image)) for image in image_names])
    processed = function(images)
    for img, name in zip(processed, image_names):
        # Use tifffile directly to avoid imagecodecs dependency
        tifffile.imwrite(os.path.join(out_dir, name), img, compression=None)

def create_composite_reference_files(image_dir, processed_dir, channel_files,
                                     composite_pattern, channel_weights=None,
                                     preprocessing_funcs=None):
    """
    Create composite reference files by combining multiple channels.

    Args:
        image_dir: Base directory containing input images
        processed_dir: Directory to save processed and composite files
        channel_files: Dict mapping channel names to lists of filenames
        composite_pattern: Pattern for naming composite files
        channel_weights: Dict mapping channel names to weights
        preprocessing_funcs: Dict mapping channel names to preprocessing functions

    Returns:
        str: Composite reference pattern
    """
    if preprocessing_funcs is None:
        preprocessing_funcs = {}

    # Use helper function to verify each channel has the same number of files
    file_counts = [len(files) for files in channel_files.values()]
    if len(set(file_counts)) != 1:
        raise ValueError(f"Channels have different file counts: {file_counts}")

    # Get number of sites from first channel
    first_channel = next(iter(channel_files))
    num_files = len(channel_files[first_channel])

    # Process each position/tile
    for i in range(num_files):
        # Load and preprocess images for this position
        position_images = {}

        for channel, files in channel_files.items():
            # Load image from original directory
            img_path = os.path.join(image_dir, files[i])
            # Use tifffile directly to avoid imagecodecs dependency
            import tifffile
            img = tifffile.imread(img_path)

            # Apply preprocessing if specified for this channel
            if channel in preprocessing_funcs and preprocessing_funcs[channel] is not None:
                print(f"Preprocessing channel {channel} for composite")
                # Check if processed file exists
                processed_file = os.path.join(processed_dir, files[i])
                if os.path.exists(processed_file):
                    # Use already processed file
                    img = imageio.imread(processed_file)
                else:
                    # Process on the fly
                    img = preprocessing_funcs[channel]([img])[0]

            position_images[channel] = img

        # Create composite using the image processing function
        composite = create_weighted_composite(position_images, channel_weights)

        # Extract site number from filename using existing pattern
        site_num = None
        filename = channel_files[first_channel][i]
        match = re.search(r's(\d+)', filename)
        if match:
            site_num = match.group(1)
        else:
            site_num = f"{i:03d}"

        # Generate output filename with the correct site number
        out_filename = composite_pattern.replace("{iii}", site_num)
        out_path = os.path.join(processed_dir, out_filename)

        # Save composite reference to processed directory
        # Use tifffile directly to avoid imagecodecs dependency
        tifffile.imwrite(out_path, composite, compression=None)

    return composite_pattern

############################
# STITCHING OPERATIONS
############################

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

def ashlar_stitch_v2(image_dir, image_pattern, positions_path,
                    grid_size_x, grid_size_y, tile_overlap=10,
                    tile_overlap_x=None, tile_overlap_y=None,
                    max_shift=20, pixel_size=None):
    """
    Stitches images in 'image_dir' matching 'image_pattern' using the Ashlar library
    (FileSeriesReader, EdgeAligner, Mosaic, PyramidWriter) in a single cycle.

    Generates a CSV file with tile positions.

    Args:
        image_dir (str): Directory containing images (tiles).
        image_pattern (str): A pattern with '{iii}' or similar numeric placeholder
        positions_path (str): Path to save the positions CSV.
        grid_size_x (int): Number of tiles horizontally.
        grid_size_y (int): Number of tiles vertically.
        tile_overlap (float): Overlap percentage (default 10%).
        tile_overlap_x (float): Horizontal overlap percentage (defaults to tile_overlap)
        tile_overlap_y (float): Vertical overlap percentage (defaults to tile_overlap)
        max_shift (int): Maximum allowed error in microns
        pixel_size (float): Size of pixel in microns (auto-detected if None)
    """
    # Set individual overlaps if not provided
    if tile_overlap_x is None:
        tile_overlap_x = tile_overlap
    if tile_overlap_y is None:
        tile_overlap_y = tile_overlap

    # Convert tile overlap from % to fractional overlap
    overlap_fraction = tile_overlap / 100.0

    # Replace "{iii}" with "*" for a simple glob search, e.g. "A05_s*_w1.TIF"
    pattern_glob = image_pattern.replace("{iii}", "*")
    image_dir_path = Path(image_dir)

    # Collect all matching files in sorted order
    all_files = sorted(image_dir_path.glob(pattern_glob))
    if not all_files:
        raise ValueError(f"No files found in {image_dir} matching pattern {image_pattern}")

    print("ashlar_stitch_v2: Found", len(all_files), "files to stitch")
    for f in all_files[:5]:
        print("  ", f)  # Show a few

    # Get pixel size from first image if not provided
    if pixel_size is None:
        first_image_path = os.path.join(image_dir, all_files[0])
        pixel_size = get_pixel_size_from_tiff(first_image_path)
        print(f"Auto-detected pixel size: {pixel_size} microns")

    # Store the original pattern for later use in generate_positions_df
    original_pattern = image_pattern
    # Replace {iii} with {series} for Ashlar
    image_pattern = image_pattern.replace("{iii}", "{series}")

    # Create a single-cycle FileSeriesReader from these files
    fs_reader = fileseries.FileSeriesReader(
        path=os.path.abspath(image_dir),
        pattern=image_pattern,
        overlap=overlap_fraction,  # Using single overlap value for now
        width=grid_size_x,
        height=grid_size_y,
        layout="raster",
        direction="horizontal",
        pixel_size=pixel_size,
    )

    # Align the tiles using EdgeAligner on the first (and only) cycle
    aligner = reg.EdgeAligner(
        fs_reader,
        channel=0,          # If multi-channel, pick the channel to align on
        filter_sigma=0,     # adjust if needed
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
    positions_df = generate_positions_df(image_dir, original_pattern, positions, grid_size_x, grid_size_y)
    positions_df.to_csv(positions_path, index=False, sep=";", header=False)

    print(f"Finished writing CSV to {positions_path}")

############################
# HIGH-LEVEL OPERATIONS
############################

def setup_directories(plate_folder):
    """Create all necessary output directories for the stitching process.
    Also handles Z-stack structure if present and standardizes filenames.

    The directory structure is:
    - input_plate/
      - TimePoint_1/
    - input_plate_stitched/
      - TimePoint_1/
    - input_plate_processed/
      - TimePoint_1/
    - input_plate_positions/
      - TimePoint_1/

    All output directories are at the same level as the input plate folder.
    """
    base_dir = Path(plate_folder).resolve()
    parent_dir = base_dir.parent
    plate_name = base_dir.name

    # Define all directory paths at the same level as input plate
    dirs = {
        'stitched': parent_dir / f"{plate_name}_stitched" / "TimePoint_1",
        'processed': parent_dir / f"{plate_name}_processed" / "TimePoint_1",
        'positions': parent_dir / f"{plate_name}_positions" / "TimePoint_1",
        'input': base_dir / "TimePoint_1"
    }

    # Print info about our directory structure
    print(f"Setting up directory structure:")
    print(f"  Input plate: {base_dir}")
    print(f"  Parent directory: {parent_dir}")

    # Create the base output directories first
    parent_dirs = {
        'stitched': parent_dir / f"{plate_name}_stitched",
        'processed': parent_dir / f"{plate_name}_processed",
        'positions': parent_dir / f"{plate_name}_positions"
    }

    for name, dir_path in parent_dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Base {name} directory created: {dir_path}")

    # Check for Z-stacks BEFORE creating TimePoint subdirectories
    input_path = dirs['input']
    if input_path.exists():
        # First, clean and standardize all filenames in the input directory
        print(f"Standardizing filenames in {input_path}")
        clean_folder(input_path)

        # Then check for Z-stack structure
        zstep_pattern = re.compile(r'^ZStep_([0-9]+)$')
        has_zstack = any(zstep_pattern.match(item.name) for item in input_path.iterdir() if item.is_dir())

        if has_zstack:
            print(f"Z-stack structure detected in {input_path}")
            # Handle Z-stack organization first
            organize_zstack_folders(plate_folder)

            # Clean filenames in TimePoint_1 again after Z-stack reorganization
            print(f"Cleaning filenames after Z-stack organization")
            clean_folder(input_path)

    # Now create all the TimePoint_1 subdirectories
    for name, dir_path in dirs.items():
        if name != 'input':  # Input directory already exists
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created: {dir_path}")

    # Copy HTD files to all output directories
    htd_files = list(base_dir.glob("*.HTD"))
    if htd_files:
        for htd_file in htd_files:
            for name, parent_dir_path in parent_dirs.items():
                dest_path = parent_dir_path / htd_file.name
                if htd_file.resolve() != dest_path.resolve():
                    shutil.copy2(htd_file, dest_path)
                    print(f"Copied HTD file to {dest_path}")

    return dirs

    # Now create all the TimePoint_1 subdirectories
    for name, dir_path in dirs.items():
        if name != 'input':  # Input directory already exists
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created: {dir_path}")

    # Copy HTD files to all output directories
    htd_files = list(base_dir.glob("*.HTD"))
    if htd_files:
        for htd_file in htd_files:
            for name, parent_dir_path in parent_dirs.items():
                dest_path = parent_dir_path / htd_file.name
                if htd_file.resolve() != dest_path.resolve():
                    shutil.copy2(htd_file, dest_path)
                    print(f"Copied HTD file to {dest_path}")

    return dirs

def setup_plate_processing(plate_folder):
    """
    Set up plate processing: create directories, detect grid, find patterns.
    Raises ValueError if grid dimensions cannot be determined.
    """
    # Setup directories
    dirs = setup_directories(plate_folder)

    # Detect grid dimensions from HTD file
    grid_size_x, grid_size_y = find_HTD_file(plate_folder)
    if not grid_size_x or not grid_size_y:
        raise ValueError(f"Could not determine grid dimensions from {plate_folder}. No valid HTD file found.")

    print(f"Found grid dimensions from HTD: {grid_size_x}x{grid_size_y}")

    # Auto-detect patterns by well and wavelength
    patterns_by_well = auto_detect_patterns(dirs['input'], placeholder="{iii}")
    if not patterns_by_well:
        print(f"No patterns detected in {dirs['input']}")
        return None, None, None

    print(f"Auto-detected patterns for {len(patterns_by_well)} wells")

    return dirs, (grid_size_x, grid_size_y), patterns_by_well

def prepare_reference_channel(well, wavelength_patterns, dirs,
                           channels=None, preprocessing_funcs=None,
                           composite_weights=None):
    """
    Prepare a reference channel for image stitching by:
    1. Preprocessing all specified channels
    2. Creating a composite if multiple channels are provided
    3. Using a single channel if only one is specified

    Args:
        well: Well identifier
        wavelength_patterns: Dictionary mapping channel names to file patterns
        dirs: Dictionary of directories ('input', 'processed', etc.)
        channels: Channel(s) to use as reference - string for single channel or list for composite
        preprocessing_funcs: Dictionary mapping channels to preprocessing functions
        composite_weights: Optional weights for channels when creating a composite

    Returns:
        Tuple: (ref_channel, ref_pattern, ref_dir, updated_wavelength_patterns)
    """
    # Make a copy of wavelength_patterns to avoid modifying the original
    wavelength_patterns = wavelength_patterns.copy()

    # Ensure channels is a list and contains valid channels
    if channels is None:
        # Default to first available channel
        channels = [next(iter(wavelength_patterns.keys()))]
    elif isinstance(channels, str):
        # Single channel case
        channels = [channels]

    # Filter to valid channels only
    valid_channels = sorted(list(wavelength_patterns.keys()))
    if not valid_channels:
        print(f"No valid channels specified, using first available")
        valid_channels = [next(iter(wavelength_patterns.keys()))]

    # Determine create_composite based on number of channels
    create_composite = len(valid_channels) > 1

    # Preprocess all required channels
    if preprocessing_funcs:
        for channel in valid_channels:
            if channel in preprocessing_funcs and preprocessing_funcs[channel]:
                print(f"Pre-processing channel {channel}")
                # Get pattern string from potentially nested structure
                pattern_string = get_pattern_string(wavelength_patterns[channel])
                process_imgs_from_pattern(
                    dirs['input'],
                    pattern_string,
                    preprocessing_funcs[channel],
                    dirs['processed']
                )

    # Set reference directory - will be 'processed' if any preprocessing happened
    ref_dir = dirs['input']
    if preprocessing_funcs and any(ch in preprocessing_funcs and preprocessing_funcs[ch]
                                  for ch in valid_channels):
        ref_dir = dirs['processed']

    # Create composite if multiple channels specified
    if create_composite:
        print(f"Creating composite reference for well {well} from {valid_channels}")

        # Get file lists for each channel
        channel_files = {}
        for channel in valid_channels:
            pattern_string = get_pattern_string(wavelength_patterns[channel])
            channel_files[channel] = path_list_from_pattern(dirs['input'], pattern_string)

        # Generate a pattern for the new composite channel
        composite_pattern = generate_composite_reference_pattern(well,
                                                              {ch: wavelength_patterns[ch] for ch in valid_channels})

        # Create the composite reference files in the processed directory
        create_composite_reference_files(
            dirs['input'],
            dirs['processed'],
            channel_files,
            composite_pattern,
            composite_weights,
            preprocessing_funcs
        )

        # Add composite as a new "wavelength"
        wavelength_patterns['composite'] = composite_pattern
        reference_channel = 'composite'
        ref_pattern = composite_pattern
        ref_dir = dirs['processed']  # Always use processed directory for composites
    else:
        # Single channel case
        reference_channel = valid_channels[0]
        ref_pattern = get_pattern_string(wavelength_patterns[reference_channel])

    return reference_channel, ref_pattern, ref_dir, wavelength_patterns

def process_well_wavelengths(well, wavelength_patterns, dirs, grid_dims,
                           ref_channel, ref_pattern, ref_dir, margin_ratio=0.1,
                           tile_overlap=10, tile_overlap_x=None, tile_overlap_y=None,
                           max_shift=50):
    """
    Process all wavelengths for a well:
    1. Generate positions using the reference channel
    2. Assemble each wavelength using these positions
    """
    grid_size_x, grid_size_y = grid_dims

    # Generate positions using Ashlar
    stitched_name = compute_stitched_name(ref_pattern)
    positions_path = dirs['positions'] / f"{Path(stitched_name).stem}.csv"

    print(f"Generating positions using Ashlar with pattern: {ref_pattern}")
    print(f"Reading reference images from: {ref_dir}")

    # Use Ashlar to generate positions from the reference channel
    ashlar_stitch_v2(
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

    # Process and stitch all original wavelengths (not the composite)
    for wavelength, pattern in wavelength_patterns.items():
        # Skip the composite channel - don't try to assemble it
        if wavelength == 'composite':
            print(f"Skipping assembly of composite channel (used only for alignment)")
            continue

        # Use original image directory for assembly
        img_dir = dirs['input']

        # Get files for this wavelength to override the composite filenames
        override_names = path_list_from_pattern(img_dir, pattern)

        # Assemble final image
        stitched_name = compute_stitched_name(pattern)
        output_path = dirs['stitched'] / stitched_name

        print(f"Assembling wavelength {wavelength} from {img_dir} to {output_path}")
        assemble_image_subpixel(
            positions_path=positions_path,
            images_dir=img_dir,
            output_path=output_path,
            margin_ratio=margin_ratio,
            override_names=override_names  # Use the correct filenames for this wavelength
        )

def process_plate_folder(plate_folder, reference_channels=['1'],
                         preprocessing_funcs=None, margin_ratio=0.1,
                         composite_weights=None, well_filter=None,
                         tile_overlap=6.5, tile_overlap_x=None, tile_overlap_y=None,
                         max_shift=50, focus_detect=False, focus_method="combined",
                         create_projections=False, projection_types=['max', 'mean'],
                         stitch_z_reference='best_focus'):
    """
    Process an entire plate folder with microscopy images.

    This function handles all aspects of microscopy image processing including:
    - Z-stack detection and organization
    - Best focus selection for Z-stacks
    - Projection creation (max, mean, median) for Z-stacks
    - Composite image creation from multiple wavelengths
    - Stitching with ashlar

    Args:
        plate_folder: Base folder for the plate
        reference_channels: List of channels to use as reference for alignment
        preprocessing_funcs: Dict mapping wavelength/channel to preprocessing function
        margin_ratio: Blending margin ratio for stitching
        composite_weights: Dict mapping channels to weights for composite
        well_filter: Optional list of wells to process (if None, process all)
        tile_overlap: Percentage of overlap between tiles (default 6.5%)
        tile_overlap_x: Horizontal overlap percentage (defaults to tile_overlap)
        tile_overlap_y: Vertical overlap percentage (defaults to tile_overlap)
        max_shift: Maximum shift allowed between tiles in microns (default 50)
        focus_detect: Whether to enable focus detection for Z-stacks
        focus_method: Focus detection method to use
        create_projections: Whether to create Z-stack projections
        projection_types: Types of projections to create
        stitch_z_reference: Z-reference for stitching ('best_focus' or z-index)
    """
    try:
        # First preprocess to handle Z-stacks if present
        has_zstack, z_info = preprocess_plate_folder(plate_folder)

        # Get the parent directory and plate name for correct folder structure
        plate_path = Path(plate_folder)
        parent_dir = plate_path.parent
        plate_name = plate_path.name

        # Handle Z-stack specific processing if detected
        if has_zstack:
            print(f"Z-stack detected in {plate_folder}")

            # Handle focus detection if requested
            best_focus_dir = None
            if focus_detect:
                print(f"Performing best focus detection using {focus_method} method")
                success, best_focus_dir = select_best_focus_zstack(
                    plate_folder,
                    focus_wavelength=reference_channels[0],
                    focus_method=focus_method
                )
                if not success:
                    print(f"Warning: Best focus detection failed, using original images")

            # Handle projections if requested
            projections_dir = None
            if create_projections:
                print(f"Creating Z-stack projections: {', '.join(projection_types)}")
                success, projections_dir = create_zstack_projections(
                    plate_folder,
                    projection_types=projection_types
                )
                if not success:
                    print(f"Warning: Projection creation failed, using original images")

            # Determine which directory to use for stitching
            stitch_source = plate_folder
            if stitch_z_reference == 'best_focus' and best_focus_dir:
                stitch_source = best_focus_dir
                print(f"Using best focus images for stitching from {best_focus_dir}")
            elif stitch_z_reference in projection_types and projections_dir:
                # Use the specified projection type
                stitch_source = projections_dir
                print(f"Using {stitch_z_reference} projections for stitching from {projections_dir}")
        else:
            # No Z-stack detected, use original folder
            stitch_source = plate_folder
            print(f"No Z-stack detected in {plate_folder}, using standard stitching")

        # 1. Setup and detection
        try:
            dirs, grid_dims, patterns_by_well = setup_plate_processing(stitch_source)
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Skipping plate {stitch_source}")
            return

        if not patterns_by_well:
            return

        # Filter wells if requested
        if well_filter:
            patterns_by_well = {well: patterns for well, patterns in patterns_by_well.items()
                               if well in well_filter}
            if not patterns_by_well:
                print(f"None of the requested wells {well_filter} found")
                return

        # 2. Process each well
        for well, wavelength_patterns in patterns_by_well.items():
            print(f"\nProcessing well {well} with {len(wavelength_patterns)} wavelength(s)")

            # Prepare reference channel
            ref_channel, ref_pattern, ref_dir, updated_patterns = prepare_reference_channel(
                well, wavelength_patterns, dirs, reference_channels, preprocessing_funcs,
                composite_weights)

            # Process all wavelengths using the reference
            process_well_wavelengths(
                well, updated_patterns, dirs, grid_dims,
                ref_channel, ref_pattern, ref_dir,
                margin_ratio=margin_ratio,
                tile_overlap=tile_overlap,
                tile_overlap_x=tile_overlap_x,
                tile_overlap_y=tile_overlap_y,
                max_shift=max_shift
            )

            print(f"Completed processing well {well}")

        print(f"\nFinished stitching all wells in plate {stitch_source}")

    except Exception as err:
        import traceback
        traceback.print_exc()
        print(f"Failed to process plate folder {plate_folder}: {err}")


############################
# MAIN EXECUTION
############################


if __name__ == "__main__":
    def dapi_process(imgs):
        """Apply tophat filter to DAPI images."""
        return [tophat(img) for img in imgs]

    # Plate folders to process
    plate_folders = [
        # Example paths - replace with actual paths when running
        '/path/to/your/plate/folder'
    ]

    if isinstance(plate_folders, str):
        plate_folders = [plate_folders]

    # Process each plate with Z-stack support
    for folder in plate_folders:
        process_plate_folder(
            folder,
            reference_channels=["1", "2"],
            composite_weights={"1": 0.1, "2": 0.9},
            preprocessing_funcs={"1": process_bf},
            tile_overlap=10,
            max_shift=50,
            focus_detect=True,
            focus_method="combined",
            create_projections=True,
            projection_types=["max", "mean"],
            stitch_z_reference="best_focus"
        )
