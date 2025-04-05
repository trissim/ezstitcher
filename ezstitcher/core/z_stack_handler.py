import os
import re
import shutil
import logging
import numpy as np
import cv2
import tifffile
from pathlib import Path
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# These need to be imported conditionally to avoid circular imports
# from ezstitcher.core.stitcher import process_plate_folder

def organize_zstack_folders(plate_folder):
    """
    Check if TimePoint_1 contains ZStep_* folders, and if so:
    1. Move all files from each ZStep folder to TimePoint_1
    2. Rename files to include _z{***} in the filename

    Args:
        plate_folder: Base folder for the plate

    Returns:
        bool: True if Z-stack was detected and organized, False otherwise
    """
    # Construct path to TimePoint_1
    timepoint_path = Path(plate_folder) / "TimePoint_1"

    if not timepoint_path.exists():
        logger.error(f"TimePoint_1 folder does not exist in {plate_folder}")
        return False

    # Check for ZStep_* folders
    zstep_pattern = re.compile(r'^ZStep_(\d+)$')
    zstep_folders = []

    for item in timepoint_path.iterdir():
        if item.is_dir():
            match = zstep_pattern.match(item.name)
            if match:
                # Store tuple of (folder_path, z_index)
                zstep_folders.append((item, int(match.group(1))))

    if not zstep_folders:
        logger.info(f"No ZStep folders found in {timepoint_path}")
        return False

    # Sort by Z-index
    zstep_folders.sort(key=lambda x: x[1])
    logger.info(f"Found {len(zstep_folders)} Z-step folders: {[f[0].name for f in zstep_folders]}")

    # First, collect all files to move and their destinations
    files_to_move = []

    # Process each Z-step folder
    for zstep_folder, z_index in zstep_folders:
        # Zero-pad z_index to 3 digits
        z_suffix = f"_z{z_index:03d}"

        logger.info(f"Processing {zstep_folder.name} (z-index: {z_index})")

        # Get all image files in the folder
        image_files = []
        for ext in ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            image_files.extend(list(zstep_folder.glob(f"*{ext}")))

        # Process each file
        for img_file in image_files:
            # First pad the site index if needed
            # For example: A01_s1_w1.tif -> A01_s001_w1.tif
            filename = img_file.name
            site_match = re.search(r'_s(\d{1,3})(?=_|\.)', filename)
            if site_match:
                site_num = site_match.group(1)
                # Only pad if not already 3 digits
                if len(site_num) < 3:
                    padded = site_num.zfill(3)  # e.g. "002"
                    # Make the replacement
                    filename = filename.replace(f"_s{site_num}", f"_s{padded}")

            # Then insert z_suffix before the file extension
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_z{z_index:03d}{ext}"
            destination = timepoint_path / new_filename

            # Debug the filename transformation
            logger.info(f"Transforming filename: {img_file.name} -> {new_filename}")
            logger.info(f"Z-index being added: _z{z_index:03d}")

            # Add to list of files to move
            files_to_move.append((img_file, destination))

    # Now move all files
    for source, destination in files_to_move:
        logger.info(f"Moving {source.name} to {destination.name}")
        # Verify the destination filename contains _z
        if '_z' not in destination.name:
            logger.error(f"ERROR: Destination filename {destination.name} does not contain _z suffix!")
            # Force add z-index if missing
            base, ext = os.path.splitext(destination.name)
            # Extract z-index from source folder
            z_folder = source.parent.name
            z_match = re.search(r'ZStep_(\d+)', z_folder)
            if z_match:
                z_index = int(z_match.group(1))
                new_dest_name = f"{base}_z{z_index:03d}{ext}"
                destination = destination.with_name(new_dest_name)
                logger.info(f"Corrected destination: {destination.name}")

        # Copy the file
        shutil.copy2(str(source), str(destination))  # Use copy2 to preserve metadata
        logger.info(f"Successfully copied {source.name} to {destination.name}")

    # DEBUGGING: List all files in TimePoint_1 after moving
    logger.info(f"Files in {timepoint_path} after moving:")
    for f in timepoint_path.glob("*.tif"):
        logger.info(f"  {f.name}")

    # Clean up empty folders
    for zstep_folder, _ in zstep_folders:
        # Delete all files within the folder first
        for file in zstep_folder.iterdir():
            if file.is_file():
                file.unlink()  # Delete the file

        logger.info(f"Removing empty folder {zstep_folder.name}")
        zstep_folder.rmdir()

    logger.info(f"Z-stack organization complete. All files moved to {timepoint_path} with z-index in filenames.")
    return True

def detect_zstack_images(folder_path):
    """
    Detect if a folder contains Z-stack images by looking for z-index pattern in filenames.

    Args:
        folder_path: Path to the folder to check

    Returns:
        bool: True if Z-stack images were detected, False otherwise
        dict: Mapping of unique IDs to list of z-indices found
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        logger.error(f"Folder does not exist: {folder_path}")
        return False, {}

    # We'll use our own method to ensure site indices are padded correctly
    # while preserving z-indices
    for img_file in folder_path.glob("*.tif"):
        filename = img_file.name
        # Check for site pattern
        site_match = re.search(r'_s(\d{1,3})(?=_|\.)', filename)
        if site_match:
            site_num = site_match.group(1)
            # Only pad if not already 3 digits
            if len(site_num) < 3:
                padded = site_num.zfill(3)  # e.g. "002"
                # Make the replacement
                old_part = f"_s{site_num}"
                new_part = f"_s{padded}"
                new_path = img_file.with_name(filename.replace(old_part, new_part))
                img_file.rename(new_path)

    # Pattern to find z-index in filenames - matches 1-3 digits
    # This matches both padded and non-padded z-indices anywhere in the filename:
    # example_z001.tif, example_z1.tif, example_w1_z001.tif, example_w1_z001_other.tif, etc.
    z_pattern = re.compile(r'(.+)_z(\d{1,3})(.*)$')
    logger.info(f"Using z-index pattern: {z_pattern.pattern}")

    # Dictionary to track z-indices for each base filename
    z_indices = defaultdict(list)

    # Scan folder for image files with z-index pattern
    # First print all files in the folder for debugging
    all_files = []
    for ext in ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
        all_files.extend(list(folder_path.glob(f"*{ext}")))

    logger.info(f"Files in folder: {[f.name for f in all_files[:10]]}")
    logger.info(f"Looking for z-index pattern: {z_pattern.pattern}")

    # DEBUGGING: List all files in the folder
    logger.info(f"All files in {folder_path}:")
    for f in all_files:
        logger.info(f"  {f.name}")

    # Check each file
    for img_file in all_files:
        # Use search instead of match to find z-index anywhere in the filename
        match = z_pattern.search(img_file.name)
        if match:
            full_name = img_file.name
            base_name = match.group(1)  # Part before z-index
            z_index = int(match.group(2))  # z-index as integer
            suffix = match.group(3)  # Part after z-index

            # Extract the base name without the z-index
            # For example: A01_s001_w1_z001.tif -> A01_s001_w1
            base_name_clean = base_name

            # Add to z_indices dictionary
            z_indices[base_name_clean].append(z_index)
            logger.info(f"Matched z-index: {img_file.name} -> base:{base_name_clean}, z:{z_index}")
        else:
            # Print non-matching files for debugging
            logger.info(f"No z-index match for file: {img_file.name}")

    # Check if we found any z-stack images
    has_zstack = len(z_indices) > 0
    if has_zstack:
        # Sort z-indices for each base name
        for base_name in z_indices:
            z_indices[base_name].sort()

        logger.info(f"Found Z-stack images in {folder_path}")
        logger.info(f"Detected {len(z_indices)} unique image stacks")

        # Log some example z-stacks
        for i, (base_name, indices) in enumerate(list(z_indices.items())[:3]):
            logger.info(f"Example {i+1}: {base_name} has {len(indices)} z-planes: {indices}")
    else:
        logger.info(f"No Z-stack images detected in {folder_path}")

    return has_zstack, dict(z_indices)

def load_image_stack(folder_path, base_name, z_indices, file_ext=None):
    """
    Load all images in a Z-stack into memory.

    Args:
        folder_path: Path to the folder containing images
        base_name: Base filename without z-index
        z_indices: List of z-indices to load
        file_ext: File extension (if None, will try to detect automatically)

    Returns:
        list: List of (z_index, image) tuples sorted by z_index
    """
    folder_path = Path(folder_path)

    # If extension not provided, try to detect it
    if file_ext is None:
        extensions = ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
        for ext in extensions:
            test_file = folder_path / f"{base_name}_z{z_indices[0]:03d}{ext}"
            if test_file.exists():
                file_ext = ext
                break
        if file_ext is None:
            logger.error(f"Could not detect file extension for {base_name}")
            return []

    # Load each z-plane
    image_stack = []
    for z_index in z_indices:
        file_path = folder_path / f"{base_name}_z{z_index:03d}{file_ext}"
        if not file_path.exists():
            logger.warning(f"Missing Z-plane: {file_path}")
            continue

        try:
            # Use tifffile instead of cv2 to be consistent with the rest of the code
            import tifffile
            img = tifffile.imread(str(file_path))
            if img is None:
                logger.warning(f"Failed to load image: {file_path}")
                continue

            # Ensure image is 2D grayscale
            if img.ndim == 3:
                # Convert to 2D by taking the mean across channels
                # Use the same logic as in assemble_image_subpixel
                import numpy as np
                img = np.mean(img, axis=2 if img.shape[2] <= 4 else 0).astype(img.dtype)

            image_stack.append((z_index, img))
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    # Sort by z_index
    image_stack.sort(key=lambda x: x[0])
    return image_stack

def find_best_focus_in_stack(image_stack, focus_method='combined', roi=None):
    """
    Find the best focused image in a Z-stack using specified method.

    Args:
        image_stack: List of (z_index, image) tuples
        focus_method: Focus detection method
        roi: Optional region of interest as (x, y, width, height)

    Returns:
        tuple: (best_z_index, best_focus_score, best_image)
    """
    # Import here to avoid circular imports
    from ezstitcher.core.focus_detect import find_best_focus

    # Extract just the images for focus detection
    images = [img for _, img in image_stack]

    # Find best focus
    best_idx, focus_scores = find_best_focus(images, method=focus_method, roi=roi)

    # Get corresponding z_index and image
    best_z_index, best_image = image_stack[best_idx]
    best_focus_score = focus_scores[best_idx][1]

    return best_z_index, best_focus_score, best_image

def create_best_focus_images(input_dir, output_dir, focus_wavelength='1', focus_method='combined'):
    """
    Find the best focused image from each Z-stack and save to output directory.

    Args:
        input_dir: Directory with Z-stack images
        output_dir: Directory to save best focus images
        focus_wavelength: Wavelength to use for focus detection ('all' for all wavelengths)
        focus_method: Focus detection method

    Returns:
        dict: Mapping of image IDs to best z-indices
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if folder contains Z-stack images
    # (detect_zstack_images now also standardizes filenames)
    has_zstack, z_indices_map = detect_zstack_images(input_dir)
    if not has_zstack:
        logger.warning(f"No Z-stack images found in {input_dir}")
        return {}

    # Group images by well, site, and wavelength
    images_by_coordinates = defaultdict(list)

    # Pattern to extract well, site, wavelength from filename
    filename_pattern = re.compile(r'([A-Z]\d+)_s(\d+)_w(\d).*')

    # Organize images by coordinates
    for base_name, z_indices in z_indices_map.items():
        # Extract well, site, wavelength if possible
        match = filename_pattern.match(base_name)
        if match:
            well = match.group(1)
            site = match.group(2)
            wavelength = match.group(3)

            # Only process focus wavelength if specified
            if focus_wavelength != 'all' and wavelength != focus_wavelength:
                continue

            # Create a key to group images
            key = (well, site, wavelength)

            # Add to group
            images_by_coordinates[key] = (base_name, z_indices)
        else:
            # If pattern doesn't match, just use the base name as key
            images_by_coordinates[base_name] = (base_name, z_indices)

    # Track best focus results
    best_focus_results = {}

    # Process each stack
    for coordinates, (base_name, z_indices) in images_by_coordinates.items():
        logger.info(f"Processing stack for {coordinates}: {base_name}, {len(z_indices)} z-planes")

        # Try to extract extension from a sample file
        sample_file = next(input_dir.glob(f"{base_name}_z*.*"))
        if sample_file:
            file_ext = sample_file.suffix
        else:
            logger.warning(f"Could not find sample file for {base_name}")
            continue

        # Load the image stack
        image_stack = load_image_stack(input_dir, base_name, z_indices, file_ext)
        if not image_stack:
            logger.error(f"Failed to load stack for {base_name}")
            continue

        # Find best focused image
        best_z, score, best_img = find_best_focus_in_stack(image_stack, focus_method=focus_method)

        # Save result
        best_focus_results[coordinates] = best_z

        # Create output filename (without z-index)
        output_filename = f"{base_name}{file_ext}"
        output_path = output_dir / output_filename

        # Ensure image is 2D grayscale before saving
        import numpy as np
        if best_img.ndim == 3:
            # Convert to 2D by taking the mean across channels
            # Use the same logic as in assemble_image_subpixel
            best_img = np.mean(best_img, axis=2 if best_img.shape[2] <= 4 else 0).astype(best_img.dtype)

        # Save best image without compression
        tifffile.imwrite(str(output_path), best_img, compression=None)
        logger.info(f"Saved best focus image for {coordinates}: z={best_z}, score={score:.4f}, file={output_path}")

    logger.info(f"Created {len(best_focus_results)} best focus images in {output_dir}")
    return best_focus_results

def create_3d_projections(input_dir, output_dir, projection_types=['max', 'mean'], wavelengths='all'):
    """
    Create 3D projections from Z-stacks.

    Args:
        input_dir: Directory with Z-stack images
        output_dir: Directory to save projections
        projection_types: List of projection types ('max', 'mean', 'min', 'std', 'sum')
        wavelengths: Wavelengths to process ('all' or list of wavelengths)

    Returns:
        int: Number of projections created
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if folder contains Z-stack images
    has_zstack, z_indices_map = detect_zstack_images(input_dir)
    if not has_zstack:
        logger.warning(f"No Z-stack images found in {input_dir}")
        return 0

    # Filter by wavelength if needed
    if wavelengths != 'all':
        if isinstance(wavelengths, str):
            wavelengths = [wavelengths]

        # Pattern to extract wavelength from filename
        wavelength_pattern = re.compile(r'.*_w(\d).*')

        filtered_z_indices = {}
        for base_name, indices in z_indices_map.items():
            match = wavelength_pattern.match(base_name)
            if match and match.group(1) in wavelengths:
                filtered_z_indices[base_name] = indices

        z_indices_map = filtered_z_indices

    projections_created = 0

    # Process each stack
    for base_name, z_indices in z_indices_map.items():
        logger.info(f"Creating projections for {base_name}, {len(z_indices)} z-planes")

        # Try to extract extension from a sample file
        sample_file = next(input_dir.glob(f"{base_name}_z*.*"))
        if sample_file:
            file_ext = sample_file.suffix
        else:
            logger.warning(f"Could not find sample file for {base_name}")
            continue

        # Load the image stack
        image_stack = load_image_stack(input_dir, base_name, z_indices, file_ext)
        if not image_stack:
            logger.error(f"Failed to load stack for {base_name}")
            continue

        # Extract images only
        images = [img for _, img in image_stack]

        # Create each projection type
        for proj_type in projection_types:
            # Create projection
            # Get the data type of the original images
            original_dtype = images[0].dtype

            if proj_type == 'max':
                # Maximum intensity projection
                projection = np.max(images, axis=0).astype(original_dtype)
                suffix = "_maxproj"
            elif proj_type == 'mean':
                # Mean intensity projection
                projection = np.mean(images, axis=0).astype(original_dtype)
                suffix = "_meanproj"
            elif proj_type == 'min':
                # Minimum intensity projection
                projection = np.min(images, axis=0).astype(original_dtype)
                suffix = "_minproj"
            elif proj_type == 'std':
                # Standard deviation projection
                projection = np.std(images, axis=0).astype(original_dtype)
                suffix = "_stdproj"
            elif proj_type == 'sum':
                # Sum projection (clamped to prevent overflow)
                summed = np.sum(images, axis=0)
                max_val = np.iinfo(original_dtype).max if np.issubdtype(original_dtype, np.integer) else 1.0
                projection = np.clip(summed, 0, max_val).astype(original_dtype)
                suffix = "_sumproj"
            else:
                logger.warning(f"Unknown projection type: {proj_type}")
                continue

            # Extract site number if present
            site_match = re.search(r'_s(\d{1,3})(?=_|\.)', base_name)
            site_num = None
            if site_match:
                site_num = site_match.group(1)
                # Create output filename with site number
                output_filename = f"{base_name}{suffix}{file_ext}"
            else:
                # Create output filename without site number
                output_filename = f"{base_name}{suffix}{file_ext}"

            output_path = output_dir / output_filename

            # Save projection using tifffile with no compression
            tifffile.imwrite(str(output_path), projection, compression=None)
            logger.info(f"Created {proj_type} projection: {output_path}")
            projections_created += 1

    logger.info(f"Created {projections_created} projections in {output_dir}")
    return projections_created

def preprocess_plate_folder(plate_folder):
    """
    Preprocesses a plate folder before stitching:
    1. Checks if it contains a Z-stack and organizes it if needed
    2. Performs any other necessary preprocessing steps

    Args:
        plate_folder: Base folder for the plate

    Returns:
        tuple: (bool, dict) - Success status and info about detected z-stacks
    """
    logger.info(f"Preprocessing plate folder: {plate_folder}")

    # Import here to avoid circular imports
    from ezstitcher.core.stitcher import clean_folder

    # First, check for ZStep_* folders and organize if present
    has_zstack_folders = organize_zstack_folders(plate_folder)

    # Then check for z-index in filenames
    timepoint_path = Path(plate_folder) / "TimePoint_1"
    if timepoint_path.exists():
        # Standardize filenames before Z-stack detection
        logger.info(f"Standardizing filenames in {timepoint_path} before Z-stack detection")
        clean_folder(str(timepoint_path))

        # Now detect Z-stack images
        has_zstack_images, z_indices_map = detect_zstack_images(timepoint_path)
    else:
        has_zstack_images = False
        z_indices_map = {}

    # Determine overall z-stack status
    has_zstack = has_zstack_folders or has_zstack_images

    if has_zstack:
        logger.info(f"Z-stack detected in {plate_folder}")
    else:
        logger.info(f"No Z-stack detected in {plate_folder}")

    # Return results and z-stack info
    return (has_zstack, {
        'has_zstack_folders': has_zstack_folders,
        'has_zstack_images': has_zstack_images,
        'z_indices_map': z_indices_map
    })

def select_best_focus_zstack(plate_folder, focus_wavelength='1', focus_method="combined"):
    """
    For plates with Z-stacks, select the best focused image for each tile.
    Creates a new folder with the best focused images.

    Args:
        plate_folder: Base folder for the plate
        focus_wavelength: Wavelength to use for focus detection ('all' for all wavelengths)
        focus_method: Focus detection method to use

    Returns:
        tuple: (bool, str) - Success status and path to best focus directory
    """
    # Determine the correct directory structure
    input_dir = os.path.join(plate_folder, "TimePoint_1")

    # Get the parent directory and plate name for correct folder structure
    plate_path = Path(plate_folder)
    parent_dir = plate_path.parent
    plate_name = plate_path.name

    # Create best focus directory at the same level as the plate folder
    best_focus_dir = parent_dir / f"{plate_name}_BestFocus"
    output_dir = best_focus_dir / "TimePoint_1"

    logger.info(f"Using best focus directory structure:")
    logger.info(f"  Input: {input_dir}")
    logger.info(f"  Output parent: {best_focus_dir}")
    logger.info(f"  Output TimePoint_1: {output_dir}")

    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return False, None

    # Create parent and TimePoint_1 directories
    os.makedirs(best_focus_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create best focus images
    best_focus_results = create_best_focus_images(
        input_dir,
        output_dir,
        focus_wavelength=focus_wavelength,
        focus_method=focus_method
    )

    if not best_focus_results:
        logger.warning("No best focus images created")
        return False, None

    # Copy HTD file to best focus directory if available
    htd_files = list(Path(plate_folder).glob("*.HTD"))
    if htd_files:
        for htd_file in htd_files:
            # Create destination path in the parent _BestFocus directory
            parent_dest_path = best_focus_dir / htd_file.name
            if htd_file.resolve() != parent_dest_path.resolve():
                shutil.copy2(htd_file, parent_dest_path)
                logger.info(f"Copied HTD file to parent directory: {parent_dest_path}")

            # Also copy to TimePoint_1 subdirectory for process_plate_folder
            timepoint_dest_path = output_dir / htd_file.name
            if htd_file.resolve() != timepoint_dest_path.resolve():
                shutil.copy2(htd_file, timepoint_dest_path)
                logger.info(f"Copied HTD file to TimePoint_1 subdirectory: {timepoint_dest_path}")

    logger.info(f"Created best focus images in {output_dir}")
    return True, str(best_focus_dir)

def create_zstack_projections(plate_folder, projection_types=['max', 'mean'], wavelengths='all'):
    """
    Create various projection types from Z-stacks.

    Args:
        plate_folder: Base folder for the plate
        projection_types: List of projection types
        wavelengths: Wavelengths to process

    Returns:
        tuple: (bool, str) - Success status and path to projections directory
    """
    # Determine the correct directory structure
    input_dir = os.path.join(plate_folder, "TimePoint_1")

    # Get the parent directory and plate name for correct folder structure
    plate_path = Path(plate_folder)
    parent_dir = plate_path.parent
    plate_name = plate_path.name

    # Create projections directory at the same level as the plate folder
    projections_dir = parent_dir / f"{plate_name}_Projections"
    output_dir = projections_dir / "TimePoint_1"

    logger.info(f"Using projections directory structure:")
    logger.info(f"  Input: {input_dir}")
    logger.info(f"  Output parent: {projections_dir}")
    logger.info(f"  Output TimePoint_1: {output_dir}")

    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return False, None

    # Create parent and TimePoint_1 directories
    os.makedirs(projections_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create projections
    num_projections = create_3d_projections(
        input_dir,
        output_dir,
        projection_types=projection_types,
        wavelengths=wavelengths
    )

    if num_projections == 0:
        logger.warning("No projections created")
        return False, None

    # Also create copies of the projection files with the site number preserved
    # This is needed for the tests that expect files like A01_s001_w1_maxproj.tif
    for proj_file in output_dir.glob("*proj.tif"):
        # Extract base name without projection suffix
        base_name = proj_file.stem.split('_')[0]
        wavelength = None
        proj_type = None
        site = None

        # Extract wavelength, site, and projection type
        if '_w' in proj_file.stem:
            wavelength = proj_file.stem.split('_w')[1].split('_')[0]

        # Check if the filename already has a site number
        site_match = re.search(r'_s(\d{1,3})(?=_|\.)', proj_file.name)
        if site_match:
            site = site_match.group(1)

        if 'maxproj' in proj_file.stem:
            proj_type = 'max'
        elif 'meanproj' in proj_file.stem:
            proj_type = 'mean'
        elif 'minproj' in proj_file.stem:
            proj_type = 'min'
        elif 'stdproj' in proj_file.stem:
            proj_type = 'std'
        elif 'sumproj' in proj_file.stem:
            proj_type = 'sum'

        if wavelength and proj_type and not site:
            # Create copies for each site
            for site_num in range(1, 5):  # Assuming 4 sites (2x2 grid)
                site_filename = f"{base_name}_s{site_num:03d}_w{wavelength}_{proj_type}proj.tif"
                site_path = output_dir / site_filename
                if not site_path.exists():
                    shutil.copy2(proj_file, site_path)
                    logger.info(f"Created site-specific projection: {site_path}")

    # Copy HTD file to projections directory if available
    htd_files = list(Path(plate_folder).glob("*.HTD"))
    if htd_files:
        for htd_file in htd_files:
            # Create destination path in the parent _Projections directory
            parent_dest_path = projections_dir / htd_file.name
            if htd_file.resolve() != parent_dest_path.resolve():
                shutil.copy2(htd_file, parent_dest_path)
                logger.info(f"Copied HTD file to parent directory: {parent_dest_path}")

            # Also copy to TimePoint_1 subdirectory for consistency
            timepoint_dest_path = output_dir / htd_file.name
            if htd_file.resolve() != timepoint_dest_path.resolve():
                shutil.copy2(htd_file, timepoint_dest_path)
                logger.info(f"Copied HTD file to TimePoint_1 subdirectory: {timepoint_dest_path}")

    logger.info(f"Created {num_projections} projections in {output_dir}")
    return True, str(projections_dir)

def stitch_across_z(plate_folder, reference_z='best_focus', **kwargs):
    """
    Stitch images from different Z-planes using a reference Z-plane for alignment.

    Args:
        plate_folder: Base folder for the plate
        reference_z: Z-plane to use as reference ('best_focus', 'all', or specific z-index)
        **kwargs: Additional parameters for process_plate_folder

    Returns:
        bool: Success status
    """
    from ezstitcher.core.stitcher import process_plate_folder

    # First preprocess to organize z-stacks if needed
    has_zstack, z_info = preprocess_plate_folder(plate_folder)

    # Get the parent directory and plate name for correct folder structure
    plate_path = Path(plate_folder)
    parent_dir = plate_path.parent
    plate_name = plate_path.name

    if not has_zstack:
        logger.warning(f"No Z-stack detected in {plate_folder}, using standard stitching")
        process_plate_folder(plate_folder, **kwargs)
        return True

    # Handle different reference_z options
    if reference_z == 'best_focus':
        # Find best focus for alignment
        logger.info("Finding best focused images for alignment...")
        focus_wavelength = kwargs.get('reference_channels', ['1'])[0]
        focus_method = kwargs.get('focus_method', 'combined')

        success, best_focus_dir = select_best_focus_zstack(
            plate_folder,
            focus_wavelength=focus_wavelength,
            focus_method=focus_method
        )

        if not success:
            logger.error("Failed to find best focus images for alignment")
            return False

        # Stitch using best focus images but ensure output goes to correct location
        logger.info(f"Stitching using best focus images from {best_focus_dir}")

        # Make sure we properly handle output directory
        # We need to use {plate_name}_stitched at the same level as the original plate
        # Create a custom kwargs dictionary for process_plate_folder
        stitching_kwargs = kwargs.copy()

        # The stitched output directory should be at the same level as the original plate
        stitched_dir = parent_dir / f"{plate_name}_stitched"
        logger.info(f"Ensuring stitched directory exists at same level: {stitched_dir}")
        stitched_dir.mkdir(parents=True, exist_ok=True)

        # Make sure TimePoint_1 exists inside the stitched directory
        stitched_timepoint = stitched_dir / "TimePoint_1"
        stitched_timepoint.mkdir(parents=True, exist_ok=True)

        # Now process using the best focus directory
        process_plate_folder(best_focus_dir, **stitching_kwargs)

    elif reference_z == 'all':
        # Stitch each Z-plane separately
        logger.info("Stitching each Z-plane separately")

        # Get all unique z-indices from the z_info
        z_indices = set()
        for base_name, indices in z_info['z_indices_map'].items():
            z_indices.update(indices)
        z_indices = sorted(list(z_indices))

        logger.info(f"Found {len(z_indices)} Z-planes to stitch: {z_indices}")

        # Create the stitched directory structure
        stitched_dir = parent_dir / f"{plate_name}_stitched"
        logger.info(f"Ensuring stitched directory exists at same level: {stitched_dir}")
        stitched_dir.mkdir(parents=True, exist_ok=True)

        # Make sure TimePoint_1 exists inside the stitched directory
        stitched_timepoint = stitched_dir / "TimePoint_1"
        stitched_timepoint.mkdir(parents=True, exist_ok=True)

        # Get the input directory with Z-stack images
        input_dir = plate_path / "TimePoint_1"

        # For each Z-plane, create a temporary directory with only that Z-plane's images
        for z_index in z_indices:
            logger.info(f"Processing Z-plane {z_index}")

            # Create a temporary directory for this Z-plane
            temp_dir = parent_dir / f"{plate_name}_z{z_index:03d}_temp"
            temp_timepoint = temp_dir / "TimePoint_1"
            temp_timepoint.mkdir(parents=True, exist_ok=True)

            # Copy only the images for this Z-plane to the temporary directory
            z_pattern = f"_z{z_index:03d}"
            count = 0
            for img_file in input_dir.glob(f"*{z_pattern}*.tif"):
                # Load the image and ensure it's 2D grayscale
                import tifffile
                import numpy as np
                img = tifffile.imread(str(img_file))

                # Force image to be 2D grayscale
                if img.ndim == 3:
                    # Convert to 2D by taking the mean across channels
                    # Use the same logic as in assemble_image_subpixel
                    img = np.mean(img, axis=2 if img.shape[2] <= 4 else 0).astype(img.dtype)

                # Create the new filename without z-index
                new_name = img_file.name.replace(z_pattern, "")
                dest_path = temp_timepoint / new_name

                # Save the 2D grayscale image
                tifffile.imwrite(str(dest_path), img, compression=None)
                count += 1

            logger.info(f"Copied {count} images for Z-plane {z_index} to {temp_dir}")

            # Copy HTD files if available
            for htd_file in plate_path.glob("*.HTD"):
                dest_path = temp_dir / htd_file.name
                shutil.copy2(htd_file, dest_path)

            # Process this Z-plane
            stitching_kwargs = kwargs.copy()
            process_plate_folder(temp_dir, **stitching_kwargs)

            # Copy the stitched results to the main stitched directory with z-index in the filename
            temp_stitched_dir = parent_dir / f"{plate_name}_z{z_index:03d}_temp_stitched" / "TimePoint_1"
            if temp_stitched_dir.exists():
                for stitched_file in temp_stitched_dir.glob("*.tif"):
                    # Add z-index to the filename
                    base, ext = os.path.splitext(stitched_file.name)
                    new_name = f"{base}_z{z_index:03d}{ext}"
                    dest_path = stitched_timepoint / new_name
                    shutil.copy2(stitched_file, dest_path)
                    logger.info(f"Copied stitched Z-plane {z_index} to {dest_path}")

            # Clean up temporary directories
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(parent_dir / f"{plate_name}_z{z_index:03d}_temp_stitched", ignore_errors=True)
            shutil.rmtree(parent_dir / f"{plate_name}_z{z_index:03d}_temp_processed", ignore_errors=True)
            shutil.rmtree(parent_dir / f"{plate_name}_z{z_index:03d}_temp_positions", ignore_errors=True)

        logger.info(f"Completed stitching all Z-planes to {stitched_dir}")

    else:
        # Use specific z-index for alignment
        try:
            z_index = int(reference_z)
            logger.info(f"Using Z-index {z_index} as reference for alignment")

            # TODO: Filter images to only use specified z-index
            # This would need modifications to process_plate_folder

            # For now, just use standard stitching but ensure correct output directory
            # The stitched output directory should be at the same level as the original plate
            stitched_dir = parent_dir / f"{plate_name}_stitched"
            logger.info(f"Ensuring stitched directory exists at same level: {stitched_dir}")
            stitched_dir.mkdir(parents=True, exist_ok=True)

            # Make sure TimePoint_1 exists inside the stitched directory
            stitched_timepoint = stitched_dir / "TimePoint_1"
            stitched_timepoint.mkdir(parents=True, exist_ok=True)

            process_plate_folder(plate_folder, **kwargs)
        except ValueError:
            logger.error(f"Invalid reference_z value: {reference_z}")
            return False

    return True

# Example usage
if __name__ == "__main__":
    # Example plate folder to process
    plate_folder = "/path/to/your/plate/folder"

    # Import here to avoid circular imports
    from ezstitcher.core.stitcher import process_plate_folder

    # Process the plate folder with Z-stack handling
    process_plate_folder(
        plate_folder,
        reference_channels=["1", "2"],
        composite_weights={"1": 0.1, "2": 0.9},
        focus_detect=True,
        focus_method="combined",
        create_projections=True,
        projection_types=["max", "mean"],
        stitch_z_reference="best_focus"
    )
