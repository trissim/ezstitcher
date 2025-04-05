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

    # Process each Z-step folder
    for zstep_folder, z_index in zstep_folders:
        # Zero-pad z_index to 3 digits
        z_suffix = f"_z{z_index:03d}"

        logger.info(f"Processing {zstep_folder.name} (z-index: {z_index})")

        # Get all image files in the folder
        image_files = []
        for ext in ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            image_files.extend(list(zstep_folder.glob(f"*{ext}")))

        # Move and rename each file
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

            # Log at INFO level for debugging
            logger.info(f"Moving {img_file.name} to {new_filename}")

            # Move the file
            shutil.move(str(img_file), str(destination))

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
    # This matches: example_z001.tif
    z_pattern = re.compile(r'(.+)_z(\d{1,3})(\..+)$')

    # Dictionary to track z-indices for each base filename
    z_indices = defaultdict(list)

    # Scan folder for image files with z-index pattern
    # First print all files in the folder for debugging
    all_files = []
    for ext in ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
        all_files.extend(list(folder_path.glob(f"*{ext}")))

    logger.info(f"Files in folder: {[f.name for f in all_files[:10]]}")
    logger.info(f"Looking for z-index pattern: {z_pattern.pattern}")

    # Check each file
    for ext in ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
        for img_file in folder_path.glob(f"*{ext}"):
            match = z_pattern.match(img_file.name)
            if match:
                base_name = match.group(1)  # Filename without z-index
                z_index = int(match.group(2))  # z-index as integer
                z_indices[base_name].append(z_index)
                logger.info(f"Matched z-index: {img_file.name} -> base:{base_name}, z:{z_index}")
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
            img = cv2.imread(str(file_path))
            if img is None:
                logger.warning(f"Failed to load image: {file_path}")
                continue

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
            if proj_type == 'max':
                # Maximum intensity projection
                projection = np.max(images, axis=0)
                suffix = "_maxproj"
            elif proj_type == 'mean':
                # Mean intensity projection
                projection = np.mean(images, axis=0).astype(np.uint8)
                suffix = "_meanproj"
            elif proj_type == 'min':
                # Minimum intensity projection
                projection = np.min(images, axis=0)
                suffix = "_minproj"
            elif proj_type == 'std':
                # Standard deviation projection
                projection = np.std(images, axis=0).astype(np.uint8)
                suffix = "_stdproj"
            elif proj_type == 'sum':
                # Sum projection (clamped to prevent overflow)
                summed = np.sum(images, axis=0)
                projection = np.clip(summed, 0, 255).astype(np.uint8)
                suffix = "_sumproj"
            else:
                logger.warning(f"Unknown projection type: {proj_type}")
                continue

            # Create output filename
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

    # First, check for ZStep_* folders and organize if present
    has_zstack_folders = organize_zstack_folders(plate_folder)

    # Then check for z-index in filenames
    timepoint_path = Path(plate_folder) / "TimePoint_1"
    if timepoint_path.exists():
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
        reference_z: Z-plane to use as reference ('best_focus' or specific z-index)
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
