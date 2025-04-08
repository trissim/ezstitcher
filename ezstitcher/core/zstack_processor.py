import re
import shutil
import logging
from pathlib import Path
from collections import defaultdict
from ezstitcher.core.config import ZStackConfig

logger = logging.getLogger(__name__)

class ZStackProcessor:
    """
    Handles Z-stack specific operations:
    - Detection
    - Projection creation
    - Best focus selection
    - Per-plane stitching
    """
    def __init__(self, config: ZStackConfig):
        self.config = config
        self._z_info = None
        self._z_indices = []

    def detect_z_stacks(self, plate_folder: str):
        has_zstack, self._z_info = self.preprocess_plate_folder(plate_folder)

        if has_zstack and 'z_indices_map' in self._z_info and self._z_info['z_indices_map']:
            all_z_indices = set()
            for base_name, indices in self._z_info['z_indices_map'].items():
                all_z_indices.update(indices)
            self._z_indices = sorted(list(all_z_indices))

        return has_zstack

    def preprocess_plate_folder(self, plate_folder):
        plate_path = Path(plate_folder)

        has_zstack_folders, z_folders = self.detect_zstack_folders(plate_folder)

        if has_zstack_folders:
            logger.info(f"Organizing Z-stack folders in {plate_folder}")
            self.organize_zstack_folders(plate_folder)

        timepoint_path = plate_path / "TimePoint_1"
        if timepoint_path.exists():
            has_zstack_images, z_indices_map = self.detect_zstack_images(timepoint_path)
        else:
            has_zstack_images = False
            z_indices_map = {}

        has_zstack = has_zstack_folders or has_zstack_images

        if has_zstack:
            logger.info(f"Z-stack detected in {plate_folder}")
        else:
            logger.info(f"No Z-stack detected in {plate_folder}")

        z_info = {
            'has_zstack_folders': has_zstack_folders,
            'z_folders': z_folders,
            'has_zstack_images': has_zstack_images,
            'z_indices_map': z_indices_map
        }

        return has_zstack, z_info

    def detect_zstack_folders(self, plate_folder):
        logger.debug(f"Called detect_zstack_folders with plate_folder={plate_folder}")

        try:
            plate_path = Path(plate_folder)
            timepoint_path = plate_path / "TimePoint_1"

            if not timepoint_path.exists():
                logger.error(f"TimePoint_1 folder does not exist in {plate_folder}")
                logger.debug("Returning (False, []) due to missing TimePoint_1")
                return False, []

            z_pattern = re.compile(r'ZStep_(\d+)')
            z_folders = []

            for item in timepoint_path.iterdir():
                if item.is_dir():
                    match = z_pattern.match(item.name)
                    if match:
                        z_index = int(match.group(1))
                        z_folders.append((z_index, item))

            z_folders.sort(key=lambda x: x[0])

            has_zstack = len(z_folders) > 0
            if has_zstack:
                logger.info(f"Found {len(z_folders)} Z-stack folders in {plate_folder}")
                for z_index, folder in z_folders[:3]:  # Log first 3 for brevity
                    logger.info(f"Z-stack folder: {folder.name}, Z-index: {z_index}")
            else:
                logger.info(f"No Z-stack folders found in {plate_folder}")

            logger.debug(f"Returning ({has_zstack}, z_folders with {len(z_folders)} items)")
            return has_zstack, z_folders

        except Exception as e:
            logger.error(f"Exception in detect_zstack_folders: {e}", exc_info=True)
            raise
    def pad_site_number(self, filename):
        """
        Pad site number in filename to 3 digits.

        Args:
            filename (str): Filename to pad

        Returns:
            str: Filename with padded site number
        """
        site_match = re.search(r'_s(\d{1,3})(?=_|\.)', filename)
        if site_match:
            site_num = site_match.group(1)
            if len(site_num) < 3:
                padded = site_num.zfill(3)
                filename = filename.replace(f"_s{site_num}", f"_s{padded}")
        return filename

    def organize_zstack_folders(self, plate_folder):
        """
        Organize Z-stack folders by moving files to TimePoint_1 with proper naming.

        Args:
            plate_folder (str or Path): Path to the plate folder

        Returns:
            bool: True if Z-stack was organized, False otherwise
        """
        has_zstack, z_folders = self.detect_zstack_folders(plate_folder)

        if not has_zstack:
            return False

        plate_path = Path(plate_folder)
        timepoint_path = plate_path / "TimePoint_1"

        for z_index, z_folder in z_folders:
            logger.info(f"Processing Z-stack folder: {z_folder.name}")

            image_files = []
            for ext in ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                image_files.extend(list(z_folder.glob(f"*{ext}")))

            for img_file in image_files:
                filename = self.pad_site_number(img_file.name)

                match = re.match(r'([A-Z]\d+)_s(\d+)_w(\d+)(\..*)', filename)

                if match:
                    well = match.group(1)
                    site = match.group(2).zfill(3)
                    wavelength = match.group(3)
                    extension = match.group(4)

                    new_name = f"{well}_s{site}_w{wavelength}_z{z_index:03d}{extension}"
                    new_path = timepoint_path / new_name

                    shutil.copy2(img_file, new_path)
                    logger.info(f"Copied {img_file.name} to {new_path.name}")
                else:
                    logger.warning(f"Could not parse filename: {img_file.name}")

        for z_index, z_folder in z_folders:
            try:
                shutil.rmtree(z_folder)
                logger.info(f"Removed Z-stack folder: {z_folder}")
            except Exception as e:
                logger.warning(f"Failed to remove Z-stack folder {z_folder}: {e}")

        return True
    def get_z_indices(self):
        """
        Return the list of detected Z indices after calling detect_z_stacks().
        """
        return getattr(self, '_z_indices', [])

        return has_zstack, z_folders
    def detect_zstack_images(self, folder_path):
        """
        Detect if a folder contains Z-stack images based on filename patterns.

        Args:
            folder_path (str or Path): Path to the folder

        Returns:
            tuple: (has_zstack, z_indices_map) where z_indices_map is a dict mapping base filenames to Z-indices
        """
        folder_path = Path(folder_path)

        all_files = []
        for ext in ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            all_files.extend(list(folder_path.glob(f"*{ext}")))

        z_pattern = re.compile(r'(.+)_z(\d+)(.+)')

        from collections import defaultdict
        z_indices = defaultdict(list)

        for img_file in all_files:
            match = z_pattern.search(img_file.name)
            if match:
                base_name = match.group(1)
                z_index = int(match.group(2))
                suffix = match.group(3)

                z_indices[base_name].append(z_index)
                logger.debug(f"Matched z-index: {img_file.name} -> base:{base_name}, z:{z_index}")
            else:
                logger.debug(f"No z-index match for file: {img_file.name}")

        has_zstack = len(z_indices) > 0
        if has_zstack:
            for base_name in z_indices:
                z_indices[base_name].sort()

            logger.info(f"Found Z-stack images in {folder_path}")
            logger.info(f"Detected {len(z_indices)} unique image stacks")

            for i, (base_name, indices) in enumerate(list(z_indices.items())[:3]):
                logger.info(f"Example {i+1}: {base_name} has {len(indices)} z-planes: {indices}")
        else:
            logger.info(f"No Z-stack images detected in {folder_path}")

        return has_zstack, dict(z_indices)

    def create_best_focus_images(self, input_dir, output_dir=None, focus_method='combined', focus_wavelength='all'):
        """
        Select the best focused image from each Z-stack and save to output directory.

        Args:
            input_dir (str or Path): Directory with Z-stack images
            output_dir (str or Path): Directory to save best focus images. If None, creates a directory named {plate_name}_best_focus
            focus_method (str): Focus detection method
            focus_wavelength (str): Wavelength to use for focus detection

        Returns:
            tuple: (success, output_dir) where success is a boolean and output_dir is the path to the output directory
        """
        from ezstitcher.core.utils import ensure_directory, load_image, save_image

        input_dir = Path(input_dir)

        # If output_dir is None, create a directory named {plate_name}_best_focus
        if output_dir is None:
            plate_path = input_dir.parent if input_dir.name == "TimePoint_1" else input_dir
            parent_dir = plate_path.parent
            plate_name = plate_path.name
            output_dir = parent_dir / f"{plate_name}_best_focus"

        # Create TimePoint_1 directory in output_dir
        timepoint_dir = output_dir / "TimePoint_1"
        timepoint_dir.mkdir(parents=True, exist_ok=True)

        # Check if folder contains Z-stack images
        has_zstack, z_indices_map = self.detect_zstack_images(input_dir)
        if not has_zstack:
            logger.warning(f"No Z-stack images found in {input_dir}")
            return False, None

        # Group images by well, site, and wavelength
        images_by_coordinates = defaultdict(list)

        # Pattern to extract well, site, wavelength from filename
        filename_pattern = re.compile(r'([A-Z]\d+)_s(\d+)_w(\d+).*')

        # Group Z-indices by coordinates
        for base_name, z_indices in z_indices_map.items():
            match = filename_pattern.match(base_name)
            if match:
                well = match.group(1)
                site = int(match.group(2))
                wavelength = int(match.group(3))

                # Create coordinates key
                coordinates = (well, site, wavelength)

                # Add to dictionary
                images_by_coordinates[coordinates] = (base_name, z_indices)
            else:
                logger.warning(f"Could not parse coordinates from {base_name}")

        # Process each set of coordinates
        best_focus_results = {}

        # Filter by wavelength if specified
        if focus_wavelength != 'all':
            focus_wavelength = int(focus_wavelength)
            focus_coordinates = [coords for coords in images_by_coordinates.keys() if coords[2] == focus_wavelength]
        else:
            focus_coordinates = list(images_by_coordinates.keys())

        # Process each set of focus coordinates
        for coordinates in focus_coordinates:
            well, site, wavelength = coordinates
            base_name, z_indices = images_by_coordinates[coordinates]

            # Load all Z-stack images for this coordinate
            image_stack = []
            for z_index in sorted(z_indices):
                img_path = input_dir / f"{base_name}_z{z_index:03d}.tif"
                img = load_image(img_path)
                if img is not None:
                    image_stack.append(img)

            if not image_stack:
                logger.warning(f"No valid images found for {base_name}")
                continue

            # Find best focus using FocusAnalyzer
            best_img, best_z, scores = self.focus_analyzer.select_best_focus(image_stack, method=focus_method)
            z_index = sorted(z_indices)[best_z]

            # Save best focus image
            output_filename = f"{well}_s{site:03d}_w{wavelength}.tif"
            output_path = timepoint_dir / output_filename
            save_image(output_path, best_img)
            logger.info(f"Saved best focus image for {base_name} (z={z_index}) to {output_path}")

            # Store best Z-index for this coordinate
            best_focus_results[coordinates] = z_index

        # If focus_wavelength is not 'all', use the same Z-index for other wavelengths
        if focus_wavelength != 'all':
            for coordinates in images_by_coordinates.keys():
                well, site, wavelength = coordinates
                if wavelength != focus_wavelength:
                    # Find the corresponding focus coordinates
                    focus_coords = (well, site, focus_wavelength)
                    if focus_coords in best_focus_results:
                        # Use the same Z-index as the focus wavelength
                        best_z = best_focus_results[focus_coords]
                        base_name, z_indices = images_by_coordinates[coordinates]

                        # Load the image at the best Z-index
                        img_path = input_dir / f"{base_name}_z{best_z:03d}.tif"
                        img = load_image(img_path)
                        if img is not None:
                            # Save the image
                            output_filename = f"{well}_s{site:03d}_w{wavelength}.tif"
                            output_path = timepoint_dir / output_filename
                            save_image(output_path, img)
                            logger.info(f"Saved best focus image for {base_name} (z={best_z}) to {output_path}")
                            best_focus_results[coordinates] = best_z

        return len(best_focus_results) > 0, output_dir