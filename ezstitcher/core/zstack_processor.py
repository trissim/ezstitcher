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
    # (rest of class unchanged)