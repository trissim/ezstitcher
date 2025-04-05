"""
Z-stack management module for ezstitcher.

This module contains the ZStackManager class for organizing and processing Z-stacks.
"""

import os
import re
import shutil
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
import tifffile

from ezstitcher.core.utils import ensure_directory, load_image, save_image
from ezstitcher.core.focus_detector import FocusDetector

logger = logging.getLogger(__name__)

class ZStackManager:
    """
    Class for handling Z-stack organization and processing.
    """
    
    @staticmethod
    def detect_zstack_folders(plate_folder):
        """
        Detect if a plate folder contains Z-stack folders.
        
        Args:
            plate_folder (str or Path): Path to the plate folder
            
        Returns:
            tuple: (has_zstack, z_folders) where z_folders is a list of Z-stack folder paths
        """
        plate_path = Path(plate_folder)
        timepoint_path = plate_path / "TimePoint_1"
        
        if not timepoint_path.exists():
            logger.error(f"TimePoint_1 folder does not exist in {plate_folder}")
            return False, []
        
        # Look for ZStep_* folders
        z_pattern = re.compile(r'ZStep_(\d+)')
        z_folders = []
        
        for item in timepoint_path.iterdir():
            if item.is_dir():
                match = z_pattern.match(item.name)
                if match:
                    z_index = int(match.group(1))
                    z_folders.append((z_index, item))
        
        # Sort by Z-index
        z_folders.sort(key=lambda x: x[0])
        
        has_zstack = len(z_folders) > 0
        if has_zstack:
            logger.info(f"Found {len(z_folders)} Z-stack folders in {plate_folder}")
            for z_index, folder in z_folders[:3]:  # Log first 3 for brevity
                logger.info(f"Z-stack folder: {folder.name}, Z-index: {z_index}")
        else:
            logger.info(f"No Z-stack folders found in {plate_folder}")
        
        return has_zstack, z_folders
    
    @staticmethod
    def organize_zstack_folders(plate_folder):
        """
        Organize Z-stack folders by moving files to TimePoint_1 with proper naming.
        
        Args:
            plate_folder (str or Path): Path to the plate folder
            
        Returns:
            bool: True if Z-stack was organized, False otherwise
        """
        has_zstack, z_folders = ZStackManager.detect_zstack_folders(plate_folder)
        
        if not has_zstack:
            return False
        
        plate_path = Path(plate_folder)
        timepoint_path = plate_path / "TimePoint_1"
        
        # Process each Z-stack folder
        for z_index, z_folder in z_folders:
            logger.info(f"Processing Z-stack folder: {z_folder.name}")
            
            # Get all image files in the Z-folder
            image_files = []
            for ext in ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                image_files.extend(list(z_folder.glob(f"*{ext}")))
            
            # Move and rename each file
            for img_file in image_files:
                # Extract components from filename
                # Typical pattern: A01_s001_w1.tif
                match = re.match(r'([A-Z]\d+_s\d+_w\d+)(\..*)', img_file.name)
                
                if match:
                    base_name = match.group(1)
                    extension = match.group(2)
                    
                    # Create new filename with Z-index
                    new_name = f"{base_name}_z{z_index:03d}{extension}"
                    new_path = timepoint_path / new_name
                    
                    # Move and rename file
                    shutil.copy2(img_file, new_path)
                    logger.info(f"Copied {img_file.name} to {new_path.name}")
                else:
                    logger.warning(f"Could not parse filename: {img_file.name}")
        
        return True
    
    @staticmethod
    def detect_zstack_images(folder_path):
        """
        Detect if a folder contains Z-stack images based on filename patterns.
        
        Args:
            folder_path (str or Path): Path to the folder
            
        Returns:
            tuple: (has_zstack, z_indices_map) where z_indices_map is a dict mapping base filenames to Z-indices
        """
        folder_path = Path(folder_path)
        
        # Get all image files
        all_files = []
        for ext in ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            all_files.extend(list(folder_path.glob(f"*{ext}")))
        
        # Pattern to extract Z-index from filename
        z_pattern = re.compile(r'(.+)_z(\d+)(.+)')
        
        # Map of base filename to list of Z-indices
        z_indices = defaultdict(list)
        
        # Check each file
        for img_file in all_files:
            match = z_pattern.search(img_file.name)
            if match:
                base_name = match.group(1)  # Part before z-index
                z_index = int(match.group(2))  # z-index as integer
                suffix = match.group(3)  # Part after z-index
                
                # Add to z_indices dictionary
                z_indices[base_name].append(z_index)
                logger.debug(f"Matched z-index: {img_file.name} -> base:{base_name}, z:{z_index}")
            else:
                logger.debug(f"No z-index match for file: {img_file.name}")
        
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
    
    @staticmethod
    def load_image_stack(folder_path, base_name, z_indices, file_ext=None):
        """
        Load all images in a Z-stack into memory.
        
        Args:
            folder_path (str or Path): Path to the folder containing images
            base_name (str): Base filename without z-index
            z_indices (list): List of z-indices to load
            file_ext (str): File extension (if None, will try to detect automatically)
            
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
            
            img = load_image(file_path)
            if img is not None:
                image_stack.append((z_index, img))
        
        # Sort by z_index
        image_stack.sort(key=lambda x: x[0])
        return image_stack
    
    @staticmethod
    def create_projection(images, proj_type='max'):
        """
        Create a projection from a stack of images.
        
        Args:
            images (list): List of images
            proj_type (str): Projection type ('max', 'min', 'mean', 'std', 'sum')
            
        Returns:
            numpy.ndarray: Projection image
        """
        if not images:
            return None
        
        # Convert to numpy array
        stack = np.stack(images, axis=0)
        dtype = images[0].dtype
        
        # Create projection
        if proj_type == 'max':
            projection = np.max(stack, axis=0)
        elif proj_type == 'min':
            projection = np.min(stack, axis=0)
        elif proj_type == 'mean':
            projection = np.mean(stack, axis=0).astype(dtype)
        elif proj_type == 'std':
            projection = np.std(stack, axis=0).astype(dtype)
        elif proj_type == 'sum':
            projection = np.sum(stack, axis=0)
            # Clip to prevent overflow
            if np.issubdtype(dtype, np.integer):
                max_val = np.iinfo(dtype).max
                projection = np.clip(projection, 0, max_val)
        else:
            logger.error(f"Unknown projection type: {proj_type}")
            return None
        
        return projection.astype(dtype)
    
    @staticmethod
    def create_zstack_projections(input_dir, output_dir, projection_types=None):
        """
        Create 3D projections from Z-stacks.
        
        Args:
            input_dir (str or Path): Directory with Z-stack images
            output_dir (str or Path): Directory to save projections
            projection_types (list): List of projection types
            
        Returns:
            dict: Mapping of base names to created projection files
        """
        if projection_types is None:
            projection_types = ['max', 'mean']
        
        input_dir = Path(input_dir)
        output_dir = ensure_directory(output_dir)
        
        # Check if folder contains Z-stack images
        has_zstack, z_indices_map = ZStackManager.detect_zstack_images(input_dir)
        if not has_zstack:
            logger.warning(f"No Z-stack images found in {input_dir}")
            return {}
        
        # Track created projections
        projections_created = defaultdict(list)
        
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
            image_stack = ZStackManager.load_image_stack(input_dir, base_name, z_indices, file_ext)
            if not image_stack:
                logger.error(f"Failed to load stack for {base_name}")
                continue
            
            # Extract images only
            images = [img for _, img in image_stack]
            
            # Create each projection type
            for proj_type in projection_types:
                projection = ZStackManager.create_projection(images, proj_type)
                if projection is None:
                    continue
                
                # Save projection
                output_name = f"{base_name}_{proj_type}proj{file_ext}"
                output_path = output_dir / output_name
                
                if save_image(output_path, projection):
                    logger.info(f"Created {proj_type} projection: {output_path}")
                    projections_created[base_name].append((proj_type, str(output_path)))
        
        return dict(projections_created)
    
    @staticmethod
    def find_best_focus_in_stack(image_stack, method='combined', roi=None):
        """
        Find the best focused image in a Z-stack.
        
        Args:
            image_stack (list): List of (z_index, image) tuples
            method (str): Focus detection method
            roi (tuple): Optional region of interest
            
        Returns:
            tuple: (best_z_index, best_image, scores)
        """
        if not image_stack:
            return None, None, []
        
        # Extract images and z-indices
        z_indices = [z for z, _ in image_stack]
        images = [img for _, img in image_stack]
        
        # Find best focus
        best_idx, scores = FocusDetector.find_best_focus(images, method, roi)
        
        # Get best z-index and image
        best_z_index = z_indices[best_idx]
        best_image = images[best_idx]
        
        return best_z_index, best_image, scores
    
    @staticmethod
    def select_best_focus_zstack(input_dir, output_dir, focus_method='combined', focus_wavelength='all'):
        """
        Select the best focused image from each Z-stack and save to output directory.
        
        Args:
            input_dir (str or Path): Directory with Z-stack images
            output_dir (str or Path): Directory to save best focus images
            focus_method (str): Focus detection method
            focus_wavelength (str): Wavelength to use for focus detection
            
        Returns:
            dict: Mapping of image coordinates to best z-indices
        """
        input_dir = Path(input_dir)
        output_dir = ensure_directory(output_dir)
        
        # Check if folder contains Z-stack images
        has_zstack, z_indices_map = ZStackManager.detect_zstack_images(input_dir)
        if not has_zstack:
            logger.warning(f"No Z-stack images found in {input_dir}")
            return {}
        
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
        
        # Process each stack to find best focus
        best_focus_results = {}
        
        # Determine which wavelengths to use for focus detection
        wavelengths_to_process = []
        if focus_wavelength == 'all':
            # Use all available wavelengths
            wavelengths_to_process = sorted(set(w for _, _, w in images_by_coordinates.keys()))
        else:
            # Use specified wavelength
            try:
                wavelengths_to_process = [int(focus_wavelength)]
            except ValueError:
                logger.error(f"Invalid wavelength: {focus_wavelength}")
                return {}
        
        logger.info(f"Using wavelength(s) {wavelengths_to_process} for focus detection")
        
        # Process each coordinate group
        for coordinates, (base_name, z_indices) in images_by_coordinates.items():
            well, site, wavelength = coordinates
            
            # Skip if not in wavelengths to process
            if wavelength not in wavelengths_to_process:
                continue
            
            # Try to extract extension from a sample file
            sample_files = list(input_dir.glob(f"{base_name}_z*.*"))
            if not sample_files:
                logger.warning(f"Could not find sample file for {base_name}")
                continue
                
            file_ext = sample_files[0].suffix
            
            # Load the image stack
            image_stack = ZStackManager.load_image_stack(input_dir, base_name, z_indices, file_ext)
            if not image_stack:
                logger.error(f"Failed to load stack for {base_name}")
                continue
            
            # Find best focus
            best_z, best_img, scores = ZStackManager.find_best_focus_in_stack(
                image_stack, method=focus_method
            )
            
            if best_z is None:
                logger.warning(f"Could not find best focus for {base_name}")
                continue
            
            # Save best focus image
            output_name = f"{well}_s{site:03d}_w{wavelength}{file_ext}"
            output_path = output_dir / output_name
            
            if save_image(output_path, best_img):
                logger.info(f"Saved best focus image for {coordinates}: z={best_z}, file={output_path}")
                best_focus_results[coordinates] = best_z
            
            # For wavelength used for focus detection, also save all other wavelengths at same z
            if wavelength in wavelengths_to_process:
                # Find all other wavelengths for this well and site
                for other_coords, (other_base, other_z_indices) in images_by_coordinates.items():
                    other_well, other_site, other_wavelength = other_coords
                    
                    # Skip if same wavelength or different well/site
                    if (other_wavelength == wavelength or 
                        other_well != well or 
                        other_site != site):
                        continue
                    
                    # Check if the best z-index exists for this wavelength
                    if best_z in other_z_indices:
                        # Load the image at the best z-index
                        other_file_ext = sample_files[0].suffix
                        other_file = input_dir / f"{other_base}_z{best_z:03d}{other_file_ext}"
                        
                        if other_file.exists():
                            other_img = load_image(other_file)
                            if other_img is not None:
                                # Save the image
                                other_output = output_dir / f"{other_well}_s{other_site:03d}_w{other_wavelength}{other_file_ext}"
                                if save_image(other_output, other_img):
                                    logger.info(f"Saved corresponding image for {other_coords}: z={best_z}")
                                    best_focus_results[other_coords] = best_z
        
        return best_focus_results
    
    @staticmethod
    def preprocess_plate_folder(plate_folder):
        """
        Preprocess a plate folder to detect and organize Z-stacks.
        
        Args:
            plate_folder (str or Path): Path to the plate folder
            
        Returns:
            tuple: (has_zstack, z_info) where z_info is a dict with Z-stack information
        """
        plate_path = Path(plate_folder)
        
        # First check for ZStep folders
        has_zstack_folders, z_folders = ZStackManager.detect_zstack_folders(plate_folder)
        
        # If Z-stack folders found, organize them
        if has_zstack_folders:
            logger.info(f"Organizing Z-stack folders in {plate_folder}")
            ZStackManager.organize_zstack_folders(plate_folder)
        
        # Then check for z-index in filenames
        timepoint_path = plate_path / "TimePoint_1"
        if timepoint_path.exists():
            # Now detect Z-stack images
            has_zstack_images, z_indices_map = ZStackManager.detect_zstack_images(timepoint_path)
        else:
            has_zstack_images = False
            z_indices_map = {}
        
        # Determine overall z-stack status
        has_zstack = has_zstack_folders or has_zstack_images
        
        if has_zstack:
            logger.info(f"Z-stack detected in {plate_folder}")
        else:
            logger.info(f"No Z-stack detected in {plate_folder}")
        
        # Return Z-stack information
        z_info = {
            'has_zstack_folders': has_zstack_folders,
            'z_folders': z_folders,
            'has_zstack_images': has_zstack_images,
            'z_indices_map': z_indices_map
        }
        
        return has_zstack, z_info
    
    @staticmethod
    def stitch_across_z(plate_folder, reference_z='all', **kwargs):
        """
        Stitch each Z-plane separately.
        
        Args:
            plate_folder (str or Path): Path to the plate folder
            reference_z (str or int): Z-index to use for reference or 'all' for all Z-planes
            **kwargs: Additional arguments to pass to process_plate_folder
            
        Returns:
            bool: Success status
        """
        # Import here to avoid circular imports
        from ezstitcher.core.stitcher import process_plate_folder
        
        # First preprocess to organize z-stacks if needed
        has_zstack, z_info = ZStackManager.preprocess_plate_folder(plate_folder)
        
        # Get the parent directory and plate name for correct folder structure
        plate_path = Path(plate_folder)
        parent_dir = plate_path.parent
        plate_name = plate_path.name
        
        if not has_zstack:
            logger.warning(f"No Z-stack detected in {plate_folder}, using standard stitching")
            process_plate_folder(plate_folder, **kwargs)
            return True
        
        # Create stitched directory
        stitched_dir = parent_dir / f"{plate_name}_stitched"
        ensure_directory(stitched_dir)
        
        # Make sure TimePoint_1 exists inside the stitched directory
        stitched_timepoint = stitched_dir / "TimePoint_1"
        ensure_directory(stitched_timepoint)
        
        if reference_z == 'all':
            # Stitch each Z-plane separately
            logger.info("Stitching each Z-plane separately")
            
            # Get all unique z-indices from the z_info
            z_indices = set()
            for base_name, indices in z_info['z_indices_map'].items():
                z_indices.update(indices)
            z_indices = sorted(list(z_indices))
            
            logger.info(f"Found {len(z_indices)} Z-planes to process")
            
            # Process each Z-plane
            for z_index in z_indices:
                logger.info(f"Processing Z-plane {z_index}")
                
                # Create temporary directory for this Z-plane
                temp_dir = parent_dir / f"{plate_name}_z{z_index:03d}_temp"
                temp_timepoint = ensure_directory(temp_dir / "TimePoint_1")
                
                # Copy only the images for this Z-plane to the temporary directory
                z_pattern = f"_z{z_index:03d}"
                count = 0
                
                for img_file in plate_path.glob(f"*{z_pattern}*.tif"):
                    # Load the image and ensure it's 2D grayscale
                    img = load_image(img_file)
                    if img is None:
                        continue
                    
                    # Create the new filename without z-index
                    new_name = img_file.name.replace(z_pattern, "")
                    dest_path = temp_timepoint / new_name
                    
                    # Save the 2D grayscale image
                    if save_image(dest_path, img):
                        count += 1
                
                logger.info(f"Copied {count} images for Z-plane {z_index}")
                
                if count == 0:
                    logger.warning(f"No images found for Z-plane {z_index}")
                    continue
                
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
                        
                        # Copy the file
                        shutil.copy2(stitched_file, dest_path)
                        logger.info(f"Copied stitched Z-plane {z_index} to {dest_path}")
                
                # Clean up temporary directories
                shutil.rmtree(temp_dir, ignore_errors=True)
                shutil.rmtree(parent_dir / f"{plate_name}_z{z_index:03d}_temp_stitched", ignore_errors=True)
            
            return True
        else:
            # Use a specific Z-plane for reference
            try:
                if reference_z != 'best_focus':
                    reference_z = int(reference_z)
                
                # Process with standard function
                process_plate_folder(plate_folder, **kwargs)
            except ValueError:
                logger.error(f"Invalid reference_z value: {reference_z}")
                return False
        
        return True
