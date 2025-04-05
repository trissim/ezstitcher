"""
Image processing module for ezstitcher.

This module contains the ImageProcessor class for handling image processing operations.
"""

import numpy as np
import logging
from pathlib import Path
import tifffile
from skimage import color, filters, exposure, morphology as morph, transform as trans
from scipy.ndimage import shift as subpixel_shift

from ezstitcher.core.utils import (
    load_image, save_image, create_linear_weight_mask, parse_positions_csv
)

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Class for handling image processing operations.
    """

    @staticmethod
    def blur(image, sigma=1):
        """
        Apply Gaussian blur to an image.

        Args:
            image (numpy.ndarray): Input image
            sigma (float): Standard deviation for Gaussian kernel

        Returns:
            numpy.ndarray: Blurred image
        """
        # Convert to float for processing
        image_float = image.astype(np.float32) / np.max(image)

        # Apply Gaussian blur
        if image_float.ndim == 3:
            blurred = filters.gaussian(image_float, sigma=sigma, channel_axis=-1)
        else:
            blurred = filters.gaussian(image_float, sigma=sigma)

        # Scale back to original range
        blurred = exposure.rescale_intensity(blurred, in_range='image', out_range=(0, 65535))
        blurred = blurred.astype(np.uint16)

        return blurred

    @staticmethod
    def find_edge(image):
        """
        Apply edge detection to an image.

        Args:
            image (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Edge map
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            image = color.rgb2gray(image)

        # Convert to float for processing
        image_float = image.astype(np.float32) / np.max(image)

        # Apply Sobel edge detection
        edge_map = filters.sobel(image_float)

        # Scale to 16-bit range
        edge_map_rescaled = exposure.rescale_intensity(edge_map, in_range='image', out_range=(0, 65535))
        edge_map_uint16 = edge_map_rescaled.astype(np.uint16)

        return edge_map_uint16

    @staticmethod
    def tophat(image, selem_radius=50, downsample_factor=4):
        """
        Apply white top-hat transform to an image.

        Args:
            image (numpy.ndarray): Input image
            selem_radius (int): Radius of the structuring element
            downsample_factor (int): Factor to downsample the image for faster processing

        Returns:
            numpy.ndarray: Processed image
        """
        # Downsample for faster processing
        image_small = trans.resize(
            image,
            (image.shape[0] // downsample_factor, image.shape[1] // downsample_factor),
            anti_aliasing=True,
            preserve_range=True
        )

        # Create structuring element
        selem_small = morph.disk(selem_radius // downsample_factor)

        # Apply white top-hat transform
        tophat_small = morph.white_tophat(image_small, selem_small)

        # Compute background
        background_small = image_small - tophat_small

        # Upscale background to original size
        background_large = trans.resize(
            background_small,
            image.shape,
            anti_aliasing=False,
            preserve_range=True
        )

        # Subtract background from original image
        result = image - background_large

        # Clip negative values and preserve dtype
        return np.clip(result, 0, None).astype(image.dtype)

    @staticmethod
    def create_weighted_composite(images_dict, weights_dict=None):
        """
        Create a composite image by weighted combination of multiple input images.

        Args:
            images_dict (dict): Dict mapping channel names to images
            weights_dict (dict): Dict mapping channel names to weights

        Returns:
            numpy.ndarray: Composite image
        """
        if not images_dict:
            raise ValueError("No images provided")

        # Use equal weights if none provided
        if weights_dict is None:
            weight = 1.0 / len(images_dict)
            weights_dict = {channel: weight for channel in images_dict.keys()}

        composite = None
        original_dtype = None

        # Combine channels with their respective weights
        for channel, img in images_dict.items():
            if original_dtype is None:
                original_dtype = img.dtype

            # Get weight for this channel
            weight = weights_dict.get(channel, 0.0)

            # Add weighted contribution
            if composite is None:
                composite = img.astype(np.float32) * weight
            else:
                composite += img.astype(np.float32) * weight

        # Normalize and convert back to original dtype
        if original_dtype is None:
            return None

        if np.issubdtype(original_dtype, np.integer):
            max_val = np.iinfo(original_dtype).max
        else:
            max_val = 1.0  # For float dtypes

        composite = np.clip(composite, 0, max_val).astype(original_dtype)

        # Ensure the composite is 2D for stitching purposes
        if composite.ndim == 3:
            if composite.shape[0] <= 4:  # Channel-first format
                composite = np.mean(composite, axis=0).astype(original_dtype)
            elif composite.shape[2] <= 4:  # Channel-last format
                composite = np.mean(composite, axis=2).astype(original_dtype)
            else:
                composite = composite[0].astype(original_dtype)

        return composite

    @staticmethod
    def normalize_16bit_global(images, lower_percentile=0.1, upper_percentile=99.9):
        """
        Normalize a list of 2D uint16 images using global lower and upper percentiles.

        Args:
            images (list): List of 2D images
            lower_percentile (float): Lower percentile to compute
            upper_percentile (float): Upper percentile to compute

        Returns:
            list: List of normalized images
        """
        # Gather all pixels from every image
        all_pixels = np.concatenate([img.ravel() for img in images])

        # Compute global thresholds
        lower_val = np.percentile(all_pixels, lower_percentile)
        upper_val = np.percentile(all_pixels, upper_percentile)

        if upper_val <= lower_val:
            logger.warning("Upper threshold must be greater than lower threshold")
            return images

        # Process each image
        normalized_images = []
        for img in images:
            # Convert to float for scaling
            img_float = img.astype(np.float32)

            # Apply linear scaling
            norm = (img_float - lower_val) / (upper_val - lower_val)

            # Clip values to [0, 1]
            norm = np.clip(norm, 0, 1)

            # Scale to full 16-bit range
            norm_img = (norm * 65535.0).astype(np.uint16)
            normalized_images.append(norm_img)

        return normalized_images

    @staticmethod
    def hist_match_stack(images, reference=None, out_range=None):
        """
        Normalize a list of images by matching each image's histogram to a reference.

        Args:
            images (list): List of images
            reference (numpy.ndarray): Reference image
            out_range (tuple): Desired output range for float images

        Returns:
            list: List of histogram-matched images
        """
        # Use median of stack as reference if none provided
        if reference is None:
            stack = np.stack(images, axis=0)
            reference = np.median(stack, axis=0)

        # Determine normalization range for reference
        if np.issubdtype(reference.dtype, np.integer):
            ref_min, ref_max = np.iinfo(reference.dtype).min, np.iinfo(reference.dtype).max
        else:
            if out_range is not None:
                ref_min, ref_max = out_range
            else:
                ref_min, ref_max = 0.0, 1.0

        # Convert reference to float [0,1]
        ref_float = (reference.astype(np.float64) - ref_min) / (ref_max - ref_min)

        normalized_images = []
        for img in images:
            if np.issubdtype(img.dtype, np.integer):
                img_min, img_max = np.iinfo(img.dtype).min, np.iinfo(img.dtype).max
            else:
                if out_range is not None:
                    img_min, img_max = out_range
                else:
                    img_min, img_max = 0.0, 1.0

            # Convert image to float [0,1]
            img_float = (img.astype(np.float64) - img_min) / (img_max - img_min)

            # Perform histogram matching
            matched_float = exposure.match_histograms(img_float, ref_float)

            # Scale back to original range
            rescaled = matched_float * (img_max - img_min) + img_min
            if np.issubdtype(img.dtype, np.integer):
                rescaled = np.round(rescaled)
                # Clip to valid range for integer dtype
                rescaled = np.clip(rescaled, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max)

            normalized_images.append(rescaled.astype(img.dtype))

        return normalized_images

    @staticmethod
    def process_bf(images):
        """
        Process brightfield images.

        Args:
            images (list): List of brightfield images

        Returns:
            list: List of processed images
        """
        # Normalize globally
        norm_images = ImageProcessor.normalize_16bit_global(
            images, upper_percentile=99, lower_percentile=0.1
        )

        # Match histograms
        norm_images = ImageProcessor.hist_match_stack(norm_images)

        # Normalize again
        norm_images = ImageProcessor.normalize_16bit_global(
            norm_images, upper_percentile=90, lower_percentile=0.1
        )

        # Find edges
        processed_images = [ImageProcessor.find_edge(img) for img in images]

        return processed_images

    @staticmethod
    def assemble_image_subpixel(positions_path, images_dir, output_path, margin_ratio=0.1, override_names=None):
        """
        Assemble a stitched image using subpixel positions from a CSV file.

        Args:
            positions_path (str or Path): Path to the CSV with subpixel positions
            images_dir (str or Path): Directory containing image tiles
            output_path (str or Path): Path to save final stitched image
            margin_ratio (float): Fraction of tile edges to blend
            override_names (list): Optional list of filenames to use instead of those in CSV

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Parse CSV file
            pos_entries = parse_positions_csv(positions_path)
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
            first_tile = load_image(images_dir / pos_entries[0][0])
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
                tile_img = load_image(images_dir / fname)
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
            save_image(output_path, blended)

            return True

        except Exception as e:
            logger.error(f"Error in assemble_image_subpixel: {e}")
            return False
