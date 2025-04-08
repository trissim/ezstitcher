import numpy as np
import logging
from pathlib import Path
import tifffile
from skimage import color, filters, exposure, morphology as morph, transform as trans
from scipy.ndimage import shift as subpixel_shift

from ezstitcher.core.utils import (
    load_image, save_image, create_linear_weight_mask, parse_positions_csv
)
from ezstitcher.core.config import ImagePreprocessorConfig

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Handles image normalization, filtering, and compositing.
    """
    def __init__(self, config: ImagePreprocessorConfig = None):
        if config is None:
            config = ImagePreprocessorConfig()
        self.config = config
        self.preprocessing_funcs = config.preprocessing_funcs or {}
        self.composite_weights = config.composite_weights or {}

    def preprocess(self, image, channel: str):
        """
        Apply preprocessing to a single image for a given channel.

        Args:
            image (numpy.ndarray): Input image
            channel (str): Channel identifier

        Returns:
            numpy.ndarray: Processed image
        """
        func = self.preprocessing_funcs.get(channel)
        if func:
            return func(image)
        return image

    def blur(self, image, sigma=1):
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

    def sharpen(self, image, radius=1, amount=1.0):
        """
        Sharpen an image using unsharp masking.

        Args:
            image (numpy.ndarray): Input image
            radius (float): Radius of Gaussian blur
            amount (float): Sharpening strength

        Returns:
            numpy.ndarray: Sharpened image
        """
        # Convert to float for processing
        image_float = image.astype(np.float32) / np.max(image)

        # Create blurred version for unsharp mask
        if image_float.ndim == 3:
            blurred = filters.gaussian(image_float, sigma=radius, channel_axis=-1)
        else:
            blurred = filters.gaussian(image_float, sigma=radius)

        # Apply unsharp mask: original + amount * (original - blurred)
        sharpened = image_float + amount * (image_float - blurred)

        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 1.0)

        # Scale back to original range
        sharpened = exposure.rescale_intensity(sharpened, in_range='image', out_range=(0, 65535))
        sharpened = sharpened.astype(np.uint16)

        return sharpened

    def enhance_contrast(self, image, percentile_low=2, percentile_high=98):
        """
        Enhance contrast using percentile-based normalization.

        Args:
            image (numpy.ndarray): Input image
            percentile_low (float): Lower percentile for contrast stretching
            percentile_high (float): Upper percentile for contrast stretching

        Returns:
            numpy.ndarray: Contrast-enhanced image
        """
        # Get percentile values
        p_low, p_high = np.percentile(image, (percentile_low, percentile_high))

        # Apply contrast stretching
        enhanced = exposure.rescale_intensity(image, in_range=(p_low, p_high), out_range=(0, 65535))
        enhanced = enhanced.astype(np.uint16)

        return enhanced

    def normalize(self, image, target_min=0, target_max=65535):
        """
        Normalize image to specified range.

        Args:
            image (numpy.ndarray): Input image
            target_min (int): Target minimum value
            target_max (int): Target maximum value

        Returns:
            numpy.ndarray: Normalized image
        """
        # Get current min and max
        img_min = np.min(image)
        img_max = np.max(image)

        # Avoid division by zero
        if img_max == img_min:
            return np.ones_like(image) * target_min

        # Normalize to target range
        normalized = (image - img_min) * (target_max - target_min) / (img_max - img_min) + target_min
        normalized = normalized.astype(np.uint16)

        return normalized

    def create_composite(self, images, weights=None):
        """
        Create a composite image from multiple channels.

        Args:
            images (dict): Dictionary mapping channel names to images
            weights (dict): Optional dictionary with weights for each channel

        Returns:
            numpy.ndarray: Composite RGB image
        """
        if weights is None:
            weights = self.composite_weights

        # Default weights if none provided
        if not weights:
            channels = list(images.keys())
            weights = {ch: 1.0 / len(channels) for ch in channels}

        # Create empty composite
        shape = next(iter(images.values())).shape
        composite = np.zeros((*shape, 3), dtype=np.float32)

        # Add each channel with its weight
        for channel, image in images.items():
            # Normalize to 0-1 range
            norm_img = image.astype(np.float32) / 65535.0

            # Get weight and color for this channel
            weight = weights.get(channel, 0.0)

            # Default color mapping (can be customized)
            if channel == '1':
                # Red channel
                composite[..., 0] += norm_img * weight
            elif channel == '2':
                # Green channel
                composite[..., 1] += norm_img * weight
            elif channel == '3':
                # Blue channel
                composite[..., 2] += norm_img * weight
            else:
                # Grayscale (add to all channels)
                for i in range(3):
                    composite[..., i] += norm_img * weight / 3.0

        # Clip to valid range
        composite = np.clip(composite, 0, 1.0)

        # Convert to 8-bit for RGB
        composite = (composite * 255).astype(np.uint8)

        return composite

    def apply_mask(self, image, mask):
        """
        Apply a mask to an image.

        Args:
            image (numpy.ndarray): Input image
            mask (numpy.ndarray): Mask image (same shape as input)

        Returns:
            numpy.ndarray: Masked image
        """
        # Ensure mask has same shape as image
        if mask.shape != image.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match image shape {image.shape}")

        # Apply mask
        masked = image.astype(np.float32) * mask.astype(np.float32)
        masked = masked.astype(image.dtype)

        return masked

    def create_weight_mask(self, shape, margin_ratio=0.1):
        """
        Create a weight mask for blending images.

        Args:
            shape (tuple): Shape of the mask (height, width)
            margin_ratio (float): Ratio of image size to use as margin

        Returns:
            numpy.ndarray: Weight mask
        """
        return create_linear_weight_mask(shape, margin_ratio)
