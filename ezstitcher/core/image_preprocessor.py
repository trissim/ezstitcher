import numpy as np
import logging
import re
from pathlib import Path
import tifffile
from skimage import color, filters, exposure, morphology as morph, transform as trans
from scipy.ndimage import shift as subpixel_shift

from ezstitcher.core.config import ImagePreprocessorConfig

logger = logging.getLogger(__name__)


def create_linear_weight_mask(height, width, margin_ratio=0.1):
    """
    Create a 2D weight mask that linearly ramps from 0 at the edges
    to 1 in the center.

    Args:
        height (int): Height of the mask
        width (int): Width of the mask
        margin_ratio (float): Ratio of the margin to the image size

    Returns:
        numpy.ndarray: 2D weight mask
    """
    margin_y = int(np.floor(height * margin_ratio))
    margin_x = int(np.floor(width * margin_ratio))

    weight_y = np.ones(height, dtype=np.float32)
    if margin_y > 0:
        ramp_top = np.linspace(0, 1, margin_y, endpoint=False)
        ramp_bottom = np.linspace(1, 0, margin_y, endpoint=False)
        weight_y[:margin_y] = ramp_top
        weight_y[-margin_y:] = ramp_bottom

    weight_x = np.ones(width, dtype=np.float32)
    if margin_x > 0:
        ramp_left = np.linspace(0, 1, margin_x, endpoint=False)
        ramp_right = np.linspace(1, 0, margin_x, endpoint=False)
        weight_x[:margin_x] = ramp_left
        weight_x[-margin_x:] = ramp_right

    # Create 2D weight mask
    weight_mask = np.outer(weight_y, weight_x)

    return weight_mask


# These functions have been moved to their appropriate classes:
# - load_image and save_image are now in FileSystemManager
# - parse_positions_csv is now in CSVHandler


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

    def percentile_normalize(self, image, low_percentile=1, high_percentile=99, target_min=0, target_max=65535):
        """
        Normalize image using percentile-based contrast stretching.

        Args:
            image (numpy.ndarray): Input image
            low_percentile (float): Lower percentile (0-100)
            high_percentile (float): Upper percentile (0-100)
            target_min (int): Target minimum value
            target_max (int): Target maximum value

        Returns:
            numpy.ndarray: Normalized image
        """
        # Get percentile values
        p_low, p_high = np.percentile(image, (low_percentile, high_percentile))

        # Avoid division by zero
        if p_high == p_low:
            return np.ones_like(image) * target_min

        # Clip and normalize to target range
        clipped = np.clip(image, p_low, p_high)
        normalized = (clipped - p_low) * (target_max - target_min) / (p_high - p_low) + target_min
        normalized = normalized.astype(np.uint16)

        return normalized

    def stack_percentile_normalize(self, stack, low_percentile=1, high_percentile=99, target_min=0, target_max=65535):
        """
        Normalize a stack of images using global percentile-based contrast stretching.
        This ensures consistent normalization across all images in the stack.

        Args:
            stack (list or numpy.ndarray): Stack of images
            low_percentile (float): Lower percentile (0-100)
            high_percentile (float): Upper percentile (0-100)
            target_min (int): Target minimum value
            target_max (int): Target maximum value

        Returns:
            numpy.ndarray: Normalized stack of images
        """
        # Convert to numpy array if it's a list
        if isinstance(stack, list):
            stack = np.array(stack)

        # Calculate global percentiles across the entire stack
        p_low = np.percentile(stack, low_percentile)
        p_high = np.percentile(stack, high_percentile)

        # Avoid division by zero
        if p_high == p_low:
            return np.ones_like(stack) * target_min

        # Clip and normalize to target range
        clipped = np.clip(stack, p_low, p_high)
        normalized = (clipped - p_low) * (target_max - target_min) / (p_high - p_low) + target_min
        normalized = normalized.astype(np.uint16)

        return normalized

    def create_composite(self, images, weights=None):
        """
        Create a grayscale composite image from multiple channels.

        Args:
            images (dict): Dictionary mapping channel names to images
            weights (dict): Optional dictionary with weights for each channel

        Returns:
            numpy.ndarray: Grayscale composite image (16-bit)
        """
        if weights is None:
            weights = self.composite_weights

        # Default weights if none provided
        if not weights:
            channels = list(images.keys())
            weights = {ch: 1.0 / len(channels) for ch in channels}

        # Get shape and dtype from first image
        first_image = next(iter(images.values()))
        shape = first_image.shape
        dtype = first_image.dtype

        # Create empty composite
        composite = np.zeros(shape, dtype=np.float32)
        total_weight = 0.0

        # Add each channel with its weight
        for channel, image in images.items():
            # Get weight for this channel
            weight = weights.get(channel, 0.0)
            if weight <= 0.0:
                continue

            # Add to composite
            composite += image.astype(np.float32) * weight
            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            composite /= total_weight

        # Convert back to original dtype (usually uint16)
        if np.issubdtype(dtype, np.integer):
            max_val = np.iinfo(dtype).max
            composite = np.clip(composite, 0, max_val).astype(dtype)
        else:
            composite = composite.astype(dtype)

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

    def max_projection(self, stack):
        """
        Create a maximum intensity projection from a Z-stack.

        Args:
            stack (list or numpy.ndarray): Stack of images

        Returns:
            numpy.ndarray: Maximum intensity projection
        """
        # Convert to numpy array if it's a list
        if isinstance(stack, list):
            stack = np.array(stack)

        # Create max projection
        return np.max(stack, axis=0)

    def mean_projection(self, stack):
        """
        Create a mean intensity projection from a Z-stack.

        Args:
            stack (list or numpy.ndarray): Stack of images

        Returns:
            numpy.ndarray: Mean intensity projection
        """
        # Convert to numpy array if it's a list
        if isinstance(stack, list):
            stack = np.array(stack)

        # Create mean projection
        return np.mean(stack, axis=0).astype(stack[0].dtype)
