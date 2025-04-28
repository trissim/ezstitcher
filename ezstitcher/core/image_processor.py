import numpy as np
import logging
from skimage import filters, exposure, morphology as morph, transform as trans

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


class ImageProcessor:
    """
    Handles image normalization, filtering, and compositing.
    All methods are static and do not require an instance.
    """

    @staticmethod
    def sharpen(image, radius=1, amount=1.0):
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

    @staticmethod
    def percentile_normalize(image, low_percentile=1, high_percentile=99, target_min=0, target_max=65535):
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

    @staticmethod
    def stack_percentile_normalize(stack, low_percentile=1, high_percentile=99, target_min=0, target_max=65535):
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

    @staticmethod
    def create_composite(images, weights=None):
        """
        Create a grayscale composite image from multiple channels.

        Args:
            images (list): List of images to composite
            weights (list, optional): List of weights for each image. If None, equal weights are used.

        Returns:
            numpy.ndarray: Grayscale composite image (16-bit)

        Raises:
            TypeError: If images is not a list or weights is not a list
            ValueError: If images list is empty
        """
        # Ensure images is a list
        if not isinstance(images, list):
            raise TypeError("images must be a list of images")

        # Check for empty list early
        if not images:
            raise ValueError("images list cannot be empty")

        # Default weights if none provided
        if weights is None:
            # Equal weights for all images
            weights = [1.0 / len(images)] * len(images)
        elif not isinstance(weights, list):
            raise TypeError("weights must be a list of values")

        # Make sure weights list is at least as long as images list
        if len(weights) < len(images):
            weights = weights + [0.0] * (len(images) - len(weights))
        # Truncate weights if longer than images
        weights = weights[:len(images)]

        first_image = images[0]
        shape = first_image.shape
        dtype = first_image.dtype

        # Create empty composite
        composite = np.zeros(shape, dtype=np.float32)
        total_weight = 0.0

        # Add each image with its weight
        for i, image in enumerate(images):
            weight = weights[i]
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

    @staticmethod
    def apply_mask(image, mask):
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

    @staticmethod
    def create_weight_mask(shape, margin_ratio=0.1):
        """
        Create a weight mask for blending images.

        Args:
            shape (tuple): Shape of the mask (height, width)
            margin_ratio (float): Ratio of image size to use as margin

        Returns:
            numpy.ndarray: Weight mask
        """
        return create_linear_weight_mask(shape[0], shape[1], margin_ratio)

    @staticmethod
    def max_projection(stack):
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

    @staticmethod
    def mean_projection(stack):
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

    @staticmethod
    def stack_equalize_histogram(stack, bins=65536, range_min=0, range_max=65535):
        """
        Apply true histogram equalization to an entire stack of images.
        This ensures consistent contrast enhancement across all images in the stack.

        Unlike standard histogram equalization applied to individual images,
        this method computes a global histogram across the entire stack and
        applies the same transformation to all images, preserving relative
        intensity relationships between Z-planes.

        Args:
            stack (list or numpy.ndarray): Stack of images
            bins (int): Number of bins for histogram computation
            range_min (int): Minimum value for histogram range
            range_max (int): Maximum value for histogram range

        Returns:
            numpy.ndarray: Histogram-equalized stack of images
        """
        # Convert to numpy array if it's a list
        if isinstance(stack, list):
            stack = np.array(stack)

        # Flatten the entire stack to compute the global histogram
        flat_stack = stack.flatten()

        # Calculate the histogram and cumulative distribution function (CDF)
        hist, bin_edges = np.histogram(flat_stack, bins=bins, range=(range_min, range_max))
        cdf = hist.cumsum()

        # Normalize the CDF to the range [0, 65535]
        # Avoid division by zero
        if cdf[-1] > 0:
            cdf = 65535 * cdf / cdf[-1]

        # Use linear interpolation to map input values to equalized values
        equalized_stack = np.interp(stack.flatten(), bin_edges[:-1], cdf).reshape(stack.shape)

        # Convert to uint16
        return equalized_stack.astype(np.uint16)


    @staticmethod
    def create_projection(stack, method="max_projection", focus_analyzer=None):
        """
        Create a projection from a stack using the specified method.

        Args:
            stack (list): List of images
            method (str): Projection method (max_projection, mean_projection, best_focus)
            focus_analyzer (FocusAnalyzer, optional): Focus analyzer for best_focus method

        Returns:
            numpy.ndarray: Projected image
        """
        if method == "max_projection":
            return ImageProcessor.max_projection(stack)

        if method == "mean_projection":
            return ImageProcessor.mean_projection(stack)

        if method == "best_focus":
            if focus_analyzer is None:
                logger.warning("No focus analyzer provided for best_focus method, "
                              "using max_projection instead")
                return ImageProcessor.max_projection(stack)
            best_idx, _ = focus_analyzer.find_best_focus(stack)
            return stack[best_idx]

        # Default case for unknown methods
        logger.warning("Unknown projection method: %s, using max_projection", method)
        return ImageProcessor.max_projection(stack)

    @staticmethod
    def tophat(image, selem_radius=50, downsample_factor=4):
        """
        Apply white top-hat filter to an image for background removal.

        This implementation uses downsampling for efficiency with large structuring elements.

        Args:
            image (numpy.ndarray): Input image
            selem_radius (int): Radius of the structuring element disk
            downsample_factor (int): Factor by which to downsample the image for processing

        Returns:
            numpy.ndarray: Filtered image with background removed
        """
        # Store original data type
        input_dtype = image.dtype

        # 1) Downsample
        #    For grayscale images: trans.resize with anti_aliasing=True
        image_small = trans.resize(image,
                                   (image.shape[0]//downsample_factor,
                                    image.shape[1]//downsample_factor),
                                   anti_aliasing=True, preserve_range=True)

        # 2) Build structuring element for the smaller image
        selem_small = morph.disk(selem_radius // downsample_factor)

        # 3) White top-hat on the smaller image
        tophat_small = morph.white_tophat(image_small, selem_small)

        # 4) Upscale background to original size
        background_small = image_small - tophat_small
        background_large = trans.resize(background_small,
                                        image.shape,
                                        anti_aliasing=False,
                                        preserve_range=True)

        # 5) Subtract background and clip negative values
        result = np.maximum(image - background_large, 0)

        # 6) Convert back to original data type
        result = result.astype(input_dtype)

        return result