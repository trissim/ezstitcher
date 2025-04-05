import os
import re
import numpy as np
import skimage.io
from scipy.ndimage import shift as subpixel_shift
from skimage import color, filters, exposure, morphology as morph, transform as trans, img_as_float, img_as_uint, img_as_ubyte

def process_bf(imgs):
    #imgs =  hist_match_stack(imgs)
    #imgs= [blur(img,sigma=2) for img in imgs]
    #imgs= [tophat(img) for img in imgs]
    ##imgs = normalize_16bit_global(imgs, upper_percentile=99.9,lower_percentile=0.1)
    ##    imgs = normalize_16bit_global(imgs, upper_percentile=99.9,lower_percentile=10)
    #imgs = [find_edge(img) for img in imgs]


    norm_images = imgs
    norm_images = normalize_16bit_global(norm_images, upper_percentile=99,lower_percentile=0.1)
    norm_images=  hist_match_stack(norm_images)
    norm_images = normalize_16bit_global(norm_images, upper_percentile=90,lower_percentile=0.1)
    imgs = [find_edge(img) for img in imgs]
    return imgs

def blur(image,sigma=1):
    image_float = img_as_float(image)
    if image_float.ndim == 3:
        blurred = filters.gaussian(image_float, sigma=sigma, multichannel=True)
    else:
        blurred = filters.gaussian(image_float, sigma=sigma)

    blurred = exposure.rescale_intensity(blurred, in_range='image', out_range=(0, 65535))
    blurred = blurred.astype(np.uint16)
    return blurred

def find_edge(image):
    if image.ndim == 3:
        image = color.rgb2gray(image)
    image = img_as_float(image)
    edge_map = filters.sobel(image)
    edge_map_rescaled = exposure.rescale_intensity(edge_map, in_range='image', out_range=(0, 65535))
    edge_map_uint16 = edge_map_rescaled.astype(np.uint16)
    #return edge_map
    return edge_map_uint16

def create_weighted_composite(images_dict, weights_dict=None):
    """
    Create a composite image by weighted combination of multiple input images.

    Args:
        images_dict: Dict mapping channel names to images (numpy arrays)
        weights_dict: Dict mapping channel names to weights (default: equal weights)

    Returns:
        np.ndarray: The composite image with the same dtype as input images
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

        # Get weight for this channel (default to 0 if not in weights_dict)
        weight = weights_dict.get(channel, 0.0)

        # Add weighted contribution
        if composite is None:
            composite = img.astype(np.float32) * weight
        else:
            composite += img.astype(np.float32) * weight

    # Normalize and convert back to original dtype
    if original_dtype is None:
        # Should never happen if images_dict is not empty
        return None

    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
    else:
        max_val = 1.0  # For float dtypes, assume [0,1] range

    composite = np.clip(composite, 0, max_val).astype(original_dtype)

    # Ensure the composite is 2D for stitching purposes
    # Convert any 3D image to 2D by taking the mean along the appropriate axis
    if composite.ndim == 3:
        # Check if it's a channel-first format (C, H, W)
        if composite.shape[0] <= 4:  # Assuming max 4 channels (RGBA)
            # Convert channel-first to 2D by taking mean across channels
            composite = np.mean(composite, axis=0).astype(original_dtype)
        # Check if it's a channel-last format (H, W, C)
        elif composite.shape[2] <= 4:  # Assuming max 4 channels (RGBA)
            # Convert channel-last to 2D by taking mean across channels
            composite = np.mean(composite, axis=2).astype(original_dtype)
        else:
            # If it's a 3D image with a different structure, use the first slice
            composite = composite[0].astype(original_dtype)

    return composite

def tophat(image, selem_radius=50, downsample_factor=4):
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
    #    The 'background_small' is effectively morphological_opening(image_small),
    #    so background_small = image_small - tophat_small
    background_small = image_small - tophat_small
    background_large = trans.resize(background_small,
                                    image.shape,
                                    anti_aliasing=False,
                                    preserve_range=True)

    # 5) Subtract background
    result = image - background_large
    return np.clip(result, 0, None).astype(image.dtype)

def compute_global_reference(image_list):
    """
    Compute a global reference image using the median of the stack.
    The reference is cast to the same dtype as the first image.
    """
    stack = np.stack(image_list, axis=0)
    reference = np.median(stack, axis=0)
    return reference.astype(image_list[0].dtype)

def normalize_16bit_global(images, lower_percentile=0.1, upper_percentile=99.9):
    """
    Normalize a list of 2D uint16 images using global lower and upper percentiles.
    Each image is scaled so that:
      - Pixels at or below the global lower percentile become 0.
      - Pixels at or above the global upper percentile become 65535.
      - Intermediate values are linearly scaled between 0 and 65535.
    Args:
      images (List[np.ndarray]): List of 2D images (dtype=np.uint16).
      lower_percentile (float): Lower percentile to compute (e.g., 0.1).
      upper_percentile (float): Upper percentile to compute (e.g., 99.9).
    Returns:
      List[np.ndarray]: List of normalized images as np.uint16.
    """
    # Gather all pixels from every image into one 1D array
    all_pixels = np.concatenate([img.ravel() for img in images])
    # Compute global lower and upper threshold values
    lower_val = np.percentile(all_pixels, lower_percentile)
    upper_val = np.percentile(all_pixels, upper_percentile)
    if upper_val <= lower_val:
        warnings.warn("The computed upper threshold must be greater than the lower threshold. Skipping Image")
        return images
    # Process each image: scale, clip, then cast back to uint16
    normalized_images = []
    for img in images:
        # Convert image to float for scaling
        img_float = img.astype(np.float32)
        # Apply linear scaling: (value - lower) / (upper - lower)
        norm = (img_float - lower_val) / (upper_val - lower_val)
        # Clip values to [0, 1]
        norm = np.clip(norm, 0, 1)
        # Scale to full 16-bit range and convert back to uint16
        norm_img = (norm * 65535.0).astype(np.uint16)
        normalized_images.append(norm_img)
    return normalized_images

def hist_match_stack(images, reference=None, out_range=None):
    """
    Normalize a list of images by matching each image's histogram to a reference histogram.
    This function works for any numeric dtype.

    For integer images, the full range is used (via np.iinfo).
    For float images, if out_range is not provided, it assumes the image values are in [0, 1].

    Args:
        images (List[np.ndarray]): List of images.
        reference (np.ndarray, optional): A reference image to match histograms to.
            If None, the first image in the list is used.
        out_range (tuple, optional): Desired output range (min, max) for float images.
            If not provided, float images are assumed to be in [0, 1].

    Returns:
        List[np.ndarray]: List of histogram-matched images, in the original dtype.
    """
    # Use the first image as reference if none provided.
    if reference is None:
        reference = compute_global_reference(images)

    # Determine the normalization range for the reference image.
    if np.issubdtype(reference.dtype, np.integer):
        ref_min, ref_max = np.iinfo(reference.dtype).min, np.iinfo(reference.dtype).max
    else:
        if out_range is not None:
            ref_min, ref_max = out_range
        else:
            ref_min, ref_max = 0.0, 1.0

    # Convert the reference image to float in [0,1]
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

        # Scale back to the original range
        rescaled = matched_float * (img_max - img_min) + img_min
        if np.issubdtype(img.dtype, np.integer):
            rescaled = np.round(rescaled)
        normalized_images.append(rescaled.astype(img.dtype))

    return normalized_images

def create_linear_weight_mask(height, width, margin_ratio=0.1):
    """
    Create a 2D weight mask that linearly ramps from 0 at the edges
    to 1 in the center.
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

    mask_2d = np.outer(weight_y, weight_x)
    return mask_2d

def parse_positions_csv(csv_path):
    """
    Parse a CSV file with lines of the form:
      file: <filename>; grid: (col, row); position: (x, y)
    Returns a list of tuples: (filename, x_float, y_float).
    """
    entries = []
    with open(csv_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Example line:
            # file: some_image.tif; grid: (0, 0); position: (123.45, 67.89)
            file_match = re.search(r'file:\s*([^;]+);', line)
            pos_match = re.search(r'position:\s*\(([^,]+),\s*([^)]+)\)', line)
            if file_match and pos_match:
                fname = file_match.group(1).strip()
                x_val = float(pos_match.group(1).strip())
                y_val = float(pos_match.group(2).strip())
                entries.append((fname, x_val, y_val))
    return entries

def assemble_image_subpixel(positions_path, images_dir, output_path, margin_ratio=0.1,override_names=None):
    """
    Assemble a stitched image using subpixel positions from a CSV file.
    We only shift each tile by the fractional part of its offset, then place it
    in the final canvas at the integer part. This avoids shape mismatch.

    CSV lines must look like:
      file: <filename>; grid: (c, r); position: (x_float, y_float)

    Args:
      csv_file (str): Path to the CSV with subpixel positions.
      images_dir (str): Directory containing image tiles.
      output_path (str): Path to save final stitched image.
      margin_ratio (float): Fraction of tile edges to blend.
      override_names (list): Optional list of filenames to use instead of those in CSV.

    Returns:
      None. Saves the stitched image to output_path.
    """

    # Make sure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Parse CSV -> (filename, x_float, y_float)
    pos_entries = parse_positions_csv(positions_path)
    if not pos_entries:
        raise RuntimeError(f"No valid entries found in {positions_path}")

     # Override filenames if provided
    if override_names is not None:
        if len(override_names) != len(pos_entries):
            raise ValueError(f"Number of override_names ({len(override_names)}) doesn't match positions ({len(pos_entries)})")
        # Create new position entries with overridden filenames but same coordinates
        pos_entries = [(override_names[i], x, y) for i, (_, x, y) in enumerate(pos_entries)]

    # Check tile existence
    for (fname, _, _) in pos_entries:
        if not os.path.exists(os.path.join(images_dir, fname)):
            raise RuntimeError(f"Missing image: {fname} in {images_dir}")

    # Read the first tile to get shape, dtype
    # Use tifffile directly to avoid imagecodecs dependency
    import tifffile
    first_tile = tifffile.imread(os.path.join(images_dir, pos_entries[0][0]))

    # Force image to be 2D grayscale
    if first_tile.ndim == 3:
        # Convert to 2D by taking the mean across channels
        first_tile = np.mean(first_tile, axis=2 if first_tile.shape[2] <= 4 else 0).astype(first_tile.dtype)

    tile_h, tile_w = first_tile.shape
    dtype = first_tile.dtype
    num_channels = 1  # Always use 1 channel (grayscale)

    # Compute bounding box of the integer offsets
    # We'll separate each tile's offset into integer + fractional parts
    x_vals = []
    y_vals = []
    for _, x_f, y_f in pos_entries:
        # offset minus the fractional part => integer offset
        # but let's do it systematically:
        x_vals.append(x_f)
        y_vals.append(y_f)

    min_x = min(x_vals)
    max_x = max(x_vals) + tile_w
    min_y = min(y_vals)
    max_y = max(y_vals) + tile_h

    # final canvas size
    final_w = int(np.ceil(max_x - min_x))
    final_h = int(np.ceil(max_y - min_y))
    print(f"Final canvas size: {final_h} x {final_w} x 1")

    # Prepare accumulators - always use 2D (grayscale)
    acc = np.zeros((final_h, final_w), dtype=np.float32)
    weight_acc = np.zeros((final_h, final_w), dtype=np.float32)

    # Prepare the tile mask - always 2D
    base_mask = create_linear_weight_mask(tile_h, tile_w, margin_ratio=margin_ratio)

    for i, (fname, x_f, y_f) in enumerate(pos_entries):
        print(f"Placing tile {i+1}/{len(pos_entries)}: {fname} at subpixel ({x_f}, {y_f})")

        # Use tifffile directly to avoid imagecodecs dependency
        tile_img = tifffile.imread(os.path.join(images_dir, fname))

        # Force image to be 2D grayscale
        if tile_img.ndim == 3:
            # Convert to 2D by taking the mean across channels
            tile_img = np.mean(tile_img, axis=2 if tile_img.shape[2] <= 4 else 0).astype(tile_img.dtype)

        if tile_img.shape != (tile_h, tile_w):
            raise RuntimeError(f"Tile shape mismatch: {tile_img.shape} vs {tile_h}x{tile_w}")
        if tile_img.dtype != dtype:
            raise RuntimeError(f"Tile dtype mismatch: {tile_img.dtype} vs {dtype}")

        tile_float = tile_img.astype(np.float32)
        weighted_tile = tile_float * base_mask

        # Separate offset into integer + fractional
        shift_x = x_f - min_x
        shift_y = y_f - min_y
        int_x = int(np.floor(shift_x))
        int_y = int(np.floor(shift_y))
        frac_x = shift_x - int_x
        frac_y = shift_y - int_y

        # Shift only by the fractional portion
        shifted_tile = subpixel_shift(weighted_tile,
                                      shift=(frac_y, frac_x),
                                      order=1, mode='constant', cval=0)
        shifted_mask = subpixel_shift(base_mask,
                                      shift=(frac_y, frac_x),
                                      order=1, mode='constant', cval=0)

        # Place at the integer offset
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

    print(f"Saving stitched image to {output_path}")
    # Use tifffile directly to avoid imagecodecs dependency
    tifffile.imwrite(output_path, blended, compression=None)
    print("Done.")
