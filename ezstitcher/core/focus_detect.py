import numpy as np
import cv2

def original_fft_focus(img):
    """
    Original FFT-based focus measure implementation.
    Counts frequency components above threshold relative to maximum.
    
    Args:
        img: Input grayscale image (2D numpy array)
        
    Returns:
        float: Focus quality score
    """
    # Ensure image is grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    x, y = img.shape
    FFT = np.fft.fft2(img)
    centerFFT = np.fft.fftshift(FFT)
    absFFT = np.absolute(centerFFT)
    maxFreq = np.max(absFFT)
    nThreshed = len(FFT[FFT > maxFreq/1000.0])
    quality = nThreshed/(x*y)
    return quality

def adaptive_fft_focus(img):
    """
    Adaptive FFT focus measure optimized for low-contrast microscopy images.
    Uses image statistics to set threshold adaptively.
    
    Args:
        img: Input grayscale image (2D numpy array)
        
    Returns:
        float: Focus quality score
    """
    # Ensure image is grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply windowing to reduce edge effects
    h, w = img.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    img_windowed = img.astype(np.float32) * window
    
    # Compute FFT and magnitude
    fft = np.fft.fft2(img_windowed)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    
    # Create a bandpass filter focusing on mid-frequencies
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    mask = np.ones((h, w))
    
    # Remove DC component and very low frequencies
    r_min = 5
    center_mask = x**2 + y**2 <= r_min**2
    mask[center_mask] = 0
    
    # Apply mask
    filtered = magnitude * mask
    
    # Adaptive thresholding based on image statistics
    mean_mag = np.mean(filtered[filtered > 0])
    std_mag = np.std(filtered[filtered > 0])
    threshold = mean_mag + 0.5 * std_mag
    
    # Count components above threshold
    count = np.sum(filtered > threshold)
    
    # Calculate focus measure
    return count / (h * w)

def normalized_variance(img):
    """
    Normalized variance focus measure.
    Performs well for microscopy images.
    
    Args:
        img: Input grayscale image (2D numpy array)
        
    Returns:
        float: Focus quality score
    """
    # Ensure image is grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img_float = img.astype(np.float32)
    mean_val = np.mean(img_float)
    if mean_val < 1e-6:  # Avoid division by near-zero
        mean_val = 1e-6
    return np.sum((img_float - mean_val)**2) / (img.shape[0] * img.shape[1] * mean_val)

def laplacian_energy(img):
    """
    Laplacian energy focus measure.
    Effective for detecting edges in microscopy images.
    
    Args:
        img: Input grayscale image (2D numpy array)
        
    Returns:
        float: Focus quality score
    """
    # Ensure image is grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return np.sum(np.abs(lap)) / (img.shape[0] * img.shape[1])

def tenengrad_variance(img, ksize=3, threshold=0):
    """
    Tenengrad variance focus measure.
    Based on gradient magnitude.
    
    Args:
        img: Input grayscale image (2D numpy array)
        ksize: Kernel size for Sobel operator
        threshold: Threshold to reduce noise impact
        
    Returns:
        float: Focus quality score
    """
    # Ensure image is grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    fm = gx**2 + gy**2
    fm[fm < threshold] = 0  # Thresholding to reduce noise impact
    return np.mean(fm)

def combined_focus_measure(img, weights=None):
    """
    Combined focus measure using multiple metrics.
    Optimized for microscopy images, especially low-contrast specimens.
    
    Args:
        img: Input grayscale image (2D numpy array)
        weights: Optional dictionary with weights for each metric
        
    Returns:
        float: Combined focus quality score
    """
    # Default weights if none provided
    if weights is None:
        weights = {
            'nvar': 0.3,
            'lap': 0.3, 
            'ten': 0.2,
            'fft': 0.2
        }
    
    # Ensure image is grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate individual metrics
    nvar = normalized_variance(img)
    lap = laplacian_energy(img)
    ten = tenengrad_variance(img)
    fft = adaptive_fft_focus(img)
    
    # Weighted combination
    score = (weights['nvar'] * nvar + 
             weights['lap'] * lap + 
             weights['ten'] * ten + 
             weights['fft'] * fft)
    
    return score

def find_best_focus(image_stack, method='combined', roi=None):
    """
    Find the best focused image in a stack using specified method
    
    Args:
        image_stack: List or array of images
        method: Focus detection method ('combined', 'nvar', 'lap', 'ten', 'fft', 'adaptive_fft')
        roi: Optional region of interest as (x, y, width, height)
        
    Returns:
        tuple: (best_focus_index, focus_scores)
    """
    focus_scores = []
    
    # Select focus measure function
    if method == 'combined':
        focus_func = combined_focus_measure
    elif method == 'nvar':
        focus_func = normalized_variance
    elif method == 'lap':
        focus_func = laplacian_energy
    elif method == 'ten':
        focus_func = tenengrad_variance
    elif method == 'fft':
        focus_func = original_fft_focus
    elif method == 'adaptive_fft':
        focus_func = adaptive_fft_focus
    else:
        raise ValueError(f"Unknown focus method: {method}")
    
    # Process each image in stack
    for i, img in enumerate(image_stack):
        # Extract ROI if specified
        if roi is not None:
            x, y, w, h = roi
            img_roi = img[y:y+h, x:x+w]
        else:
            img_roi = img
            
        # Calculate focus score
        score = focus_func(img_roi)
        focus_scores.append((i, score))
    
    # Find index with maximum focus score
    best_focus_idx = max(focus_scores, key=lambda x: x[1])[0]
    return best_focus_idx, focus_scores


# Example usage
if __name__ == "__main__":
    # Example: Load a z-stack of images
    # image_stack = [cv2.imread(f"z_stack/image_{i}.tif") for i in range(10)]
    
    # Find best focused image
    # best_idx, scores = find_best_focus(image_stack, method='combined')
    # print(f"Best focused image index: {best_idx}")
    # print(f"Focus scores: {scores}")
    
    # Example with ROI
    # roi = (100, 100, 200, 200)  # x, y, width, height
    # best_idx, scores = find_best_focus(image_stack, method='combined', roi=roi)
    pass
