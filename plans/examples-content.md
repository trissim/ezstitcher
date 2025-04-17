# Examples Content Plan

## Status: In Progress
## Progress: 0%
## Last Updated: 2024-05-15
## Dependencies: [plans/documentation-outline.md]

This document outlines the detailed content for the Examples section of the EZStitcher documentation.

## 4.1 Basic Stitching

### Simple Stitching with Default Parameters

```python
from ezstitcher.core.main import process_plate_folder

# Process a plate folder with default parameters
process_plate_folder(
    'path/to/plate_folder',
    reference_channels=["1"]
)
```

### Stitching with Custom Overlap

```python
from ezstitcher.core.main import process_plate_folder

# Process a plate folder with custom overlap
process_plate_folder(
    'path/to/plate_folder',
    reference_channels=["1"],
    tile_overlap=15.0,  # 15% overlap between tiles
    max_shift=75        # Allow larger shifts for alignment
)
```

### Stitching with Well Filtering

```python
from ezstitcher.core.main import process_plate_folder

# Process only specific wells
process_plate_folder(
    'path/to/plate_folder',
    reference_channels=["1"],
    well_filter=["A01", "A02", "B01", "B02"]
)
```

### Command-Line Stitching

```bash
# Basic stitching
ezstitcher /path/to/plate_folder --reference-channels 1

# Stitching with custom overlap
ezstitcher /path/to/plate_folder --reference-channels 1 --tile-overlap 15 --max-shift 75

# Stitching with well filtering
ezstitcher /path/to/plate_folder --reference-channels 1 --wells A01 A02 B01 B02
```

## 4.2 Z-Stack Processing

### Z-Stack Max Projection

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Create configuration for max projection
config = PipelineConfig(
    reference_channels=["1"],
    reference_flatten="max_projection",
    stitch_flatten="max_projection"
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/zstack_plate_folder")
```

### Z-Stack Best Focus

```python
from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Create configuration for best focus
config = PipelineConfig(
    reference_channels=["1"],
    reference_flatten="best_focus",
    stitch_flatten="best_focus",
    focus_method="combined",
    focus_config=FocusAnalyzerConfig(
        method="combined",
        roi=None  # Use entire image
    )
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/zstack_plate_folder")
```

### Z-Stack Per-Plane Stitching

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Create configuration for per-plane stitching
config = PipelineConfig(
    reference_channels=["1"],
    reference_flatten="max_projection",  # Use max projection for position generation
    stitch_flatten=None                  # Stitch each Z-plane separately
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/zstack_plate_folder")
```

### Custom Z-Stack Processing

```python
import numpy as np
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Define custom Z-stack processing function
def weighted_projection(stack):
    """Create a weighted projection that emphasizes middle planes."""
    if not stack:
        return None
    
    num_planes = len(stack)
    weights = np.ones(num_planes)
    
    # Emphasize middle planes
    middle = num_planes // 2
    for i in range(num_planes):
        weights[i] = 1.0 - 0.5 * abs(i - middle) / middle
    
    # Apply weights
    weighted_stack = np.array([stack[i] * weights[i] for i in range(num_planes)])
    return np.sum(weighted_stack, axis=0) / np.sum(weights)

# Create configuration with custom Z-stack processing
config = PipelineConfig(
    reference_channels=["1"],
    reference_flatten=weighted_projection,
    stitch_flatten=weighted_projection
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/zstack_plate_folder")
```

## 4.3 Custom Preprocessing

### Image Normalization

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.image_preprocessor import ImagePreprocessor

# Create configuration with normalization
config = PipelineConfig(
    reference_channels=["1"],
    reference_processing=ImagePreprocessor.normalize
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/plate_folder")
```

### Background Subtraction

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.image_preprocessor import ImagePreprocessor

# Create configuration with background subtraction
config = PipelineConfig(
    reference_channels=["1"],
    reference_processing=lambda img: ImagePreprocessor.background_subtract(img, radius=100)
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/plate_folder")
```

### Contrast Enhancement

```python
import numpy as np
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Define custom contrast enhancement function
def enhance_contrast(image):
    """Enhance contrast using percentile normalization."""
    p_low, p_high = np.percentile(image, (2, 98))
    return np.clip((image - p_low) * (65535 / (p_high - p_low)), 0, 65535).astype(np.uint16)

# Create configuration with contrast enhancement
config = PipelineConfig(
    reference_channels=["1"],
    reference_processing=enhance_contrast
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/plate_folder")
```

### Custom Preprocessing Functions

```python
import numpy as np
from scipy import ndimage
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Define multiple custom preprocessing functions
def denoise(image):
    """Apply Gaussian denoising."""
    return ndimage.gaussian_filter(image, sigma=1)

def sharpen(image):
    """Apply unsharp masking for sharpening."""
    blurred = ndimage.gaussian_filter(image, sigma=1)
    return np.clip(image * 1.5 - blurred * 0.5, 0, 65535).astype(np.uint16)

def enhance_edges(image):
    """Enhance edges using Sobel filter."""
    sobel_x = ndimage.sobel(image, axis=0)
    sobel_y = ndimage.sobel(image, axis=1)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.clip(image + magnitude * 0.2, 0, 65535).astype(np.uint16)

# Create configuration with channel-specific preprocessing
config = PipelineConfig(
    reference_channels=["1", "2", "3"],
    reference_processing={
        "1": denoise,
        "2": sharpen,
        "3": enhance_edges
    }
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/plate_folder")
```

## 4.4 Custom Focus Detection

### ROI-Based Focus Detection

```python
from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Create configuration with ROI-based focus detection
config = PipelineConfig(
    reference_channels=["1"],
    reference_flatten="best_focus",
    stitch_flatten="best_focus",
    focus_method="combined",
    focus_config=FocusAnalyzerConfig(
        method="combined",
        roi=(100, 100, 200, 200)  # (x, y, width, height)
    )
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/zstack_plate_folder")
```

### Custom Focus Metrics

```python
import numpy as np
from scipy import ndimage
from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.focus_analyzer import FocusAnalyzer

# Create a custom focus analyzer with a new metric
class CustomFocusAnalyzer(FocusAnalyzer):
    def __init__(self, config=None):
        super().__init__(config)
    
    def gradient_magnitude_variance(self, image):
        """Calculate gradient magnitude variance as a focus measure."""
        grad_x = ndimage.sobel(image, axis=0)
        grad_y = ndimage.sobel(image, axis=1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.var(magnitude)
    
    def custom_combined_focus(self, image):
        """Custom combined focus measure."""
        nvar = self.normalized_variance(image)
        lap = self.laplacian_energy(image)
        grad = self.gradient_magnitude_variance(image)
        
        # Custom weighting
        return 0.3 * nvar + 0.3 * lap + 0.4 * grad

# Create configuration with custom focus detection
config = PipelineConfig(
    reference_channels=["1"],
    reference_flatten="best_focus",
    stitch_flatten="best_focus",
    focus_method="custom_combined_focus",  # Use our custom method
    focus_config=FocusAnalyzerConfig(
        method="custom_combined_focus"
    )
)

# Create and run pipeline with custom focus analyzer
pipeline = PipelineOrchestrator(config)
pipeline.focus_analyzer = CustomFocusAnalyzer(config.focus_config)
pipeline.run("path/to/zstack_plate_folder")
```

### Focus Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.file_system_manager import FileSystemManager

# Load a Z-stack
fs_manager = FileSystemManager()
z_stack_dir = "path/to/zstack_folder"
z_stack_files = sorted(fs_manager.list_image_files(z_stack_dir))
z_stack = [fs_manager.load_image(f) for f in z_stack_files]

# Create focus analyzer
focus_analyzer = FocusAnalyzer()

# Calculate focus scores for different methods
methods = ["normalized_variance", "laplacian", "tenengrad", "fft", "combined"]
scores = {}

for method in methods:
    _, focus_scores = focus_analyzer.find_best_focus(z_stack, method=method)
    scores[method] = [score for _, score in focus_scores]

# Normalize scores for comparison
for method in methods:
    max_score = max(scores[method])
    scores[method] = [score / max_score for score in scores[method]]

# Plot focus scores
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(scores[method], label=method)

plt.xlabel("Z-Plane")
plt.ylabel("Normalized Focus Score")
plt.title("Focus Scores Across Z-Stack")
plt.legend()
plt.grid(True)
plt.savefig("focus_scores.png")
plt.show()
```

### Multi-Metric Focus Detection

```python
import numpy as np
from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Create configuration with custom focus weights
config = PipelineConfig(
    reference_channels=["1"],
    reference_flatten="best_focus",
    stitch_flatten="best_focus",
    focus_method="combined",
    focus_config=FocusAnalyzerConfig(
        method="combined",
        weights={
            "nvar": 0.4,  # Normalized variance
            "lap": 0.3,   # Laplacian energy
            "ten": 0.2,   # Tenengrad variance
            "fft": 0.1    # FFT-based metric
        }
    )
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/zstack_plate_folder")
```

## 4.5 Advanced Configuration

### Configuration Presets

```python
from ezstitcher.core import process_plate_folder_with_config

# Process using a predefined configuration preset
process_plate_folder_with_config(
    'path/to/plate_folder',
    config_preset='z_stack_best_focus'
)

# Process using a different preset
process_plate_folder_with_config(
    'path/to/plate_folder',
    config_preset='z_stack_per_plane'
)

# Process using the high-resolution preset
process_plate_folder_with_config(
    'path/to/plate_folder',
    config_preset='high_resolution'
)
```

### Custom Configuration Files

```python
from ezstitcher.core.config import PipelineConfig, StitcherConfig, FocusAnalyzerConfig
import json

# Create a custom configuration
config = PipelineConfig(
    reference_channels=["1", "2"],
    well_filter=["A01", "A02"],
    stitcher=StitcherConfig(
        tile_overlap=15.0,
        max_shift=75,
        margin_ratio=0.15
    ),
    focus_config=FocusAnalyzerConfig(
        method="laplacian",
        roi=(100, 100, 200, 200)
    ),
    reference_flatten="max_projection",
    stitch_flatten="best_focus",
    additional_projections=["max", "mean"]
)

# Save to JSON
with open("my_config.json", "w") as f:
    json.dump(config.__dict__, f, indent=2)

# Process using the configuration file
from ezstitcher.core import process_plate_folder_with_config

process_plate_folder_with_config(
    'path/to/plate_folder',
    config_file='my_config.json'
)
```

### Configuration Inheritance

```python
from ezstitcher.core.config import PipelineConfig, StitcherConfig

# Create a base configuration
base_config = PipelineConfig(
    reference_channels=["1"],
    stitcher=StitcherConfig(
        tile_overlap=10.0,
        max_shift=50
    )
)

# Create a derived configuration
derived_config = PipelineConfig(
    **base_config.__dict__,  # Inherit all base config properties
    reference_channels=["1", "2"],  # Override reference channels
    well_filter=["A01", "A02"]      # Add well filter
)

# Use the derived configuration
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

pipeline = PipelineOrchestrator(derived_config)
pipeline.run("path/to/plate_folder")
```

### Dynamic Configuration

```python
import numpy as np
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.image_locator import ImageLocator

# Function to analyze image and determine optimal parameters
def analyze_and_configure(plate_folder):
    """Analyze images and create optimal configuration."""
    fs_manager = FileSystemManager()
    image_dir = ImageLocator.find_image_directory(plate_folder)
    
    # Find sample image
    sample_images = fs_manager.list_image_files(image_dir, recursive=True)
    if not sample_images:
        return PipelineConfig()  # Default config if no images found
    
    # Load sample image
    sample_image = fs_manager.load_image(sample_images[0])
    if sample_image is None:
        return PipelineConfig()  # Default config if image loading fails
    
    # Analyze image properties
    mean_intensity = np.mean(sample_image)
    std_intensity = np.std(sample_image)
    
    # Determine optimal parameters based on image properties
    if std_intensity / mean_intensity < 0.2:
        # Low contrast image - use contrast enhancement
        from ezstitcher.core.image_preprocessor import ImagePreprocessor
        preprocessing_func = ImagePreprocessor.equalize_histogram
    else:
        # Normal contrast - use background subtraction
        from ezstitcher.core.image_preprocessor import ImagePreprocessor
        preprocessing_func = lambda img: ImagePreprocessor.background_subtract(img, radius=50)
    
    # Create configuration with dynamic parameters
    config = PipelineConfig(
        reference_channels=["1"],
        reference_processing=preprocessing_func
    )
    
    return config

# Analyze images and create configuration
config = analyze_and_configure("path/to/plate_folder")

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/plate_folder")
```

## 4.6 Opera Phenix Examples

### Opera Phenix File Structure

```
plate_folder/
├── Images/
│   ├── 0101K1F1P1R1.tiff  # Well A01, Channel 1, Field 1, Plane 1, Round 1
│   ├── 0101K1F1P2R1.tiff  # Well A01, Channel 1, Field 1, Plane 2, Round 1
│   ├── 0101K1F2P1R1.tiff  # Well A01, Channel 1, Field 2, Plane 1, Round 1
│   └── ...
├── Index.xml
└── ...
```

### Opera Phenix Metadata

```python
from ezstitcher.microscopes.opera_phenix import OperaPhenixMetadataHandler
from pathlib import Path

# Create metadata handler
metadata_handler = OperaPhenixMetadataHandler()

# Find metadata file
plate_path = Path("path/to/opera_phenix_plate")
metadata_file = metadata_handler.find_metadata_file(plate_path)
print(f"Metadata file: {metadata_file}")

# Get grid dimensions
grid_dimensions = metadata_handler.get_grid_dimensions(plate_path)
print(f"Grid dimensions: {grid_dimensions}")

# Get pixel size
pixel_size = metadata_handler.get_pixel_size(plate_path)
print(f"Pixel size: {pixel_size} µm")
```

### Opera Phenix Stitching

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Create configuration for Opera Phenix
config = PipelineConfig(
    reference_channels=["1"],  # Channel 1 (K1 in Opera Phenix)
    stitcher=StitcherConfig(
        tile_overlap=10.0,
        max_shift=50
    )
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/opera_phenix_plate")
```

### Opera Phenix Z-Stacks

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Create configuration for Opera Phenix Z-stacks
config = PipelineConfig(
    reference_channels=["1"],  # Channel 1 (K1 in Opera Phenix)
    reference_flatten="max_projection",
    stitch_flatten="best_focus",
    focus_method="combined"
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/opera_phenix_zstack_plate")
```

## 4.7 ImageXpress Examples

### ImageXpress File Structure

```
plate_folder/
├── TimePoint_1/
│   ├── A01_s1_w1.tif  # Well A01, Site 1, Channel 1
│   ├── A01_s1_w2.tif  # Well A01, Site 1, Channel 2
│   ├── A01_s2_w1.tif  # Well A01, Site 2, Channel 1
│   └── ...
└── ...

# Z-stack structure
plate_folder/
├── TimePoint_1/
│   ├── ZStep_1/
│   │   ├── A01_s1_w1.tif
│   │   └── ...
│   ├── ZStep_2/
│   │   ├── A01_s1_w1.tif
│   │   └── ...
│   └── ...
└── ...
```

### ImageXpress Metadata

```python
from ezstitcher.microscopes.imagexpress import ImageXpressMetadataHandler
from pathlib import Path

# Create metadata handler
metadata_handler = ImageXpressMetadataHandler()

# Find metadata file
plate_path = Path("path/to/imagexpress_plate")
metadata_file = metadata_handler.find_metadata_file(plate_path)
print(f"Metadata file: {metadata_file}")

# Get grid dimensions
grid_dimensions = metadata_handler.get_grid_dimensions(plate_path)
print(f"Grid dimensions: {grid_dimensions}")

# Get pixel size
pixel_size = metadata_handler.get_pixel_size(plate_path)
print(f"Pixel size: {pixel_size} µm")
```

### ImageXpress Stitching

```python
from ezstitcher.core.config import PipelineConfig, StitcherConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Create configuration for ImageXpress
config = PipelineConfig(
    reference_channels=["1"],
    stitcher=StitcherConfig(
        tile_overlap=10.0,
        max_shift=50
    )
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/imagexpress_plate")
```

### ImageXpress Z-Stacks

```python
from ezstitcher.core.config import PipelineConfig
from ezstitcher.core.processing_pipeline import PipelineOrchestrator

# Create configuration for ImageXpress Z-stacks
config = PipelineConfig(
    reference_channels=["1"],
    reference_flatten="max_projection",
    stitch_flatten="best_focus",
    focus_method="combined"
)

# Create and run pipeline
pipeline = PipelineOrchestrator(config)
pipeline.run("path/to/imagexpress_zstack_plate")
```
