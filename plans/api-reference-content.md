# API Reference Content Plan

## Status: In Progress
## Progress: 0%
## Last Updated: 2024-05-15
## Dependencies: [plans/documentation-outline.md]

This document outlines the detailed content for the API Reference section of the EZStitcher documentation.

## 3.1 Core Classes

### PipelineOrchestrator

```python
class PipelineOrchestrator:
    """
    A robust pipeline orchestrator for microscopy image processing.

    The pipeline follows a clear, linear flow:
    1. Load and organize images
    2. Process tiles (per well, per site, per channel)
    3. Select or compose channels
    4. Flatten Z-stacks (if present)
    5. Generate stitching positions
    6. Stitch images
    """

    def __init__(self, config: PipelineConfig):
        """Initialize with configuration."""
        pass

    def run(self, plate_folder):
        """
        Process a plate through the complete pipeline.

        Args:
            plate_folder: Path to the plate folder

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    def process_well(self, well, wavelength_patterns, wavelength_patterns_z, dirs):
        """
        Process a single well through the pipeline.

        Args:
            well: Well identifier
            wavelength_patterns: Dictionary mapping wavelengths to varying site patterns
            wavelength_patterns_z: Dictionary mapping wavelengths to varying z_index patterns
            dirs: Dictionary of directories
        """
        pass

    # Additional methods...
```

### Stitcher

```python
class Stitcher:
    """
    Class for handling image stitching operations.
    """

    def __init__(self, config: Optional[StitcherConfig] = None, filename_parser: Optional[FilenameParser] = None):
        """
        Initialize the Stitcher.

        Args:
            config (StitcherConfig): Configuration for stitching
            filename_parser (FilenameParser): Parser for microscopy filenames
        """
        pass

    def generate_positions(self, image_dir, image_pattern, output_path, grid_size_x, grid_size_y, 
                          overlap=10.0, pixel_size=1.0, max_shift=50):
        """
        Generate subpixel positions for stitching.

        Args:
            image_dir: Directory containing image tiles
            image_pattern: Pattern for image filenames
            output_path: Path to save positions CSV
            grid_size_x: Number of tiles in X direction
            grid_size_y: Number of tiles in Y direction
            overlap: Percentage overlap between tiles
            pixel_size: Pixel size in micrometers
            max_shift: Maximum allowed shift in pixels

        Returns:
            tuple: (success, pattern_used)
        """
        pass

    def assemble_image(self, positions_path, images_dir, output_path, override_names=None):
        """
        Assemble a stitched image using subpixel positions from a CSV file.

        Args:
            positions_path: Path to the CSV with subpixel positions
            images_dir: Directory containing image tiles
            output_path: Path to save final stitched image
            override_names: Optional list of filenames to use instead of those in CSV

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    # Additional methods...
```

### FocusAnalyzer

```python
class FocusAnalyzer:
    """
    Class for analyzing focus quality in microscopy images.
    """

    def __init__(self, config: Optional[FocusAnalyzerConfig] = None):
        """
        Initialize the FocusAnalyzer.

        Args:
            config (FocusAnalyzerConfig): Configuration for focus analysis
        """
        pass

    def normalized_variance(self, image):
        """
        Calculate normalized variance as a focus measure.

        Args:
            image: Input image

        Returns:
            float: Focus score
        """
        pass

    def laplacian_energy(self, image):
        """
        Calculate Laplacian energy as a focus measure.

        Args:
            image: Input image

        Returns:
            float: Focus score
        """
        pass

    def tenengrad_variance(self, image):
        """
        Calculate Tenengrad variance as a focus measure.

        Args:
            image: Input image

        Returns:
            float: Focus score
        """
        pass

    def adaptive_fft_focus(self, image):
        """
        Calculate FFT-based focus measure.

        Args:
            image: Input image

        Returns:
            float: Focus score
        """
        pass

    def combined_focus_measure(self, image):
        """
        Calculate combined focus measure using multiple metrics.

        Args:
            image: Input image

        Returns:
            float: Focus score
        """
        pass

    def find_best_focus(self, image_stack, method='combined', roi=None):
        """
        Find the best focused image in a stack using specified method.

        Args:
            image_stack: List of images
            method: Focus detection method
            roi: Optional region of interest as (x, y, width, height)

        Returns:
            tuple: (best_focus_index, focus_scores)
        """
        pass

    def select_best_focus(self, image_stack, method='combined', roi=None):
        """
        Select the best focus plane from a stack of images.

        Args:
            image_stack: List of images
            method: Focus detection method
            roi: Optional region of interest as (x, y, width, height)

        Returns:
            tuple: (best_focus_image, best_focus_index, focus_scores)
        """
        pass

    # Additional methods...
```

### ImagePreprocessor

```python
class ImagePreprocessor:
    """
    Handles image normalization, filtering, and compositing.
    All methods are static and do not require an instance.
    """

    @staticmethod
    def preprocess(image, channel, preprocessing_funcs=None):
        """
        Apply preprocessing to a single image for a given channel.

        Args:
            image: Input image
            channel: Channel identifier
            preprocessing_funcs: Dictionary mapping channels to preprocessing functions

        Returns:
            Processed image
        """
        pass

    @staticmethod
    def blur(image, sigma=1):
        """
        Apply Gaussian blur to an image.

        Args:
            image: Input image
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Blurred image
        """
        pass

    @staticmethod
    def normalize(image, target_min=0, target_max=65535):
        """
        Normalize image to specified range.

        Args:
            image: Input image
            target_min: Target minimum value
            target_max: Target maximum value

        Returns:
            Normalized image
        """
        pass

    @staticmethod
    def equalize_histogram(image):
        """
        Apply histogram equalization to an image.

        Args:
            image: Input image

        Returns:
            Equalized image
        """
        pass

    @staticmethod
    def background_subtract(image, radius=50):
        """
        Subtract background from an image.

        Args:
            image: Input image
            radius: Radius for background estimation

        Returns:
            Background-subtracted image
        """
        pass

    @staticmethod
    def create_composite(images, weights=None):
        """
        Create a composite image from multiple channels.

        Args:
            images: List of images
            weights: Dictionary mapping channel indices to weights

        Returns:
            Composite image
        """
        pass

    @staticmethod
    def max_projection(stack):
        """
        Create a maximum intensity projection from a Z-stack.

        Args:
            stack: Stack of images

        Returns:
            Maximum intensity projection
        """
        pass

    @staticmethod
    def mean_projection(stack):
        """
        Create a mean intensity projection from a Z-stack.

        Args:
            stack: Stack of images

        Returns:
            Mean intensity projection
        """
        pass

    # Additional methods...
```

### FileSystemManager

```python
class FileSystemManager:
    """
    Manages file system operations for ezstitcher.
    Abstracts away direct file system interactions for improved testability.
    """

    @staticmethod
    def ensure_directory(directory):
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            directory: Directory path to ensure exists

        Returns:
            Path object for the directory
        """
        pass

    @staticmethod
    def list_image_files(directory, extensions=None, recursive=False, flatten=False):
        """
        List all image files in a directory with specified extensions.

        Args:
            directory: Directory to search
            extensions: List of file extensions to include
            recursive: Whether to search recursively
            flatten: Whether to flatten Z-stack directories (implies recursive)

        Returns:
            List of Path objects for image files
        """
        pass

    @staticmethod
    def load_image(file_path, as_gray=True):
        """
        Load an image from disk.

        Args:
            file_path: Path to the image file
            as_gray: Whether to convert to grayscale

        Returns:
            Loaded image as numpy array, or None if loading fails
        """
        pass

    @staticmethod
    def save_image(file_path, image, compression=None):
        """
        Save an image to disk.

        Args:
            file_path: Path to save the image
            image: Image to save
            compression: Compression method

        Returns:
            True if successful, False otherwise
        """
        pass

    @staticmethod
    def detect_zstack_folders(directory):
        """
        Detect if a directory contains Z-stack folders.

        Args:
            directory: Directory to check

        Returns:
            tuple: (has_zstack_folders, zstack_folders)
        """
        pass

    @staticmethod
    def organize_zstack_folders(directory, filename_parser=None):
        """
        Organize Z-stack folders for consistent processing.

        Args:
            directory: Directory containing Z-stack folders
            filename_parser: Optional filename parser

        Returns:
            True if successful, False otherwise
        """
        pass

    # Additional methods...
```

### ImageLocator

```python
class ImageLocator:
    """
    Locates images in various directory structures.
    """

    @staticmethod
    def find_images_in_directory(directory, extensions=None, recursive=True, filename_parser=None):
        """
        Find all images in a directory.

        Args:
            directory: Directory to search
            extensions: List of file extensions to include
            recursive: Whether to search recursively
            filename_parser: Optional filename parser

        Returns:
            List of Path objects for image files
        """
        pass

    @staticmethod
    def find_image_locations(plate_folder, extensions=None):
        """
        Find all image files recursively within plate_folder.

        Args:
            plate_folder: Path to the plate folder
            extensions: List of file extensions to include

        Returns:
            Dictionary with all images found in the plate folder
        """
        pass

    @staticmethod
    def find_image_directory(plate_folder, extensions=None):
        """
        Find the directory where images are actually located.

        Args:
            plate_folder: Base directory to search
            extensions: List of file extensions to include

        Returns:
            Path to the directory containing images
        """
        pass

    @staticmethod
    def find_z_stack_dirs(directory, pattern=None):
        """
        Find Z-stack directories within a directory.

        Args:
            directory: Directory to search
            pattern: Regex pattern for Z-stack directories

        Returns:
            List of (z_index, directory) tuples
        """
        pass

    # Additional methods...
```

## 3.2 Configuration Classes

### PipelineConfig

```python
@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""
    # Input/output configuration
    processed_dir_suffix: str = "_processed"
    post_processed_dir_suffix: str = "_post_processed"
    positions_dir_suffix: str = "_positions"
    stitched_dir_suffix: str = "_stitched"

    # Well filtering
    well_filter: Optional[List[str]] = None

    # Reference processing (for position generation)
    reference_channels: List[str] = field(default_factory=lambda: ["1"])
    reference_processing: Optional[Union[Callable, List[Callable], Dict[str, Union[Callable, List[Callable]]]]] = None
    reference_composite_weights: Optional[Dict[str, float]] = None

    # Final processing (for stitched output)
    final_processing: Optional[Dict[str, Callable]] = None

    # Stitching configuration
    stitcher: StitcherConfig = field(default_factory=StitcherConfig)

    # Z-stack processing configuration
    reference_flatten: Union[str, Callable[[List[Any]], Any]] = "max_projection"
    stitch_flatten: Optional[Union[str, Callable[[List[Any]], Any]]] = None
    save_reference: bool = True
    additional_projections: Optional[List[str]] = None
    focus_method: str = "combined"
    focus_config: FocusAnalyzerConfig = field(default_factory=FocusAnalyzerConfig)
```

### StitcherConfig

```python
@dataclass
class StitcherConfig:
    """Configuration for the Stitcher class."""
    # Stitching parameters
    tile_overlap: float = 10.0
    max_shift: int = 50
    margin_ratio: float = 0.1
    pixel_size: Optional[float] = None
    grid_size: Optional[Tuple[int, int]] = None
```

### FocusAnalyzerConfig

```python
@dataclass
class FocusAnalyzerConfig:
    """Configuration for the FocusAnalyzer class."""
    method: str = "combined"
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    weights: Optional[Dict[str, float]] = None
```

### ImagePreprocessorConfig

```python
@dataclass
class ImagePreprocessorConfig:
    """Configuration for the ImagePreprocessor class."""
    preprocessing_funcs: Dict[str, Callable] = field(default_factory=dict)
    composite_weights: Optional[Dict[str, float]] = None
```

## 3.3 Microscope Handlers

### MicroscopeHandler

```python
class MicroscopeHandler:
    """Composed class for handling microscope-specific functionality."""

    def __init__(self, plate_folder=None, parser=None, metadata_handler=None, microscope_type='auto'):
        """Initialize with plate folder and optional components."""
        pass

    def parse_filename(self, filename):
        """Delegate to parser."""
        pass

    def get_components(self, filename):
        """Delegate to parser."""
        pass

    def get_grid_dimensions(self, plate_path):
        """Delegate to metadata handler."""
        pass

    def get_pixel_size(self, plate_path):
        """Delegate to metadata handler."""
        pass

    # Additional methods...
```

### FilenameParser (ABC)

```python
class FilenameParser(ABC):
    """
    Abstract base class for parsing microscopy filenames.
    
    This class defines the interface for parsing filenames from different microscope types.
    """
    
    @abstractmethod
    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse a filename into its components.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Dictionary of components, or None if parsing fails
        """
        pass
    
    @abstractmethod
    def get_components(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get components from a filename.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Dictionary of components, or None if parsing fails
        """
        pass
    
    def path_list_from_pattern(self, folder_path: Union[str, Path],
                              pattern: str) -> List[Path]:
        """
        Get a list of paths matching a pattern.
        
        Args:
            folder_path: Folder to search
            pattern: Pattern to match
            
        Returns:
            List of matching paths
        """
        pass
    
    def auto_detect_patterns(self, folder_path: Union[str, Path],
                            extensions: Optional[List[str]] = None,
                            group_by: Optional[str] = None,
                            variable_components: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Auto-detect patterns in a folder.
        
        Args:
            folder_path: Folder to search
            extensions: List of file extensions to include
            group_by: Component to group by (e.g., 'well')
            variable_components: Components that can vary (e.g., ['site'])
            
        Returns:
            Dictionary mapping wells to patterns
        """
        pass
```

### MetadataHandler (ABC)

```python
class MetadataHandler(ABC):
    """
    Abstract base class for handling microscope metadata.
    
    This class defines the interface for extracting metadata from different microscope types.
    """
    
    @abstractmethod
    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """
        Find the metadata file for a plate.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            Path to the metadata file, or None if not found
        """
        pass
    
    @abstractmethod
    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Optional[Tuple[int, int]]:
        """
        Get the grid dimensions from metadata.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            Tuple of (grid_size_x, grid_size_y), or None if not available
        """
        pass
    
    @abstractmethod
    def get_pixel_size(self, plate_path: Union[str, Path]) -> Optional[float]:
        """
        Get the pixel size from metadata.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            Pixel size in micrometers, or None if not available
        """
        pass
```

### ImageXpressFilenameParser

```python
class ImageXpressFilenameParser(FilenameParser):
    """
    Filename parser for ImageXpress microscopes.
    
    Handles parsing filenames like 'A01_s1_w1.tif'.
    """
    
    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse an ImageXpress filename into its components.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Dictionary of components, or None if parsing fails
        """
        pass
    
    def get_components(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get components from an ImageXpress filename.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Dictionary of components, or None if parsing fails
        """
        pass
```

### ImageXpressMetadataHandler

```python
class ImageXpressMetadataHandler(MetadataHandler):
    """
    Metadata handler for ImageXpress microscopes.
    
    Handles finding and parsing HTD files for ImageXpress microscopes.
    """
    
    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """
        Find the HTD file for an ImageXpress plate.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            Path to the HTD file, or None if not found
        """
        pass
    
    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Optional[Tuple[int, int]]:
        """
        Get the grid dimensions from an ImageXpress HTD file.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            Tuple of (grid_size_x, grid_size_y), or None if not available
        """
        pass
    
    def get_pixel_size(self, plate_path: Union[str, Path]) -> Optional[float]:
        """
        Get the pixel size from an ImageXpress HTD file.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            Pixel size in micrometers, or None if not available
        """
        pass
```

### OperaPhenixFilenameParser

```python
class OperaPhenixFilenameParser(FilenameParser):
    """
    Filename parser for Opera Phenix microscopes.
    
    Handles parsing filenames like '0101K1F1P1R1.tiff'.
    """
    
    def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse an Opera Phenix filename into its components.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Dictionary of components, or None if parsing fails
        """
        pass
    
    def get_components(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get components from an Opera Phenix filename.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Dictionary of components, or None if parsing fails
        """
        pass
```

### OperaPhenixMetadataHandler

```python
class OperaPhenixMetadataHandler(MetadataHandler):
    """
    Metadata handler for Opera Phenix microscopes.
    
    Handles finding and parsing Index.xml files for Opera Phenix microscopes.
    """
    
    def find_metadata_file(self, plate_path: Union[str, Path]) -> Optional[Path]:
        """
        Find the Index.xml file for an Opera Phenix plate.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            Path to the Index.xml file, or None if not found
        """
        pass
    
    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> Optional[Tuple[int, int]]:
        """
        Get the grid dimensions from an Opera Phenix Index.xml file.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            Tuple of (grid_size_x, grid_size_y), or None if not available
        """
        pass
    
    def get_pixel_size(self, plate_path: Union[str, Path]) -> Optional[float]:
        """
        Get the pixel size from an Opera Phenix Index.xml file.
        
        Args:
            plate_path: Path to the plate folder
            
        Returns:
            Pixel size in micrometers, or None if not available
        """
        pass
```

## 3.4 Utility Classes

### Pattern Matching

```python
def create_pattern_from_components(components, variable_components=None):
    """
    Create a filename pattern from components.
    
    Args:
        components: Dictionary of components
        variable_components: List of components that can vary
        
    Returns:
        Pattern string
    """
    pass

def extract_components_from_pattern(pattern):
    """
    Extract components from a pattern.
    
    Args:
        pattern: Pattern string
        
    Returns:
        Dictionary of components
    """
    pass

def match_pattern(pattern, filename):
    """
    Match a pattern against a filename.
    
    Args:
        pattern: Pattern string
        filename: Filename to match
        
    Returns:
        Dictionary of matched components, or None if no match
    """
    pass
```

### File Operations

```python
def find_file_recursive(directory, filename):
    """
    Find a file recursively in a directory.
    
    Args:
        directory: Directory to search
        filename: Filename to find
        
    Returns:
        Path to the file, or None if not found
    """
    pass

def copy_file(source, destination):
    """
    Copy a file.
    
    Args:
        source: Source file
        destination: Destination file
        
    Returns:
        True if successful, False otherwise
    """
    pass

def move_file(source, destination):
    """
    Move a file.
    
    Args:
        source: Source file
        destination: Destination file
        
    Returns:
        True if successful, False otherwise
    """
    pass
```

### Image Operations

```python
def create_linear_weight_mask(shape, margin_ratio=0.1):
    """
    Create a linear weight mask for blending images.
    
    Args:
        shape: Shape of the mask (height, width)
        margin_ratio: Ratio of image size to use as margin
        
    Returns:
        Weight mask
    """
    pass

def apply_weight_mask(image, mask):
    """
    Apply a weight mask to an image.
    
    Args:
        image: Input image
        mask: Weight mask
        
    Returns:
        Masked image
    """
    pass

def blend_images(image1, image2, mask1, mask2):
    """
    Blend two images using weight masks.
    
    Args:
        image1: First image
        image2: Second image
        mask1: Weight mask for first image
        mask2: Weight mask for second image
        
    Returns:
        Blended image
    """
    pass
```
