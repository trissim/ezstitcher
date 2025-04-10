# Phase 1: Update References to Deprecated Code

This document outlines the specific changes needed in Phase 1 of the refactoring plan, which focuses on updating references to deprecated code.

## 1. DirectoryManager Replacement

### Files to Modify:
- `ezstitcher/core/file_system_manager.py`
- `ezstitcher/core/plate_processor.py`
- `ezstitcher/core/zstack_processor.py`
- `ezstitcher/core/zstack_organizer.py`
- `ezstitcher/core/zstack_projector.py`
- `ezstitcher/core/zstack_focus_manager.py`
- `ezstitcher/core/zstack_stitcher.py`

### Changes:

#### In FileSystemManager:

```python
# Replace
def find_wells(self, timepoint_dir):
    return self.directory_manager.find_wells(timepoint_dir)

# With
def find_wells(self, timepoint_dir):
    """
    Find all wells in the timepoint directory.
    
    Deprecated: Use initialize_dir_structure(plate_folder).get_wells() instead.
    
    Args:
        timepoint_dir (str or Path): Path to the TimePoint_1 directory
        
    Returns:
        list: List of well names
    """
    warnings.warn(
        "FileSystemManager.find_wells() is deprecated. Use initialize_dir_structure(plate_folder).get_wells() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Initialize directory structure manager for the parent directory
    dir_structure = self.initialize_dir_structure(Path(timepoint_dir).parent)
    
    # Return wells
    return dir_structure.get_wells()
```

#### In PlateProcessor:

```python
# Replace
wells = self.fs_manager.find_wells(timepoint_path)

# With
dir_structure = self.fs_manager.initialize_dir_structure(plate_folder)
wells = dir_structure.get_wells()
```

## 2. ZStackProcessor Refactoring

### Changes:

#### In ZStackProcessor:

```python
# Replace
timepoint_path = plate_path / "TimePoint_1"
if timepoint_path.exists():
    has_zstack_images, z_indices_map = self.detect_zstack_images(timepoint_path)
else:
    has_zstack_images = False
    z_indices_map = {}

# With
dir_structure = self.fs_manager.initialize_dir_structure(plate_path)
timepoint_path = dir_structure.get_timepoint_dir()
if timepoint_path:
    has_zstack_images, z_indices_map = self.detect_zstack_images(timepoint_path)
else:
    has_zstack_images = False
    z_indices_map = {}
```

## 3. Path Operations Replacement

### Changes:

#### Replace direct Path operations with FileSystemManager methods:

```python
# Replace
directory = Path(directory)
directory.mkdir(parents=True, exist_ok=True)

# With
directory = self.fs_manager.ensure_directory(directory)
```

```python
# Replace
for file in timepoint_dir.glob(f"*{ext}"):
    # Process file

# With
image_files = self.fs_manager.list_image_files(timepoint_dir)
for file in image_files:
    # Process file
```

## 4. ZStackProcessorConfig Deprecated Parameters

### Files to Modify:
- `ezstitcher/core/main.py`
- Any other files that create ZStackProcessorConfig instances

### Changes:

#### In main.py:

```python
# Replace
if any(param is not None for param in [stitch_z_reference, focus_detect, create_projections, save_projections, reference_method]):
    # Use deprecated parameters if provided
    zstack_config = ZStackProcessorConfig(
        # Deprecated parameters
        focus_detect=focus_detect,
        focus_method=focus_method,
        create_projections=create_projections,
        stitch_z_reference=stitch_z_reference,
        save_projections=save_projections,
        reference_method=reference_method,
        stitch_all_z_planes=stitch_all_z_planes
    )
else:
    # Use new parameters
    zstack_config = ZStackProcessorConfig(
        z_reference_function=z_reference_function,
        focus_method=focus_method,
        save_reference=save_reference,
        additional_projections=additional_projections,
        stitch_all_z_planes=stitch_all_z_planes
    )

# With
# Always use new parameters, let the config handle backward compatibility
zstack_config = ZStackProcessorConfig(
    z_reference_function=z_reference_function or stitch_z_reference or reference_method,
    focus_method=focus_method,
    save_reference=save_reference if save_reference is not None else (create_projections or save_projections),
    additional_projections=additional_projections or projection_types,
    stitch_all_z_planes=stitch_all_z_planes
)
```

## 5. Legacy Config Classes

### Files to Modify:
- `ezstitcher/core/config.py`
- Any files that use StitchingConfig or ZStackConfig

### Changes:

#### In config.py:

```python
# Add adapter methods to convert legacy configs to new configs

def convert_stitching_config_to_plate_processor_config(stitching_config):
    """
    Convert a legacy StitchingConfig to a PlateProcessorConfig.
    
    Args:
        stitching_config: Legacy StitchingConfig
        
    Returns:
        PlateProcessorConfig
    """
    # Create StitcherConfig
    stitcher_config = StitcherConfig(
        tile_overlap=stitching_config.tile_overlap,
        max_shift=stitching_config.max_shift,
        margin_ratio=stitching_config.margin_ratio
    )
    
    # Create FocusConfig
    focus_config = FocusConfig(
        method=stitching_config.focus_method
    )
    
    # Create ImagePreprocessorConfig
    image_config = ImagePreprocessorConfig(
        preprocessing_funcs=stitching_config.preprocessing_funcs,
        composite_weights=stitching_config.composite_weights
    )
    
    # Create ZStackProcessorConfig
    zstack_config = ZStackProcessorConfig(
        focus_detect=stitching_config.focus_detect,
        focus_method=stitching_config.focus_method,
        create_projections=stitching_config.create_projections,
        stitch_z_reference=stitching_config.stitch_z_reference,
        save_projections=stitching_config.save_projections,
        stitch_all_z_planes=stitching_config.stitch_all_z_planes
    )
    
    # Create PlateProcessorConfig
    plate_config = PlateProcessorConfig(
        reference_channels=stitching_config.reference_channels,
        well_filter=stitching_config.well_filter,
        stitcher=stitcher_config,
        focus_analyzer=focus_config,
        image_preprocessor=image_config,
        z_stack_processor=zstack_config
    )
    
    return plate_config
```

## 6. Utility Functions Replacement

### Files to Modify:
- `ezstitcher/core/utils.py`
- Any files that use utils.list_image_files

### Changes:

#### In utils.py:

```python
# Replace
def list_image_files(directory, extensions=None):
    """
    List all image files in a directory with specified extensions.
    
    Args:
        directory (str or Path): Directory to search
        extensions (list): List of file extensions to include (default: common image formats)
        
    Returns:
        list: List of Path objects for image files
    """
    if extensions is None:
        extensions = ['.tif', '.TIF', '.tiff', '.TIFF', '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
        
    directory = Path(directory)
    image_files = []
    
    for ext in extensions:
        image_files.extend(list(directory.glob(f"*{ext}")))
        
    return sorted(image_files)

# With
def list_image_files(directory, extensions=None):
    """
    List all image files in a directory with specified extensions.
    
    Deprecated: Use ImageLocator.find_images_in_directory() instead.
    
    Args:
        directory (str or Path): Directory to search
        extensions (list): List of file extensions to include (default: common image formats)
        
    Returns:
        list: List of Path objects for image files
    """
    warnings.warn(
        "utils.list_image_files() is deprecated. Use ImageLocator.find_images_in_directory() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    from ezstitcher.core.image_locator import ImageLocator
    return ImageLocator.find_images_in_directory(directory, extensions)
```

## 7. Testing Strategy

1. After each set of changes, run the tests to ensure they still pass:
   ```bash
   python -m pytest tests/test_directory_structure_manager.py
   python -m pytest tests/test_directory_structure_integration.py
   python -m pytest tests/test_synthetic_imagexpress_refactored.py
   python -m pytest tests/test_synthetic_opera_phenix_refactored.py
   ```

2. If tests fail, fix the issues before proceeding to the next set of changes.

3. Once all changes are complete, run the full test suite:
   ```bash
   python -m pytest
   ```
