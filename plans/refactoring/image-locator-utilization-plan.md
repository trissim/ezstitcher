# Image Locator Utilization Plan

Status: In Progress
Progress: 40%
Last Updated: 2023-07-11
Dependencies: None

## 1. Problem Analysis

The `ImageLocator` class was designed to centralize and standardize image file location operations across the codebase. However, it's currently underutilized, with many components implementing their own file location logic. This leads to:

1. **Code duplication**: Similar file location logic is repeated in multiple places
2. **Inconsistent behavior**: Different components may handle edge cases differently
3. **Reduced maintainability**: Changes to file location logic need to be made in multiple places
4. **Missed functionality**: The `ImageLocator` class has robust handling for various directory structures that isn't being leveraged

**Current Usage**:
- `DirectoryStructureManager` uses `ImageLocator` for finding sample files and detecting directory structure
- `utils.list_image_files()` is deprecated in favor of `ImageLocator.find_images_in_directory()`
- Several test files import `ImageLocator` but don't use it extensively

**Areas for Improvement**:
- `FileSystemManager.list_image_files()` duplicates `ImageLocator.find_images_in_directory()`
- `PlateProcessor._initialize_filename_parser_and_convert()` has custom file location logic
- `ZStackProcessor` and its component classes have their own file location logic
- `PatternMatcher.path_list_from_pattern()` has custom directory structure handling
- `main.py` has custom logic for finding the timepoint directory

## 2. High-Level Solution

1. **Refactor `FileSystemManager`** to use `ImageLocator` for all file location operations
2. **Update `PlateProcessor`** to use `ImageLocator` for file location and directory structure detection
3. **Refactor Z-stack components** to use `ImageLocator` consistently
4. **Enhance `PatternMatcher`** to use `ImageLocator` for directory structure handling
5. **Update `main.py`** to use `ImageLocator` for timepoint directory detection

## 3. Implementation Details

### 3.1 Refactor `FileSystemManager`

```python
# In FileSystemManager class
def list_image_files(self, directory: Union[str, Path],
                     extensions: Optional[List[str]] = None) -> List[Path]:
    """
    List all image files in a directory with specified extensions.

    Args:
        directory (str or Path): Directory to search
        extensions (list): List of file extensions to include

    Returns:
        list: List of Path objects for image files
    """
    from ezstitcher.core.image_locator import ImageLocator
    return ImageLocator.find_images_in_directory(directory, extensions)
```

### 3.2 Update `PlateProcessor`

```python
# In PlateProcessor._initialize_filename_parser_and_convert
def _initialize_filename_parser_and_convert(self, plate_path):
    config = self.config
    if config.microscope_type.lower() == 'auto':
        from ezstitcher.core.image_locator import ImageLocator

        # Use ImageLocator to find sample files across all possible locations
        image_locations = ImageLocator.find_image_locations(plate_path)
        sample_files = []

        # Collect sample files from all locations
        for location_type, images in image_locations.items():
            if location_type == 'z_stack':
                # Handle z_stack specially since it's a nested dictionary
                for z_index, z_images in images.items():
                    sample_files.extend([Path(f).name for f in z_images[:5]])
            else:
                sample_files.extend([Path(f).name for f in images[:5]])

        if not sample_files:
            logger.warning(f"No image files found in {plate_path}. Microscope type autodetection failed.")
            raise ValueError(f"Microscope type autodetection failed: no image files found in {plate_path}")

        self.filename_parser = detect_parser(sample_files)
        logger.info(f"Auto-detected microscope type: {self.filename_parser.__class__.__name__}")

        # Handle Opera Phenix conversion if needed
        if self.filename_parser.__class__.__name__ == 'OperaPhenixFilenameParser':
            logger.info(f"Converting Opera Phenix files to ImageXpress format...")
            # Use ImageLocator to find the appropriate directory
            if 'images' in image_locations:
                logger.info(f"Found Images directory, using it for Opera Phenix files")
                self.filename_parser.rename_all_files_in_directory(plate_path / "Images")
            else:
                self.filename_parser.rename_all_files_in_directory(plate_path)
    elif config.microscope_type.lower() == 'imagexpress':
        from ezstitcher.core.filename_parser import ImageXpressFilenameParser
        self.filename_parser = ImageXpressFilenameParser()
    elif config.microscope_type.lower() == 'operaphenix':
        from ezstitcher.core.filename_parser import OperaPhenixFilenameParser
        self.filename_parser = OperaPhenixFilenameParser()
    else:
        raise ValueError(f"Unsupported microscope type: {config.microscope_type}")
```

### 3.3 Refactor Z-stack Components

```python
# In ZStackProcessor.detect_zstack_images
def detect_zstack_images(self, folder_path):
    """
    Detect Z-stack images in a folder.

    Args:
        folder_path: Path to the folder

    Returns:
        tuple: (has_zstack, z_indices_map)
    """
    from ezstitcher.core.image_locator import ImageLocator

    folder_path = Path(folder_path)

    # Use ImageLocator to find all image locations
    image_locations = ImageLocator.find_image_locations(folder_path, self.config.timepoint_dir_name)

    # Check if there are Z-stack images
    if 'z_stack' in image_locations:
        z_stack_images = image_locations['z_stack']
        return True, z_stack_images

    # If no Z-stack directories, check for Z-index in filenames
    z_indices_map = defaultdict(list)

    # Check all image locations for Z-index in filenames
    for location_type, images in image_locations.items():
        if location_type != 'z_stack':  # Already handled above
            for img_path in images:
                metadata = self.filename_parser.parse_filename(str(img_path))
                if metadata and 'z_index' in metadata and metadata['z_index'] is not None:
                    # Create a base name without the Z-index
                    well = metadata['well']
                    site = metadata['site']
                    channel = metadata['channel']

                    # Create a consistent base name for grouping
                    base_name = f"{well}_s{site:03d}_w{channel}"

                    # Add the Z-index to the list for this base name
                    z_indices_map[base_name].append(metadata['z_index'])

    has_zstack = len(z_indices_map) > 0
    return has_zstack, dict(z_indices_map)
```

### 3.4 Enhance `PatternMatcher`

```python
# In PatternMatcher.path_list_from_pattern
def path_list_from_pattern(self, directory, pattern):
    """
    Get a list of filenames matching a pattern in a directory.

    Args:
        directory (str or Path): Directory to search
        pattern (str): Pattern to match with {iii} placeholder for site index

    Returns:
        list: List of matching filenames
    """
    from ezstitcher.core.image_locator import ImageLocator

    directory = Path(directory)

    # Handle substitution of {series} if present (from Ashlar)
    if "{series}" in pattern:
        pattern = pattern.replace("{series}", "{iii}")

    # Convert pattern to regex
    regex_pattern = pattern.replace('{iii}', '(\\d+)')

    # Handle _z001 suffix
    if '_z' not in regex_pattern:
        regex_pattern = regex_pattern.replace('_w1', '_w1(?:_z\\d+)?')

    # Handle .tif vs .tiff extension
    if regex_pattern.endswith('.tif'):
        regex_pattern = regex_pattern[:-4] + '(?:\.tif|\.tiff)'

    regex = re.compile(regex_pattern)

    # Use ImageLocator to find all possible image locations
    image_locations = ImageLocator.find_image_locations(directory)

    # Search for matching files in all locations
    matching_files = []

    # Check plate folder
    if 'plate' in image_locations:
        for file_path in image_locations['plate']:
            if regex.match(file_path.name):
                matching_files.append(file_path.name)

    # Check timepoint directory
    if 'timepoint' in image_locations and not matching_files:
        for file_path in image_locations['timepoint']:
            if regex.match(file_path.name):
                matching_files.append(file_path.name)

    # Check Images directory
    if 'images' in image_locations and not matching_files:
        for file_path in image_locations['images']:
            if regex.match(file_path.name):
                matching_files.append(file_path.name)

    return sorted(matching_files)
```

### 3.5 Update `main.py`

```python
# In process_plate_auto function
def process_plate_auto(
    plate_folder: str | Path,
    config: PlateProcessorConfig | None = None,
    **kwargs
) -> bool:
    """
    High-level function to process a plate folder.

    Automatically detects if the plate contains Z-stacks and runs the appropriate workflow
    using the modular OOP components internally.

    Args:
        plate_folder: Path to the plate folder.
        config: Optional PlateProcessorConfig. If None, a default config is created.
        **kwargs: Optional overrides for config parameters.

    Returns:
        True if processing succeeded, False otherwise.
    """
    from ezstitcher.core.image_locator import ImageLocator

    plate_folder = Path(plate_folder)

    # Create default config if none provided
    if config is None:
        config = PlateProcessorConfig()

    # Apply any overrides
    apply_nested_overrides(config, kwargs)

    # Instantiate PlateProcessor with auto-detection capabilities
    processor = PlateProcessor(config)

    # Apply nested overrides again to internal component configs
    apply_nested_overrides(processor.stitcher.config, kwargs)
    apply_nested_overrides(processor.focus_analyzer.config, kwargs)
    apply_nested_overrides(processor.zstack_processor.config, kwargs)
    apply_nested_overrides(processor.image_preprocessor.config, kwargs)

    # Use ImageLocator to find the timepoint directory
    timepoint_dir = ImageLocator.find_timepoint_dir(plate_folder, config.timepoint_dir_name)
    if not timepoint_dir:
        timepoint_dir = plate_folder  # Fall back to plate folder if TimePoint_1 doesn't exist

    # Detect if Z-stacks are present
    detector = ZStackProcessor(config.z_stack_processor)
    has_zstack, _ = detector.detect_zstack_images(timepoint_dir)

    if has_zstack:
        logging.info("Z-stacks detected. Running full Z-stack processing pipeline.")
        # Run full Z-stack pipeline
        success = processor.run(plate_folder)
    else:
        logging.info("No Z-stacks detected. Running standard 2D stitching pipeline.")
        # Adjust config to skip Z-stack steps
        config.z_stack_processor.stitch_all_z_planes = False
        processor = PlateProcessor(config)
        success = processor.run(plate_folder)

    return success
```

## 4. Validation

### 4.1 Unit Tests

1. Test that `FileSystemManager.list_image_files()` returns the same results as `ImageLocator.find_images_in_directory()`
2. Test that `PlateProcessor._initialize_filename_parser_and_convert()` correctly detects microscope type using `ImageLocator`
3. Test that `ZStackProcessor.detect_zstack_images()` correctly detects Z-stacks using `ImageLocator`
4. Test that `PatternMatcher.path_list_from_pattern()` correctly finds files using `ImageLocator`
5. Test that `process_plate_auto()` correctly finds the timepoint directory using `ImageLocator`

### 4.2 Integration Tests

1. Test with real Opera Phenix data
2. Test with real ImageXpress data
3. Test with Z-stack and non-Z-stack data
4. Test with various directory structures

## 5. Implementation Order

1. ✅ Refactor `FileSystemManager.list_image_files()` to use `ImageLocator`
2. ✅ Update `main.py` to use `ImageLocator` for timepoint directory detection
3. ✅ Refactor `PlateProcessor._initialize_filename_parser_and_convert()` to use `ImageLocator`
4. Refactor Z-stack components to use `ImageLocator`
5. Enhance `PatternMatcher.path_list_from_pattern()` to use `ImageLocator`

## 6. Benefits

1. **Reduced code duplication**: Centralized file location logic
2. **Consistent behavior**: Standardized handling of directory structures
3. **Improved maintainability**: Changes to file location logic only need to be made in one place
4. **Enhanced functionality**: Leveraging the robust directory structure handling of `ImageLocator`
5. **Better error handling**: Consistent error messages and logging

## 7. Risks and Mitigations

1. **Risk**: Changes might break existing functionality
   **Mitigation**: Comprehensive unit and integration tests

2. **Risk**: Performance impact from additional abstraction
   **Mitigation**: Profile before and after to ensure no significant performance degradation

3. **Risk**: Circular dependencies
   **Mitigation**: Careful import management, possibly using lazy imports

## 8. References

- `ezstitcher/core/image_locator.py` - `ImageLocator` class
- `ezstitcher/core/file_system_manager.py` - `FileSystemManager` class
- `ezstitcher/core/plate_processor.py` - `PlateProcessor` class
- `ezstitcher/core/zstack_processor.py` - `ZStackProcessor` class
- `ezstitcher/core/pattern_matcher.py` - `PatternMatcher` class
- `ezstitcher/core/main.py` - `process_plate_auto` function
