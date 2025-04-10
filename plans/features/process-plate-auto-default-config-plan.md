# Plan: Auto-Detection for process_plate_auto with Default Config

Status: In Progress  
Progress: 0%  
Last Updated: 2023-07-10  
Dependencies: None

## 1. Problem Analysis

Currently, `process_plate_auto` fails when no config is provided because it tries to access attributes of a `None` config object. The error occurs specifically when trying to access `config.z_stack_processor` to initialize the `ZStackProcessor`.

The function should be able to:
1. Create a default config when none is provided
2. Auto-detect microscope type (Opera Phenix or ImageXpress)
3. Auto-detect if Z-stacks are present
4. Configure the appropriate settings based on these detections

**Constraints:**
- Must maintain backward compatibility
- Should work with both Opera Phenix and ImageXpress formats
- Should handle both Z-stack and non-Z-stack plates

**Edge Cases:**
- Empty plate folders
- Mixed format plates
- Unusual directory structures

## 2. High-Level Solution

1. Modify `process_plate_auto` to create a default `PlateProcessorConfig` when `config=None`
2. Implement auto-detection for microscope type before creating the `PlateProcessor`
3. Implement auto-detection for Z-stacks before deciding on the processing pipeline
4. Apply any provided overrides to the default config

## 3. Implementation Details

### 3.1 Default Config Creation

```python
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
    plate_folder = Path(plate_folder)
    
    # Create default config if none provided
    if config is None:
        config = PlateProcessorConfig()
        
    # Apply any overrides
    apply_nested_overrides(config, kwargs)
    
    # Auto-detect microscope type if set to 'auto'
    if config.microscope_type.lower() == 'auto':
        fs_manager = FileSystemManager()
        sample_files = fs_manager.list_image_files(plate_folder, extensions=['.tif', '.tiff', '.TIF', '.TIFF'])[:10]
        sample_files = [Path(f).name for f in sample_files]
        
        if sample_files:
            detected_format = FilenameParser.detect_format(sample_files)
            if detected_format:
                config.microscope_type = detected_format
                logging.info(f"Auto-detected microscope type: {detected_format}")
            else:
                # Default to ImageXpress if detection fails
                config.microscope_type = 'ImageXpress'
                logging.info("Could not auto-detect microscope type, defaulting to ImageXpress")
        else:
            # Default to ImageXpress if no files found
            config.microscope_type = 'ImageXpress'
            logging.info("No image files found, defaulting to ImageXpress")
    
    # Instantiate PlateProcessor
    processor = PlateProcessor(config)
    
    # Apply nested overrides again to internal component configs
    apply_nested_overrides(processor.stitcher.config, kwargs)
    apply_nested_overrides(processor.focus_analyzer.config, kwargs)
    apply_nested_overrides(processor.zstack_processor.config, kwargs)
    apply_nested_overrides(processor.image_preprocessor.config, kwargs)
    
    # Detect if Z-stacks are present
    timepoint_dir = plate_folder / config.timepoint_dir_name
    if not timepoint_dir.exists():
        timepoint_dir = plate_folder  # Fall back to plate folder if TimePoint_1 doesn't exist
    
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

### 3.2 Modify PlateProcessor.__init__ to Handle None Config

```python
def __init__(self, config: PlateProcessorConfig = None):
    self.config = config or PlateProcessorConfig()
    self.filename_parser = None  # Will be initialized in run() based on microscope_type
    self.fs_manager = FileSystemManager()
    self.zstack_processor = ZStackProcessor(self.config.z_stack_processor)
    self.focus_analyzer = FocusAnalyzer(self.config.focus_analyzer)
    self.image_preprocessor = ImagePreprocessor(self.config.image_preprocessor)
    self.stitcher = Stitcher(self.config.stitcher)
```

### 3.3 Ensure ZStackProcessor.__init__ Handles None Config

```python
def __init__(self, config: ZStackProcessorConfig = None, filename_parser=None, preprocessing_funcs=None):
    """
    Initialize the ZStackProcessor.
    
    Args:
        config: Configuration for Z-stack processing
        filename_parser: Parser for microscopy filenames
        preprocessing_funcs: Dictionary mapping channels to preprocessing functions
    """
    self.config = config or ZStackProcessorConfig()
    self.fs_manager = FileSystemManager()
    self._z_info = None
    self._z_indices = []
    self.preprocessing_funcs = preprocessing_funcs or {}
    
    # Initialize the filename parser
    self.filename_parser = filename_parser
    
    # Initialize the focus analyzer
    focus_config = self.config.focus_config or FocusConfig(method=self.config.focus_method)
    self.focus_analyzer = FocusAnalyzer(focus_config)
```

## 4. Validation

### 4.1 Unit Tests

1. Test with no config provided
2. Test with auto-detection of microscope type
3. Test with auto-detection of Z-stacks
4. Test with overrides applied to default config

### 4.2 Integration Tests

1. Test with real Opera Phenix data
2. Test with real ImageXpress data
3. Test with Z-stack and non-Z-stack data

## 5. References

- `ezstitcher/core/main.py` - `process_plate_auto` implementation
- `ezstitcher/core/plate_processor.py` - `PlateProcessor` class
- `ezstitcher/core/zstack_processor.py` - `ZStackProcessor` class
- `ezstitcher/core/filename_parser.py` - Microscope type detection
