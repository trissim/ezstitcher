# ZStackProcessor Refactoring Plan

## Current Issues with ZStackProcessor:

1. **Size and Complexity**: The ZStackProcessor class is over 1200 lines long, making it difficult to maintain and understand.
2. **Responsibility Overlap**: It contains functionality that overlaps with other classes like FileSystemManager, FocusAnalyzer, and ImagePreprocessor.
3. **Inconsistent Abstraction**: Some operations are delegated to other classes, while similar operations are implemented directly.
4. **Code Duplication**: There are repeated patterns for file operations, parsing, and image processing.
5. **Mixed Concerns**: The class handles file operations, image processing, and Z-stack management all in one place.

## Detailed Refactoring Plan:

### 1. Extract Z-Stack Detection and Organization Logic

**Create a new `ZStackOrganizer` class:**
- Move `detect_zstack_folders`, `detect_zstack_images`, `organize_zstack_folders` methods
- This class will be responsible for detecting and organizing Z-stack folders and images

```python
class ZStackOrganizer:
    def __init__(self, config, filename_parser=None, fs_manager=None):
        self.config = config
        self.fs_manager = fs_manager or FileSystemManager()
        self.filename_parser = filename_parser
        
    def detect_zstack_folders(self, plate_folder):
        # Implementation from ZStackProcessor
        
    def detect_zstack_images(self, folder_path):
        # Implementation from ZStackProcessor
        
    def organize_zstack_folders(self, plate_folder):
        # Implementation from ZStackProcessor
```

### 2. Extract Z-Stack Projection Logic

**Create a new `ZStackProjector` class:**
- Move `create_zstack_projections` method
- Delegate to ImagePreprocessor for actual projection creation

```python
class ZStackProjector:
    def __init__(self, config, image_preprocessor=None, fs_manager=None):
        self.config = config
        self.fs_manager = fs_manager or FileSystemManager()
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        
    def create_projections(self, input_dir, output_dir, projection_types=None, preprocessing_funcs=None):
        # Implementation from ZStackProcessor.create_zstack_projections
```

### 3. Extract Z-Stack Stitching Logic

**Create a new `ZStackStitcher` class:**
- Move `stitch_across_z` method
- Delegate to PlateProcessor for actual stitching

```python
class ZStackStitcher:
    def __init__(self, config, fs_manager=None):
        self.config = config
        self.fs_manager = fs_manager or FileSystemManager()
        
    def stitch_across_z(self, plate_folder, reference_z=None, stitch_all_z_planes=True, processor=None, preprocessing_funcs=None):
        # Implementation from ZStackProcessor.stitch_across_z
```

### 4. Improve Focus Analysis Integration

**Refactor focus-related methods to better use FocusAnalyzer:**
- Move `create_best_focus_images` and `find_best_focus` methods to a new `ZStackFocusManager` class
- Delegate to FocusAnalyzer for actual focus detection

```python
class ZStackFocusManager:
    def __init__(self, config, focus_analyzer=None, fs_manager=None):
        self.config = config
        self.fs_manager = fs_manager or FileSystemManager()
        self.focus_analyzer = focus_analyzer or FocusAnalyzer(config.focus_config)
        
    def create_best_focus_images(self, input_dir, output_dir=None, focus_method='combined', focus_wavelength='all'):
        # Implementation from ZStackProcessor.create_best_focus_images
        
    def find_best_focus(self, timepoint_dir, output_dir):
        # Implementation from ZStackProcessor.find_best_focus
```

### 5. Refactor Reference Function Handling

**Create a new `ZStackReferenceAdapter` class:**
- Move `_create_reference_function`, `_adapt_function`, and `_preprocess_stack` methods
- This class will handle adapting various functions to work with Z-stacks

```python
class ZStackReferenceAdapter:
    def __init__(self, image_preprocessor=None, focus_analyzer=None):
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        self.focus_analyzer = focus_analyzer or FocusAnalyzer(FocusAnalyzerConfig())
        
    def create_reference_function(self, func_or_name, focus_method='combined'):
        # Implementation from ZStackProcessor._create_reference_function
        
    def adapt_function(self, func):
        # Implementation from ZStackProcessor._adapt_function
        
    def preprocess_stack(self, stack, channel, preprocessing_funcs=None):
        # Implementation from ZStackProcessor._preprocess_stack
```

### 6. Refactor ZStackProcessor as a Coordinator

**Refactor ZStackProcessor to use the new classes:**
- Keep high-level methods like `detect_z_stacks`, `preprocess_plate_folder`
- Delegate to the new classes for specific operations
- Maintain backward compatibility by forwarding method calls

```python
class ZStackProcessor:
    def __init__(self, config, filename_parser=None, preprocessing_funcs=None):
        self.config = config
        self.fs_manager = FileSystemManager()
        self._z_info = None
        self._z_indices = []
        self.preprocessing_funcs = preprocessing_funcs or {}
        
        # Initialize components
        self.focus_analyzer = FocusAnalyzer(config.focus_config)
        self.filename_parser = filename_parser or ImageXpressFilenameParser()
        self.image_preprocessor = ImagePreprocessor()
        
        # Initialize new component classes
        self.organizer = ZStackOrganizer(config, self.filename_parser, self.fs_manager)
        self.projector = ZStackProjector(config, self.image_preprocessor, self.fs_manager)
        self.focus_manager = ZStackFocusManager(config, self.focus_analyzer, self.fs_manager)
        self.stitcher = ZStackStitcher(config, self.fs_manager)
        self.reference_adapter = ZStackReferenceAdapter(self.image_preprocessor, self.focus_analyzer)
        
        # Initialize the reference function
        self._reference_function = self.reference_adapter.create_reference_function(config.z_reference_function)
        
    # High-level methods remain in ZStackProcessor
    def detect_z_stacks(self, plate_folder):
        # Implementation using self.organizer
        
    def preprocess_plate_folder(self, plate_folder):
        # Implementation using self.organizer
        
    # Forward method calls to appropriate components for backward compatibility
    def detect_zstack_folders(self, plate_folder):
        return self.organizer.detect_zstack_folders(plate_folder)
        
    def detect_zstack_images(self, folder_path):
        return self.organizer.detect_zstack_images(folder_path)
        
    def organize_zstack_folders(self, plate_folder):
        return self.organizer.organize_zstack_folders(plate_folder)
        
    def create_zstack_projections(self, input_dir, output_dir, projection_types=None, preprocessing_funcs=None):
        return self.projector.create_projections(input_dir, output_dir, projection_types, preprocessing_funcs)
        
    def create_best_focus_images(self, input_dir, output_dir=None, focus_method='combined', focus_wavelength='all'):
        return self.focus_manager.create_best_focus_images(input_dir, output_dir, focus_method, focus_wavelength)
        
    def find_best_focus(self, timepoint_dir, output_dir):
        return self.focus_manager.find_best_focus(timepoint_dir, output_dir)
        
    def stitch_across_z(self, plate_folder, reference_z=None, stitch_all_z_planes=True, processor=None, preprocessing_funcs=None):
        return self.stitcher.stitch_across_z(plate_folder, reference_z, stitch_all_z_planes, processor, preprocessing_funcs)
        
    # Adapter methods for backward compatibility
    def _create_reference_function(self, func_or_name):
        return self.reference_adapter.create_reference_function(func_or_name, self.config.focus_method)
        
    def _adapt_function(self, func):
        return self.reference_adapter.adapt_function(func)
        
    def _preprocess_stack(self, stack, channel):
        return self.reference_adapter.preprocess_stack(stack, channel, self.preprocessing_funcs)
```

### 7. Improve Filename Parsing Integration

**Refactor filename parsing to consistently use FilenameParser:**
- Remove direct filename parsing in ZStackProcessor
- Ensure all filename operations go through the FilenameParser instance
- Remove the `pad_site_number` method and use the FilenameParser's methods instead

### 8. Improve File System Operations

**Refactor file system operations to consistently use FileSystemManager:**
- Remove direct file system operations in ZStackProcessor
- Ensure all file operations go through the FileSystemManager instance
- Remove duplicate code for file listing, loading, and saving

## Implementation Strategy:

1. Create the new classes one by one
2. Refactor ZStackProcessor to use the new classes
3. Update tests to ensure backward compatibility
4. Update documentation to reflect the new architecture
