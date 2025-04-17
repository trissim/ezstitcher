# Development and Troubleshooting Content Plan

## Status: In Progress
## Progress: 0%
## Last Updated: 2024-05-15
## Dependencies: [plans/documentation-outline.md]

This document outlines the detailed content for the Development and Troubleshooting sections of the EZStitcher documentation.

## 5.1 Architecture Overview

### Component Diagram

```
+---------------------+
|                     |
| PipelineOrchestrator|
|                     |
+----------+----------+
           |
           | coordinates
           v
+----------+----------+     +-------------------+
|                     |     |                   |
| MicroscopeHandler   +---->+ FilenameParser    |
|                     |     |                   |
+----------+----------+     +-------------------+
           |
           | delegates to    +-------------------+
           +---------------->+                   |
                             | MetadataHandler   |
                             |                   |
                             +-------------------+
+---------------------+
|                     |      +-------------------+
| FileSystemManager   +----->+                   |
|                     |      | ImageLocator      |
+----------+----------+      |                   |
           |                 +-------------------+
           v
+----------+----------+
|                     |
| ImagePreprocessor   |
|                     |
+----------+----------+
           |
           v
+----------+----------+
|                     |
| FocusAnalyzer       |
|                     |
+----------+----------+
           |
           v
+----------+----------+
|                     |
| Stitcher            |
|                     |
+---------------------+
```

### Class Hierarchy

```
- PipelineOrchestrator
  - Uses MicroscopeHandler
  - Uses FileSystemManager
  - Uses ImagePreprocessor
  - Uses FocusAnalyzer
  - Uses Stitcher

- MicroscopeHandler
  - Composes FilenameParser
  - Composes MetadataHandler

- FilenameParser (ABC)
  - ImageXpressFilenameParser
  - OperaPhenixFilenameParser

- MetadataHandler (ABC)
  - ImageXpressMetadataHandler
  - OperaPhenixMetadataHandler

- FileSystemManager
  - Uses ImageLocator

- ImageLocator

- ImagePreprocessor

- FocusAnalyzer

- Stitcher
```

### Data Flow

1. **Input**: Plate folder path
   - PipelineOrchestrator receives plate folder path
   - MicroscopeHandler auto-detects microscope type
   - FileSystemManager prepares images (padding filenames, organizing Z-stacks)

2. **Tile Processing**:
   - PipelineOrchestrator identifies wells and patterns
   - For each well and channel:
     - Load images
     - Apply preprocessing functions
     - Save processed images

3. **Channel Selection/Composition**:
   - For each well:
     - Load processed images
     - Select or compose channels
     - Save post-processed images

4. **Z-Stack Flattening**:
   - For each well and channel:
     - Load Z-stack images
     - Apply flattening function (max projection, best focus, etc.)
     - Save flattened images

5. **Position Generation**:
   - For each well and reference channel:
     - Load reference images
     - Generate stitching positions
     - Save positions to CSV

6. **Stitching**:
   - For each well and channel:
     - Load images
     - Load positions
     - Stitch images
     - Save stitched image

### Extension Points

1. **Adding New Microscope Types**:
   - Create new FilenameParser implementation
   - Create new MetadataHandler implementation
   - Register in MicroscopeHandler

2. **Custom Preprocessing Functions**:
   - Create new preprocessing function
   - Pass to PipelineConfig.reference_processing or PipelineConfig.final_processing

3. **Custom Focus Detection**:
   - Extend FocusAnalyzer class
   - Implement new focus metrics
   - Pass to PipelineConfig.focus_config

4. **Custom Z-Stack Flattening**:
   - Create new flattening function
   - Pass to PipelineConfig.reference_flatten or PipelineConfig.stitch_flatten

## 5.2 Contributing Guidelines

### Code Style

- Use **snake_case** for variables, functions, and methods
- Use **CamelCase** for class names
- Use **UPPER_CASE** for constants
- Follow PEP 8 guidelines
- Include docstrings for all modules, classes, and functions
- Use type hints where appropriate

Example:

```python
def calculate_focus_score(image: np.ndarray, method: str = "combined") -> float:
    """
    Calculate focus score for an image.

    Args:
        image: Input image
        method: Focus detection method

    Returns:
        Focus score
    """
    # Implementation
    pass
```

### Pull Request Process

1. **Fork the Repository**:
   - Fork the repository on GitHub
   - Clone your fork locally

2. **Create a Branch**:
   - Create a branch for your feature or bug fix
   - Use a descriptive name (e.g., `feature/add-new-focus-metric`)

3. **Make Changes**:
   - Make your changes
   - Follow code style guidelines
   - Add tests for your changes
   - Update documentation

4. **Run Tests**:
   - Run all tests to ensure they pass
   - Check test coverage

5. **Submit Pull Request**:
   - Push your changes to your fork
   - Create a pull request
   - Describe your changes
   - Reference any related issues

6. **Code Review**:
   - Address review comments
   - Make requested changes
   - Ensure tests pass

7. **Merge**:
   - Once approved, your pull request will be merged

### Issue Reporting

When reporting issues, please include:

1. **Description**:
   - Clear description of the issue
   - Steps to reproduce
   - Expected behavior
   - Actual behavior

2. **Environment**:
   - Operating system
   - Python version
   - EZStitcher version
   - Dependencies versions

3. **Example**:
   - Minimal code example that reproduces the issue
   - Sample data (if possible)

4. **Screenshots**:
   - Screenshots or images (if applicable)

5. **Logs**:
   - Error messages
   - Stack traces
   - Log output

### Documentation Standards

1. **Docstrings**:
   - Use Google-style docstrings
   - Include Args, Returns, and Raises sections
   - Document all parameters
   - Document return values
   - Document exceptions

2. **Examples**:
   - Include examples in docstrings
   - Ensure examples are runnable
   - Use realistic values

3. **RST Files**:
   - Use reStructuredText for documentation
   - Follow Sphinx conventions
   - Include cross-references
   - Use proper headings

4. **README**:
   - Keep README up-to-date
   - Include installation instructions
   - Include basic usage examples
   - Link to full documentation

## 5.3 Testing Guidelines

### Test Organization

Tests are organized into the following directories:

- **`unit/`**: Unit tests for individual components
- **`integration/`**: Integration tests for the full workflow
- **`generators/`**: Synthetic data generators for testing

### Writing Tests

1. **Test Structure**:
   - Use pytest for testing
   - Use descriptive test names
   - Group related tests in classes
   - Use fixtures for common setup

2. **Test Coverage**:
   - Aim for high test coverage
   - Test both normal and edge cases
   - Test error handling

3. **Test Data**:
   - Use synthetic data for testing
   - Keep test data small
   - Include test data in repository

4. **Test Documentation**:
   - Document test purpose
   - Document test inputs and expected outputs
   - Document any special setup

Example:

```python
def test_focus_detection(self):
    """
    Test focus detection on a synthetic Z-stack.
    
    This test creates a synthetic Z-stack with varying focus levels,
    applies focus detection, and verifies that the best focus plane
    is correctly identified.
    """
    # Create synthetic Z-stack
    z_stack = create_synthetic_zstack(5, focus_plane=2)
    
    # Apply focus detection
    focus_analyzer = FocusAnalyzer()
    best_idx, scores = focus_analyzer.find_best_focus(z_stack)
    
    # Verify results
    assert best_idx == 2, f"Expected best focus at plane 2, got {best_idx}"
    assert scores[best_idx][1] > scores[0][1], "Best focus score should be higher than first plane"
    assert scores[best_idx][1] > scores[-1][1], "Best focus score should be higher than last plane"
```

### Running Tests

1. **Setup**:
   - Install test dependencies:
     ```bash
     pip install pytest pytest-cov
     ```

2. **Running All Tests**:
   - Run all tests:
     ```bash
     pytest tests/
     ```

3. **Running Specific Tests**:
   - Run a specific test file:
     ```bash
     pytest tests/test_image_processing.py
     ```
   - Run a specific test class:
     ```bash
     pytest tests/test_image_processing.py::TestImageProcessing
     ```
   - Run a specific test method:
     ```bash
     pytest tests/test_image_processing.py::TestImageProcessing::test_blur
     ```

4. **Test Coverage**:
   - Generate test coverage report:
     ```bash
     pytest --cov=ezstitcher tests/
     ```
   - Generate detailed HTML report:
     ```bash
     pytest --cov=ezstitcher --cov-report=html tests/
     ```

### Test Coverage

1. **Coverage Goals**:
   - Aim for at least 80% code coverage
   - Focus on critical components
   - Ensure all public APIs are tested

2. **Coverage Report**:
   - Review coverage report
   - Identify uncovered code
   - Add tests for uncovered code

3. **Continuous Integration**:
   - Run tests on CI
   - Enforce coverage thresholds
   - Block PRs that decrease coverage

## 5.4 Release Process

### Version Numbering

EZStitcher follows Semantic Versioning (SemVer):

- **Major Version**: Incompatible API changes
- **Minor Version**: Backwards-compatible new features
- **Patch Version**: Backwards-compatible bug fixes

Example: 1.2.3
- 1 = Major version
- 2 = Minor version
- 3 = Patch version

### Release Checklist

1. **Prepare Release**:
   - Update version number in `setup.py`
   - Update version number in `__init__.py`
   - Update CHANGELOG.md
   - Update documentation

2. **Run Tests**:
   - Run all tests
   - Check test coverage
   - Fix any failing tests

3. **Build Documentation**:
   - Build documentation
   - Check for warnings
   - Fix any documentation issues

4. **Create Release Branch**:
   - Create release branch (e.g., `release/1.2.3`)
   - Push branch to GitHub

5. **Create Release**:
   - Create GitHub release
   - Tag release with version number
   - Include release notes

6. **Publish to PyPI**:
   - Build distribution:
     ```bash
     python setup.py sdist bdist_wheel
     ```
   - Upload to PyPI:
     ```bash
     twine upload dist/*
     ```

7. **Post-Release**:
   - Update version number to next development version
   - Announce release

### Changelog Management

1. **Format**:
   - Use Keep a Changelog format
   - Group changes by type (Added, Changed, Fixed, etc.)
   - Include version number and release date

2. **Content**:
   - Describe changes in user-friendly terms
   - Link to issues or pull requests
   - Credit contributors

3. **Example**:
   ```
   ## [1.2.3] - 2024-05-15

   ### Added
   - New focus detection algorithm (#123)
   - Support for Opera Phenix microscope (#124)

   ### Changed
   - Improved performance of stitching algorithm (#125)
   - Updated documentation (#126)

   ### Fixed
   - Fixed bug in Z-stack processing (#127)
   - Fixed memory leak in image loading (#128)
   ```

### Distribution

1. **PyPI**:
   - Register package on PyPI
   - Upload releases to PyPI
   - Ensure dependencies are correctly specified

2. **GitHub**:
   - Tag releases on GitHub
   - Attach distribution files to releases
   - Include release notes

3. **Documentation**:
   - Update documentation on Read the Docs
   - Ensure documentation matches release

## 6.1 Common Issues

### File Not Found Errors

1. **Issue**: Files not found during processing
   - **Cause**: Incorrect file paths, missing files, or unexpected directory structure
   - **Solution**: 
     - Check that the plate folder exists and contains images
     - Verify directory structure matches expected format
     - Use `ImageLocator.find_image_directory` to locate images
     - Enable debug logging to see which files are being searched for

2. **Issue**: Metadata file not found
   - **Cause**: Missing HTD or XML file, or incorrect microscope type
   - **Solution**:
     - Check that metadata file exists
     - Verify microscope type is correctly detected
     - Manually specify microscope type if auto-detection fails

3. **Issue**: Z-stack folders not found
   - **Cause**: Incorrect Z-stack folder naming or structure
   - **Solution**:
     - Check Z-stack folder naming (should be ZStep_#)
     - Verify Z-stack folder structure
     - Use `FileSystemManager.detect_zstack_folders` to check for Z-stack folders

### Memory Issues

1. **Issue**: Out of memory during processing
   - **Cause**: Processing large images or many images at once
   - **Solution**:
     - Process fewer wells at a time using `well_filter`
     - Reduce image size before processing
     - Increase available memory
     - Use memory-efficient processing options

2. **Issue**: Memory leak during processing
   - **Cause**: Images not being released after processing
   - **Solution**:
     - Ensure images are properly released after processing
     - Use context managers for file handling
     - Run garbage collection explicitly

3. **Issue**: Slow processing due to memory swapping
   - **Cause**: Not enough physical memory
   - **Solution**:
     - Reduce batch size
     - Process one well at a time
     - Close other memory-intensive applications

### Performance Problems

1. **Issue**: Slow image loading
   - **Cause**: Large images, slow disk, or inefficient loading
   - **Solution**:
     - Use faster storage
     - Optimize image loading
     - Load images in parallel

2. **Issue**: Slow image processing
   - **Cause**: Computationally intensive processing
   - **Solution**:
     - Simplify preprocessing functions
     - Use more efficient algorithms
     - Process images in parallel

3. **Issue**: Slow stitching
   - **Cause**: Large images, many tiles, or complex alignment
   - **Solution**:
     - Reduce image size
     - Optimize alignment parameters
     - Use faster stitching algorithms

### Stitching Artifacts

1. **Issue**: Misaligned tiles
   - **Cause**: Incorrect overlap, max_shift, or alignment failure
   - **Solution**:
     - Adjust tile_overlap parameter
     - Increase max_shift parameter
     - Use better reference channels
     - Apply preprocessing to improve feature detection

2. **Issue**: Visible seams
   - **Cause**: Brightness differences between tiles or poor blending
   - **Solution**:
     - Apply background subtraction
     - Use histogram equalization
     - Adjust margin_ratio for better blending

3. **Issue**: Missing tiles
   - **Cause**: Tiles not found or excluded during processing
   - **Solution**:
     - Check that all tiles exist
     - Verify file naming patterns
     - Check for errors during processing

## 6.2 Error Messages

### Understanding Error Messages

1. **File Not Found Errors**:
   - `FileNotFoundError: No such file or directory: 'path/to/file'`
     - **Meaning**: The specified file does not exist
     - **Action**: Check file path, verify file exists

2. **Type Errors**:
   - `TypeError: function() got an unexpected keyword argument 'param'`
     - **Meaning**: Incorrect parameter name or type
     - **Action**: Check function signature, verify parameter names and types

3. **Value Errors**:
   - `ValueError: Grid size mismatch: 3×3≠8`
     - **Meaning**: Expected 9 files (3×3 grid) but found 8
     - **Action**: Check grid size, verify all files exist

4. **Import Errors**:
   - `ImportError: No module named 'module_name'`
     - **Meaning**: Missing dependency
     - **Action**: Install missing dependency

5. **Runtime Errors**:
   - `RuntimeError: Error during stitching: ...`
     - **Meaning**: Error occurred during stitching
     - **Action**: Check error details, verify inputs

### Common Error Codes

1. **Error Code 1**: Invalid microscope type
   - **Meaning**: Specified microscope type is not supported
   - **Action**: Use 'auto', 'ImageXpress', or 'OperaPhenix'

2. **Error Code 2**: Invalid focus method
   - **Meaning**: Specified focus method is not supported
   - **Action**: Use 'combined', 'nvar', 'lap', 'ten', or 'fft'

3. **Error Code 3**: Invalid projection method
   - **Meaning**: Specified projection method is not supported
   - **Action**: Use 'max_projection', 'mean_projection', or 'best_focus'

4. **Error Code 4**: Invalid configuration
   - **Meaning**: Configuration is invalid
   - **Action**: Check configuration parameters

5. **Error Code 5**: Processing error
   - **Meaning**: Error occurred during processing
   - **Action**: Check error details, verify inputs

### Debugging Strategies

1. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   ```

2. **Step-by-Step Debugging**:
   - Process one well at a time
   - Process one channel at a time
   - Process one step at a time

3. **Inspect Intermediate Results**:
   - Save intermediate images
   - Check intermediate results
   - Verify each step

4. **Use Try-Except Blocks**:
   ```python
   try:
       pipeline.run("path/to/plate_folder")
   except Exception as e:
       print(f"Error: {e}")
       import traceback
       traceback.print_exc()
   ```

5. **Check File Paths**:
   ```python
   from pathlib import Path
   
   plate_path = Path("path/to/plate_folder")
   print(f"Plate path exists: {plate_path.exists()}")
   print(f"Plate path is directory: {plate_path.is_dir()}")
   print(f"Plate path contents: {list(plate_path.iterdir())}")
   ```

### Logging Configuration

1. **Basic Logging**:
   ```python
   import logging
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   ```

2. **Advanced Logging**:
   ```python
   import logging
   
   # Create logger
   logger = logging.getLogger('ezstitcher')
   logger.setLevel(logging.DEBUG)
   
   # Create console handler
   console_handler = logging.StreamHandler()
   console_handler.setLevel(logging.INFO)
   
   # Create file handler
   file_handler = logging.FileHandler('ezstitcher.log')
   file_handler.setLevel(logging.DEBUG)
   
   # Create formatter
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   console_handler.setFormatter(formatter)
   file_handler.setFormatter(formatter)
   
   # Add handlers to logger
   logger.addHandler(console_handler)
   logger.addHandler(file_handler)
   ```

3. **Logging in Configuration**:
   ```python
   from ezstitcher.core.config import PipelineConfig
   
   config = PipelineConfig(
       # Other parameters...
   )
   
   # Enable debug logging
   import logging
   logging.getLogger('ezstitcher').setLevel(logging.DEBUG)
   ```

## 6.3 Performance Optimization

### Memory Management

1. **Reduce Memory Usage**:
   - Process one well at a time
   - Process one channel at a time
   - Release images after processing
   - Use memory-efficient data types

2. **Optimize Image Loading**:
   - Load images only when needed
   - Use lazy loading
   - Release images after processing
   - Use memory-mapped files

3. **Manage Large Images**:
   - Downsample images before processing
   - Process regions of interest
   - Use tiling for large images
   - Use streaming processing

4. **Example**:
   ```python
   # Process one well at a time
   config = PipelineConfig(
       reference_channels=["1"]
   )
   
   pipeline = PipelineOrchestrator(config)
   
   # Get all wells
   from ezstitcher.core.image_locator import ImageLocator
   from ezstitcher.core.file_system_manager import FileSystemManager
   
   fs_manager = FileSystemManager()
   plate_path = "path/to/plate_folder"
   image_dir = ImageLocator.find_image_directory(plate_path)
   
   # Auto-detect microscope type
   microscope_handler = create_microscope_handler('auto', plate_folder=plate_path)
   
   # Get all patterns
   patterns = microscope_handler.parser.auto_detect_patterns(image_dir, group_by='well')
   
   # Process one well at a time
   for well in patterns:
       print(f"Processing well {well}")
       config.well_filter = [well]
       pipeline = PipelineOrchestrator(config)
       pipeline.run(plate_path)
   ```

### Parallel Processing

1. **Parallel Image Processing**:
   - Process multiple images in parallel
   - Use multiprocessing for CPU-bound tasks
   - Use threading for I/O-bound tasks
   - Balance parallelism with memory usage

2. **Parallel Well Processing**:
   - Process multiple wells in parallel
   - Use process pool for well processing
   - Balance parallelism with memory usage

3. **Example**:
   ```python
   import multiprocessing as mp
   from functools import partial
   
   def process_well(well, plate_path, config):
       """Process a single well."""
       print(f"Processing well {well}")
       config.well_filter = [well]
       pipeline = PipelineOrchestrator(config)
       return pipeline.run(plate_path)
   
   # Get all wells
   from ezstitcher.core.image_locator import ImageLocator
   from ezstitcher.core.file_system_manager import FileSystemManager
   from ezstitcher.core.microscope_interfaces import create_microscope_handler
   
   fs_manager = FileSystemManager()
   plate_path = "path/to/plate_folder"
   image_dir = ImageLocator.find_image_directory(plate_path)
   
   # Auto-detect microscope type
   microscope_handler = create_microscope_handler('auto', plate_folder=plate_path)
   
   # Get all patterns
   patterns = microscope_handler.parser.auto_detect_patterns(image_dir, group_by='well')
   
   # Create base configuration
   config = PipelineConfig(
       reference_channels=["1"]
   )
   
   # Process wells in parallel
   with mp.Pool(processes=4) as pool:
       process_func = partial(process_well, plate_path=plate_path, config=config)
       results = pool.map(process_func, patterns.keys())
   
   # Check results
   for well, success in zip(patterns.keys(), results):
       print(f"Well {well}: {'Success' if success else 'Failure'}")
   ```

### File I/O Optimization

1. **Reduce Disk Access**:
   - Batch file operations
   - Use memory-mapped files
   - Cache frequently accessed files
   - Use SSD for temporary files

2. **Optimize File Formats**:
   - Use efficient file formats (TIFF with compression)
   - Use appropriate compression
   - Balance compression with speed
   - Use metadata for faster access

3. **Example**:
   ```python
   from ezstitcher.core.config import PipelineConfig
   
   # Use compression for output files
   config = PipelineConfig(
       reference_channels=["1"]
   )
   
   # Create and run pipeline
   pipeline = PipelineOrchestrator(config)
   
   # Override save_image to use compression
   from ezstitcher.core.file_system_manager import FileSystemManager
   original_save_image = FileSystemManager.save_image
   
   def save_image_with_compression(file_path, image, compression='zlib'):
       """Save image with compression."""
       return original_save_image(file_path, image, compression=compression)
   
   # Replace method
   FileSystemManager.save_image = save_image_with_compression
   
   # Run pipeline
   pipeline.run("path/to/plate_folder")
   
   # Restore original method
   FileSystemManager.save_image = original_save_image
   ```

### Image Processing Optimization

1. **Optimize Algorithms**:
   - Use efficient algorithms
   - Avoid unnecessary operations
   - Use vectorized operations
   - Use GPU acceleration when available

2. **Reduce Image Size**:
   - Downsample images
   - Crop images
   - Use regions of interest
   - Balance size with quality

3. **Example**:
   ```python
   import numpy as np
   from ezstitcher.core.config import PipelineConfig
   
   # Define efficient preprocessing function
   def efficient_preprocess(image):
       """Efficient preprocessing function."""
       # Downsample image
       if image.shape[0] > 1000 or image.shape[1] > 1000:
           from skimage.transform import resize
           image = resize(image, (image.shape[0] // 2, image.shape[1] // 2),
                         preserve_range=True).astype(image.dtype)
       
       # Normalize using vectorized operations
       p_low, p_high = np.percentile(image, (2, 98))
       return np.clip((image - p_low) * (65535 / (p_high - p_low)), 0, 65535).astype(np.uint16)
   
   # Create configuration with efficient preprocessing
   config = PipelineConfig(
       reference_channels=["1"],
       reference_processing=efficient_preprocess
   )
   
   # Create and run pipeline
   pipeline = PipelineOrchestrator(config)
   pipeline.run("path/to/plate_folder")
   ```
