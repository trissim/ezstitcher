# Plan: Fix Site Number Padding in Image Filenames

Status: Complete
Progress: 100%
Last Updated: 2023-07-15
Dependencies: None

## 1. Problem Analysis

The log output shows that the site numbers in the filenames are not consistently padded:

```
INFO     ezstitcher.core.stitcher:stitcher.py:638 Placing tile 1/16: A01_s10_w2.tif at (0.0, 1.0)
INFO     ezstitcher.core.stitcher:stitcher.py:638 Placing tile 2/16: A01_s11_w2.tif at (113.68, 0.0)
...
INFO     ezstitcher.core.stitcher:stitcher.py:638 Placing tile 8/16: A01_s1_w2.tif at (343.23999980926516, 113.68)
INFO     ezstitcher.core.stitcher:stitcher.py:638 Placing tile 9/16: A01_s2_w2.tif at (0.09999990463256836, 229.36)
```

Notice that we have `A01_s10_w2.tif` and `A01_s1_w2.tif` - the site numbers are not zero-padded, which can lead to incorrect sorting and processing order. This is particularly problematic because:

1. Files are typically sorted lexicographically by the filesystem
2. Without padding, `s1, s10, s11, s2, s3...` is the order instead of the correct `s1, s2, s3..., s10, s11`
3. This can cause tiles to be placed in the wrong positions during stitching

The issue appears to be in how the filenames are read from the directory and processed by the stitcher.

## 2. Root Cause Investigation

### 2.1 File Reading Process

The file reading process in ezstitcher involves several components:

1. **DirectoryStructureManager**: Handles directory structure and file paths
2. **FilenameParser**: Parses filenames to extract well, site, and wavelength information
3. **ImageLocator**: Locates images in the directory structure
4. **Stitcher**: Uses the located images for stitching

The issue likely occurs in one of these components, where the site numbers are not being properly padded or sorted.

### 2.2 Filename Pattern Handling

The current implementation appears to use patterns like `{well}_s{site}_w{wavelength}.tif` to match filenames. When extracting the site number, it's likely converting it directly to an integer without ensuring consistent padding when sorting or displaying.

### 2.3 Sorting Mechanism

The sorting of files is critical for correct stitching. The current implementation might be:

1. Reading files from the directory using a glob pattern
2. Sorting them lexicographically (which doesn't work correctly for non-padded numbers)
3. Processing them in that order

## 3. Potential Solutions

### 3.1 Natural Sorting

Implement a "natural sort" algorithm that correctly sorts strings containing numbers by interpreting the numbers as integers rather than characters. This would sort `s1, s2, s3..., s10, s11` correctly regardless of padding.

```python
import re

def natural_sort_key(s):
    """
    Sort strings with embedded numbers in natural order.
    E.g., ["s1", "s10", "s2"] -> ["s1", "s2", "s10"]
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Usage
sorted_files = sorted(files, key=natural_sort_key)
```

### 3.2 Consistent Padding

Modify the FilenameParser to always pad site numbers to a consistent width when generating or parsing filenames:

```python
def format_site_number(site_num, width=3):
    """
    Format site number with consistent padding.
    E.g., 1 -> "001", 10 -> "010"
    """
    return f"{site_num:0{width}d}"
```

### 3.3 Metadata-Based Sorting

Instead of relying on filename sorting, extract metadata (well, site, wavelength) from each filename and sort based on the numeric values of these fields:

```python
def metadata_sort_key(filename):
    """
    Extract metadata from filename and use for sorting.
    """
    metadata = parse_filename(filename)
    return (metadata['well'], int(metadata['site']), int(metadata['wavelength']))

# Usage
sorted_files = sorted(files, key=metadata_sort_key)
```

## 4. Recommended Solution

The most robust solution is a combination of approaches:

1. **Metadata-Based Sorting**: Sort files based on extracted metadata rather than raw filenames
2. **Consistent Display**: When logging or displaying filenames, use consistent padding for site numbers
3. **Natural Sorting Fallback**: Implement natural sorting as a fallback for cases where metadata extraction fails

This approach ensures correct processing order regardless of how the filenames are formatted on disk.

## 5. Implementation Plan

### 5.1 Identify Key Components for Modification

Based on the code analysis, the following components need to be modified:

1. **PatternMatcher.path_list_from_pattern**: This method is responsible for finding files matching a pattern and sorting them. It currently uses `sorted()` which performs lexicographical sorting.

2. **FileSystemManager.find_files_by_parser**: This method sorts files based on metadata but doesn't handle site number padding consistently.

3. **FilenameParser.parse_site**: This method extracts site numbers from filenames but doesn't handle padding.

### 5.2 Implement Natural Sorting in PatternMatcher

```python
def path_list_from_pattern(self, directory, pattern):
    # ... existing code ...

    # Find all matching files
    matching_files = []
    for file_path in directory.glob('*'):
        if file_path.is_file() and regex.match(file_path.name):
            matching_files.append(file_path.name)

    # Use natural sorting instead of lexicographical sorting
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    return sorted(matching_files, key=natural_sort_key)
```

### 5.3 Update FileSystemManager.find_files_by_parser

```python
def find_files_by_parser(self, directory, parser=None, well=None, site=None, channel=None, z_plane=None):
    # ... existing code ...

    # Sort by well, site, channel, z_plane if available
    def sort_key(item):
        meta = item[1]
        if meta is None:
            return ('', 0, 0, 0)  # Default values for sorting

        return (
            meta.get('well', ''),
            int(meta.get('site', 0)),  # Ensure site is treated as an integer
            int(meta.get('channel', 0)),  # Ensure channel is treated as an integer
            int(meta.get('z_plane', 0))  # Ensure z_plane is treated as an integer
        )

    sorted_files = sorted(matching_files, key=sort_key)
    return sorted_files
```

### 5.4 Add pad_site_number Method to FilenameParser Classes

```python
def pad_site_number(self, filename, width=3):
    """
    Ensure site number is padded to a consistent width.

    Args:
        filename (str): Filename to pad
        width (int): Width to pad to

    Returns:
        str: Filename with padded site number
    """
    # Extract just the filename without the path
    basename = os.path.basename(filename)

    # For ImageXpress format
    pattern = re.compile(r'([A-Z]\d+)_s(\d+)_w(\d+)(?:_z(\d+))?\.(.*)$')
    match = pattern.match(basename)
    if match:
        well = match.group(1)
        site = match.group(2)
        wavelength = match.group(3)
        z_index = match.group(4) or ''
        extension = match.group(5)

        # Pad site number
        padded_site = site.zfill(width)

        # Reconstruct filename
        if z_index:
            return f"{well}_s{padded_site}_w{wavelength}_z{z_index}.{extension}"
        else:
            return f"{well}_s{padded_site}_w{wavelength}.{extension}"

    # For Opera Phenix format
    # ... similar implementation for Opera Phenix format ...

    # If no match, return original filename
    return filename
```

### 5.5 Update Stitcher.assemble_image to Use Consistent Site Number Display

```python
def assemble_image(self, positions_path, images_dir, output_path, override_names=None):
    # ... existing code ...

    # When logging tile placement, ensure site numbers are displayed consistently
    for i, (filename, x_f, y_f) in enumerate(pos_entries):
        # Extract site number for display
        site_match = re.search(r'_s(\d+)_', filename)
        if site_match:
            site_num = site_match.group(1)
            # Ensure consistent display by padding to 3 digits
            padded_site = site_num.zfill(3)
            display_filename = filename.replace(f"_s{site_num}_", f"_s{padded_site}_")
        else:
            display_filename = filename

        logger.info(f"Placing tile {i+1}/{len(pos_entries)}: {display_filename} at ({x_f}, {y_f})")

    # ... rest of the method ...
```

### 5.6 Add Tests

1. Create tests with mixed padding in filenames
2. Verify that files are processed in the correct order
3. Ensure the stitching output is correct

## 6. Affected Components

1. **FilenameParser**: May need updates to handle non-padded site numbers
2. **ImageLocator**: May need updates to sort files correctly
3. **Stitcher**: May need updates to process files in the correct order
4. **DirectoryStructureManager**: May need updates to handle file listing and sorting

## 7. Risks and Mitigations

**Risk**: Changes to file sorting could affect existing workflows that depend on the current behavior.
**Mitigation**: Add thorough tests and ensure backward compatibility.

**Risk**: Different microscope formats might have different filename patterns.
**Mitigation**: Ensure the solution works with all supported formats and is extensible for new formats.

**Risk**: Performance impact of more complex sorting.
**Mitigation**: Optimize the implementation and benchmark to ensure minimal impact.

## 8. Testing Strategy

1. Create test datasets with mixed padding in filenames
2. Test with both ImageXpress and Opera Phenix formats
3. Verify correct sorting and processing order
4. Ensure stitching output is correct
5. Test edge cases (e.g., very large site numbers, missing site numbers)

## 9. Implementation Steps

1. **Implement Natural Sorting in PatternMatcher**
   - Update `path_list_from_pattern` to use natural sorting
   - Test with mixed padding in filenames

2. **Update FileSystemManager.find_files_by_parser**
   - Ensure site numbers are treated as integers for sorting
   - Test with mixed metadata

3. **Add pad_site_number Method to FilenameParser Classes**
   - Implement for both ImageXpress and Opera Phenix formats
   - Add unit tests for the method

4. **Update Stitcher.assemble_image**
   - Modify logging to display consistent site numbers
   - Test with mixed padding in filenames

5. **Add Integration Tests**
   - Create test datasets with mixed padding
   - Verify correct sorting and processing
   - Ensure stitching output is correct

6. **Document Changes**
   - Update docstrings and comments
   - Add notes about the importance of consistent site number handling

## 10. Expected Outcome

After implementing these changes, the stitcher should correctly handle filenames with mixed site number padding. The key improvements will be:

1. Files will be sorted correctly based on the numeric value of the site number, not lexicographically
2. Logging will display consistent site numbers for better readability
3. The stitching output will be correct regardless of how the filenames are formatted on disk

This will make the stitcher more robust and user-friendly, especially when working with datasets that have inconsistent filename formatting.

## 11. Implementation Summary

Date: 2023-07-15

The implementation has been completed successfully. The following changes were made:

1. **Added Natural Sorting in PatternMatcher**
   - Implemented the `_natural_sort` method in the `PatternMatcher` class
   - Updated `path_list_from_pattern` to use natural sorting instead of lexicographical sorting

2. **Updated FileSystemManager.find_files_by_parser**
   - Modified the `sort_key` function to ensure site numbers are treated as integers for sorting
   - Added explicit type conversion for site, channel, and z_plane values

3. **Added pad_site_number Method to FilenameParser**
   - Implemented a generic `pad_site_number` method in the `FilenameParser` base class
   - Added support for both ImageXpress and Opera Phenix formats

4. **Updated Stitcher.assemble_image**
   - Modified the logging to display site numbers with consistent padding
   - Added code to extract and pad site numbers for display purposes

5. **Added Tests**
   - Created a new test file `tests/unit/test_natural_sorting.py`
   - Added tests for natural sorting, site number padding, and their integration

These changes ensure that files are sorted correctly based on the numeric value of the site number, not lexicographically, and that site numbers are displayed consistently in logs. This makes the stitcher more robust and user-friendly, especially when working with datasets that have inconsistent filename formatting.
