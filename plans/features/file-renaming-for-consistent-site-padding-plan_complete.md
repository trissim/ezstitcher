# Plan: Implement File Renaming for Consistent Site Number Padding

Status: Complete  
Progress: 100%  
Last Updated: 2023-07-15  
Dependencies: [plans/bugs/site-number-padding-issue-plan_complete.md]

## 1. Problem Analysis

In our previous implementation, we added functionality to:
1. Sort files correctly using natural sorting
2. Display site numbers with consistent padding in logs
3. Provide a `pad_site_number` method in the `FilenameParser` class

However, we did not implement functionality to actually rename the physical files on disk to have consistent site number padding. This could be beneficial because:

1. **Consistency**: Having consistently padded filenames makes the dataset more organized and easier to understand
2. **Compatibility**: Some external tools might rely on lexicographical sorting and would benefit from padded filenames
3. **Debugging**: Consistent filenames make debugging easier
4. **User Experience**: Users might prefer to have consistently named files

The current implementation handles inconsistent padding internally, but doesn't modify the actual files.

## 2. Current State Assessment

Currently, we have:
- `FilenameParser.pad_site_number`: A method that returns a padded version of a filename
- Natural sorting in `PatternMatcher`: Ensures correct sorting regardless of padding
- Integer-based sorting in `FileSystemManager`: Ensures correct sorting based on metadata

What we're missing:
- A function to rename actual files on disk to have consistent padding
- Integration of this function into the workflow

## 3. High-Level Solution

Add a new function to rename files with inconsistent site number padding to have consistent padding. This function should:

1. Scan a directory for image files
2. Identify files with inconsistent site number padding
3. Rename these files to have consistent padding
4. Handle potential conflicts (e.g., if both `s1_w1.tif` and `s001_w1.tif` exist)
5. Be optional and configurable

## 4. Implementation Details

### 4.1 Add a rename_files_with_consistent_padding Function to FileSystemManager

```python
def rename_files_with_consistent_padding(self, directory, parser=None, width=3, dry_run=False):
    """
    Rename files in a directory to have consistent site number padding.
    
    Args:
        directory (str or Path): Directory containing files to rename
        parser (FilenameParser, optional): Parser to use for filename parsing and padding
        width (int, optional): Width to pad site numbers to
        dry_run (bool, optional): If True, only print what would be done without actually renaming
        
    Returns:
        dict: Dictionary mapping original filenames to new filenames
    """
    directory = Path(directory)
    
    # Use default parser if none provided
    if parser is None:
        # Try to detect format from files in directory
        files = list(directory.glob('*.tif')) + list(directory.glob('*.tiff'))
        if not files:
            logger.warning(f"No image files found in {directory}")
            return {}
            
        # Get filenames only
        filenames = [f.name for f in files]
        
        # Detect format
        format_type = FilenameParser.detect_format(filenames)
        if format_type is None:
            logger.warning(f"Could not detect format for files in {directory}")
            return {}
            
        # Create parser
        parser = create_parser(format_type)
    
    # Find all image files
    files = list(directory.glob('*.tif')) + list(directory.glob('*.tiff'))
    
    # Map original filenames to padded filenames
    rename_map = {}
    for file_path in files:
        original_name = file_path.name
        padded_name = parser.pad_site_number(original_name, width=width)
        
        # Only include files that need renaming
        if original_name != padded_name:
            rename_map[original_name] = padded_name
    
    # Check for conflicts (e.g., both s1_w1.tif and s001_w1.tif exist)
    # In this case, we'll skip renaming to avoid overwriting files
    new_names = set(rename_map.values())
    existing_names = set(f.name for f in files)
    conflicts = new_names.intersection(existing_names)
    
    if conflicts:
        logger.warning(f"Found {len(conflicts)} filename conflicts. These files will not be renamed.")
        for conflict in conflicts:
            # Find all original names that would map to this conflict
            conflicting_originals = [orig for orig, new in rename_map.items() if new == conflict]
            logger.warning(f"Conflict: {conflicting_originals} -> {conflict}")
            
            # Remove these entries from the rename map
            for orig in conflicting_originals:
                if orig in rename_map:
                    del rename_map[orig]
    
    # Perform the renaming
    if not dry_run:
        for original_name, padded_name in rename_map.items():
            original_path = directory / original_name
            padded_path = directory / padded_name
            
            try:
                original_path.rename(padded_path)
                logger.info(f"Renamed {original_name} -> {padded_name}")
            except Exception as e:
                logger.error(f"Failed to rename {original_name} -> {padded_name}: {e}")
    else:
        for original_name, padded_name in rename_map.items():
            logger.info(f"Would rename {original_name} -> {padded_name}")
    
    return rename_map
```

### 4.2 Add a Command-Line Option to the Main Script

```python
def add_rename_files_option(parser):
    """Add rename-files option to the argument parser."""
    parser.add_argument("--rename-files", action="store_true",
                      help="Rename files to have consistent site number padding")
    parser.add_argument("--padding-width", type=int, default=3,
                      help="Width to pad site numbers to (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                      help="Only print what would be done without actually renaming files")
```

### 4.3 Update the Main Function to Use the New Option

```python
def main():
    # ... existing code ...
    
    # Parse arguments
    args = parser.parse_args()
    
    # ... existing code ...
    
    # Rename files if requested
    if args.rename_files:
        fs_manager = FileSystemManager()
        parser = create_parser(args.microscope_type)
        rename_map = fs_manager.rename_files_with_consistent_padding(
            args.input_dir,
            parser=parser,
            width=args.padding_width,
            dry_run=args.dry_run
        )
        
        if rename_map:
            print(f"Renamed {len(rename_map)} files to have consistent site number padding.")
        else:
            print("No files needed renaming.")
    
    # ... existing code ...
```

### 4.4 Add a Method to PlateProcessor to Rename Files

```python
def rename_files_with_consistent_padding(self, width=3, dry_run=False):
    """
    Rename files in the plate directory to have consistent site number padding.
    
    Args:
        width (int, optional): Width to pad site numbers to
        dry_run (bool, optional): If True, only print what would be done without actually renaming
        
    Returns:
        dict: Dictionary mapping original filenames to new filenames
    """
    # Get the appropriate directory
    directory = self.get_input_directory()
    
    # Get the appropriate parser
    parser = self.filename_parser
    
    # Rename files
    return self.fs_manager.rename_files_with_consistent_padding(
        directory,
        parser=parser,
        width=width,
        dry_run=dry_run
    )
```

### 4.5 Add a Configuration Option to Enable File Renaming

```python
class PlateProcessorConfig(BaseModel):
    # ... existing fields ...
    
    rename_files: bool = Field(
        default=False,
        description="Whether to rename files to have consistent site number padding"
    )
    padding_width: int = Field(
        default=3,
        description="Width to pad site numbers to"
    )
```

## 5. Testing Plan

### 5.1 Unit Tests

1. Test `rename_files_with_consistent_padding` with various scenarios:
   - Files with mixed padding
   - Files with consistent padding
   - Files with conflicts
   - Empty directory
   - Invalid filenames

2. Test with dry_run=True to ensure no actual renaming occurs

### 5.2 Integration Tests

1. Test the command-line option
2. Test the PlateProcessor method
3. Test with both ImageXpress and Opera Phenix formats

## 6. Implementation Steps

1. Add the `rename_files_with_consistent_padding` function to FileSystemManager
2. Add the command-line options to the main script
3. Update the main function to use the new option
4. Add the method to PlateProcessor
5. Add the configuration option to PlateProcessorConfig
6. Add unit tests
7. Add integration tests
8. Update documentation

## 7. Potential Risks and Mitigations

**Risk**: Renaming files could break existing workflows that depend on specific filenames.
**Mitigation**: Make the renaming optional and off by default. Provide a dry-run option.

**Risk**: Renaming could fail due to file system permissions or other issues.
**Mitigation**: Add robust error handling and logging.

**Risk**: Conflicts could occur if both padded and unpadded versions of the same file exist.
**Mitigation**: Detect conflicts and skip renaming in these cases.

## 8. Future Enhancements

1. Add an option to create a backup of the original files
2. Add support for renaming files in subdirectories
3. Add support for renaming files with other inconsistencies (e.g., well names, wavelength numbers)
4. Add a GUI for file renaming

## 9. Implementation Summary

Date: 2023-07-15

The implementation has been completed successfully. The following changes were made:

1. **Added rename_files_with_consistent_padding to FileSystemManager**
   - Implemented a function to rename files with inconsistent site number padding
   - Added conflict detection and handling
   - Added dry-run option for previewing changes

2. **Added rename_files_with_consistent_padding to PlateProcessor**
   - Implemented a method to rename files in the plate directory
   - Uses the FileSystemManager function internally

3. **Updated PlateProcessorConfig**
   - Added rename_files, padding_width, and dry_run options
   - Default values: rename_files=False, padding_width=3, dry_run=False

4. **Updated process_plate_auto**
   - Added support for file renaming
   - Added rename_only option to skip processing after renaming
   - Added logging for renaming operations

5. **Added Tests**
   - Created a new test file tests/unit/test_file_renaming.py
   - Added tests for FileSystemManager.rename_files_with_consistent_padding
   - Added tests for PlateProcessor.rename_files_with_consistent_padding
   - Added tests for process_plate_auto with rename_files=True
   - Added tests for conflict handling

These changes provide a robust solution for renaming files with inconsistent site number padding, making it easier to work with datasets that have mixed padding. The implementation includes conflict detection and handling, dry-run option for previewing changes, and comprehensive tests to ensure correctness.
