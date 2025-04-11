# Documentation Update Plan - July 2023

Status: Complete
Progress: 100%
Last Updated: 2023-07-15
Dependencies: [plans/documentation/documentation-update-plan_complete.md]

## 1. Problem Analysis

The documentation needs to be updated to reflect recent changes to the codebase, including:

1. **File Renaming Functionality**: New functionality to rename files with inconsistent site number padding
2. **Site Number Padding**: Improvements to handle inconsistent site number padding in filenames
3. **Combined Microscope Tests**: Refactoring of microscope tests into a combined parameterized test file
4. **Directory Structure Handling**: Removal of manual directory structure handling in microscope tests
5. **Wells Parameter in Tests**: Fix for the wells parameter in synthetic microscope tests
6. **Workflow Diagram**: The workflow_diagram.md file is empty and needs to be populated

Additionally, there are some general improvements needed:

1. **Incomplete Documentation**: Some features are not fully documented
2. **Inconsistent Examples**: Some examples may not match the current API
3. **Missing API Documentation**: Detailed API documentation for some components is missing
4. **Outdated Information**: Some documentation may contain outdated information

## 2. High-Level Solution

1. **Update README.md**: Add information about new features and update examples
2. **Update Feature Documentation**: Create or update documentation for specific features
3. **Create Workflow Diagrams**: Create workflow diagrams to illustrate the processing pipeline
4. **Update API Documentation**: Update API documentation to reflect current codebase
5. **Add Examples**: Add examples for new features
6. **Ensure Consistency**: Ensure consistency across all documentation

## 3. Implementation Details

### 3.1 Update README.md

#### 3.1.1 Add File Renaming Functionality

Add a section about the file renaming functionality:

```markdown
### File Renaming for Consistent Site Number Padding

EZStitcher can rename files with inconsistent site number padding to have consistent padding:

```bash
# Rename files with inconsistent site number padding
ezstitcher /path/to/plate_folder --rename-files

# Rename files with custom padding width
ezstitcher /path/to/plate_folder --rename-files --padding-width 4

# Preview renaming without actually renaming files
ezstitcher /path/to/plate_folder --rename-files --dry-run
```

Python API:

```python
from ezstitcher.core.main import process_plate_auto

# Rename files with inconsistent site number padding
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True
)

# Rename files with custom padding width
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True,
    padding_width=4
)

# Preview renaming without actually renaming files
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True,
    dry_run=True
)

# Rename files only (skip processing)
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True,
    rename_only=True
)
```
```

#### 3.1.2 Update Features Section

Add information about natural sorting and site number padding:

```markdown
- Natural sorting of filenames with mixed site number padding
- Consistent display of site numbers in logs
- File renaming for consistent site number padding
```

#### 3.1.3 Update Python API Examples

Ensure all Python API examples use the latest API and include examples for new features.

### 3.2 Create/Update Feature Documentation

#### 3.2.1 Create File Renaming Documentation

Create a new file `docs/file_renaming.md` with detailed documentation about the file renaming functionality:

```markdown
# File Renaming for Consistent Site Number Padding

EZStitcher includes functionality to rename files with inconsistent site number padding to have consistent padding. This is useful when working with datasets that have mixed padding, as it ensures consistent filenames and makes it easier to work with the data.

## How It Works

The file renaming functionality works by:

1. Scanning a directory for image files
2. Identifying files with inconsistent site number padding
3. Renaming these files to have consistent padding
4. Handling potential conflicts (e.g., if both `s1_w1.tif` and `s001_w1.tif` exist)

## Using File Renaming

### Command Line

```bash
# Rename files with inconsistent site number padding
ezstitcher /path/to/plate_folder --rename-files

# Rename files with custom padding width
ezstitcher /path/to/plate_folder --rename-files --padding-width 4

# Preview renaming without actually renaming files
ezstitcher /path/to/plate_folder --rename-files --dry-run

# Rename files only (skip processing)
ezstitcher /path/to/plate_folder --rename-files --rename-only
```

### Python API

```python
from ezstitcher.core.main import process_plate_auto

# Rename files with inconsistent site number padding
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True
)

# Rename files with custom padding width
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True,
    padding_width=4
)

# Preview renaming without actually renaming files
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True,
    dry_run=True
)

# Rename files only (skip processing)
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True,
    rename_only=True
)
```

### Using PlateProcessor Directly

```python
from ezstitcher.core.config import PlateProcessorConfig
from ezstitcher.core.plate_processor import PlateProcessor

# Create configuration
config = PlateProcessorConfig(
    rename_files=True,
    padding_width=3,
    dry_run=False
)

# Create processor
processor = PlateProcessor(config)

# Rename files
rename_map = processor.rename_files_with_consistent_padding()

# Print renamed files
for original, renamed in rename_map.items():
    print(f"Renamed {original} -> {renamed}")
```

## Conflict Handling

If both padded and unpadded versions of the same file exist (e.g., both `s1_w1.tif` and `s001_w1.tif`), the renaming will skip these files to avoid conflicts. A warning will be logged with details about the conflicts.

## Configuration Options

| Option | Description | Default |
| ------ | ----------- | ------- |
| `rename_files` | Whether to rename files | `False` |
| `padding_width` | Width to pad site numbers to | `3` |
| `dry_run` | Preview renaming without actually renaming files | `False` |
| `rename_only` | Rename files only (skip processing) | `False` |
```

#### 3.2.2 Update Site Number Padding Documentation

Add information about site number padding to the existing documentation:

```markdown
## Site Number Padding

EZStitcher handles inconsistent site number padding in filenames by:

1. Using natural sorting to correctly sort filenames with mixed padding
2. Treating site numbers as integers for sorting
3. Displaying site numbers with consistent padding in logs

This ensures that files are processed in the correct order regardless of how the site numbers are padded in the filenames.
```

### 3.3 Create Workflow Diagrams

Create workflow diagrams to illustrate the processing pipeline:

#### 3.3.1 Update workflow_diagram.md

```markdown
# Workflow Diagrams

This document contains workflow diagrams that illustrate the processing pipeline in EZStitcher.

## Overall Processing Pipeline

```
┌─────────────────┐
│ process_plate_auto │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PlateProcessor  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Auto-detection  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ File Renaming   │◄───── Optional
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Z-Stack Detection│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image Processing│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stitching       │
└─────────────────┘
```

## Z-Stack Processing Pipeline

```
┌─────────────────┐
│ ZStackProcessor │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Z-Stack Detection│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Focus Detection │◄───── Optional
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Projection      │◄───── Optional
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reference Generation│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Position Detection│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Per-Plane Stitching│◄───── Optional
└─────────────────┘
```

## File Renaming Workflow

```
┌─────────────────┐
│ rename_files_with_consistent_padding │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Find Image Files│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Map Original to │
│ Padded Filenames│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check for       │
│ Conflicts       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Rename Files    │◄───── Skip if dry_run=True
└─────────────────┘
```
```

### 3.4 Update API Documentation

#### 3.4.1 Update PlateProcessorConfig Documentation

Add documentation for the new configuration options:

```markdown
## PlateProcessorConfig

The `PlateProcessorConfig` class includes the following options for file renaming:

| Option | Description | Default |
| ------ | ----------- | ------- |
| `rename_files` | Whether to rename files | `False` |
| `padding_width` | Width to pad site numbers to | `3` |
| `dry_run` | Preview renaming without actually renaming files | `False` |
```

#### 3.4.2 Update PlateProcessor Documentation

Add documentation for the new methods:

```markdown
## PlateProcessor

The `PlateProcessor` class includes the following methods for file renaming:

### rename_files_with_consistent_padding

```python
def rename_files_with_consistent_padding(self, width=3, dry_run=False):
    """
    Rename files in the plate directory to have consistent site number padding.

    Args:
        width (int, optional): Width to pad site numbers to. Defaults to 3.
        dry_run (bool, optional): If True, only print what would be done without actually renaming

    Returns:
        dict: Dictionary mapping original filenames to new filenames
    """
```
```

#### 3.4.3 Update FileSystemManager Documentation

Add documentation for the new methods:

```markdown
## FileSystemManager

The `FileSystemManager` class includes the following methods for file renaming:

### rename_files_with_consistent_padding

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
```
```

### 3.5 Add Examples

#### 3.5.1 Add File Renaming Examples

Add examples for the file renaming functionality:

```python
# Example: Rename files with inconsistent site number padding
from ezstitcher.core.main import process_plate_auto

# Rename files with inconsistent site number padding
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True
)

# Rename files with custom padding width
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True,
    padding_width=4
)

# Preview renaming without actually renaming files
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True,
    dry_run=True
)

# Rename files only (skip processing)
process_plate_auto(
    'path/to/plate_folder',
    rename_files=True,
    rename_only=True
)
```

### 3.6 Ensure Consistency

Ensure consistency across all documentation:

1. Use consistent terminology
2. Use consistent formatting
3. Use consistent examples
4. Use consistent structure

## 4. Validation

### 4.1 Documentation Completeness

1. Verify that all features are documented
2. Verify that all API functions are documented
3. Verify that all configuration options are documented

### 4.2 Documentation Accuracy

1. Verify that documentation reflects the current codebase
2. Verify that examples work with the current API
3. Verify that installation instructions are correct

### 4.3 Documentation Usability

1. Verify that documentation is easy to navigate
2. Verify that documentation is easy to understand
3. Verify that documentation is helpful for users

## 5. Implementation Order

1. Update README.md
2. Create/update feature documentation
3. Create workflow diagrams
4. Update API documentation
5. Add examples
6. Ensure consistency

## 6. Benefits

1. **Improved user experience**: Users can easily understand how to use the new features
2. **Reduced support burden**: Fewer questions from users about the new features
3. **Easier onboarding**: New developers can quickly understand the new features
4. **Better maintainability**: Documentation helps maintain the codebase
5. **Increased adoption**: Good documentation encourages adoption of the new features

## 7. Risks and Mitigations

1. **Risk**: Documentation may become outdated as the codebase evolves
   **Mitigation**: Establish a process for updating documentation when the codebase changes

2. **Risk**: Documentation may be incomplete or inaccurate
   **Mitigation**: Review documentation regularly and solicit feedback from users

3. **Risk**: Documentation may be difficult to maintain
   **Mitigation**: Use tools and processes that make it easy to maintain documentation

## 8. Implementation Summary

Date: 2023-07-15

The documentation update has been completed successfully. The following changes were made:

1. **Updated README.md**
   - Added information about file renaming functionality
   - Added information about site number padding
   - Updated features list
   - Added command-line examples for file renaming
   - Added Python API examples for file renaming
   - Updated PlateProcessorConfig example to include file renaming options
   - Updated list of tested features
   - Added link to file renaming documentation

2. **Created File Renaming Documentation**
   - Created a new file `docs/file_renaming.md`
   - Added detailed documentation about the file renaming functionality
   - Added examples for different use cases
   - Added information about conflict handling
   - Added configuration options

3. **Updated Workflow Diagrams**
   - Populated the empty `docs/workflow_diagram.md` file
   - Added overall processing pipeline diagram
   - Added Z-stack processing pipeline diagram
   - Added file renaming workflow diagram
   - Added microscope type auto-detection workflow diagram
   - Added stitching workflow diagram

4. **Updated Auto-Detection Documentation**
   - Added a section about site number padding
   - Added examples of filenames with mixed padding
   - Added link to file renaming documentation

5. **Updated Opera Phenix Support Documentation**
   - Added a section about site number padding
   - Added examples of Opera Phenix filenames with mixed padding
   - Added link to file renaming documentation

These changes ensure that the documentation is up-to-date with the latest features and improvements in the codebase. The documentation now includes comprehensive information about file renaming functionality, site number padding, and workflow diagrams to help users understand the processing pipeline.
