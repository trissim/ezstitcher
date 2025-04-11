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

## Site Number Padding

EZStitcher handles inconsistent site number padding in filenames by:

1. Using natural sorting to correctly sort filenames with mixed padding
2. Treating site numbers as integers for sorting
3. Displaying site numbers with consistent padding in logs

This ensures that files are processed in the correct order regardless of how the site numbers are padded in the filenames.

## Examples

### Example 1: Renaming Files with Inconsistent Padding

Consider a directory with the following files:
- `A01_s1_w1.tif`
- `A01_s2_w1.tif`
- `A01_s10_w1.tif`

After running:
```python
process_plate_auto('path/to/plate_folder', rename_files=True)
```

The directory will contain:
- `A01_s001_w1.tif`
- `A01_s002_w1.tif`
- `A01_s010_w1.tif`

### Example 2: Handling Conflicts

Consider a directory with the following files:
- `A01_s1_w1.tif`
- `A01_s001_w1.tif`

After running:
```python
process_plate_auto('path/to/plate_folder', rename_files=True)
```

The directory will remain unchanged, and a warning will be logged:
```
WARNING: Found 1 filename conflicts. These files will not be renamed.
WARNING: Conflict: ['A01_s1_w1.tif'] -> A01_s001_w1.tif
```

### Example 3: Custom Padding Width

Consider a directory with the following files:
- `A01_s1_w1.tif`
- `A01_s2_w1.tif`
- `A01_s10_w1.tif`

After running:
```python
process_plate_auto('path/to/plate_folder', rename_files=True, padding_width=4)
```

The directory will contain:
- `A01_s0001_w1.tif`
- `A01_s0002_w1.tif`
- `A01_s0010_w1.tif`

## Implementation Details

The file renaming functionality is implemented in the following classes:

- `FileSystemManager.rename_files_with_consistent_padding`: Renames files in a directory to have consistent site number padding
- `PlateProcessor.rename_files_with_consistent_padding`: Renames files in the plate directory to have consistent site number padding
- `FilenameParser.pad_site_number`: Pads site numbers in filenames to a consistent width

The implementation includes:
- Conflict detection and handling
- Dry-run option for previewing changes
- Custom padding width
- Support for both ImageXpress and Opera Phenix formats
