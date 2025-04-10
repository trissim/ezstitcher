# Microscope Auto-Detection

EZStitcher includes a powerful auto-detection feature that can automatically identify the microscope type based on the file patterns and directory structure. This allows you to process data from different microscopes without having to specify the microscope type explicitly.

## How Auto-Detection Works

The auto-detection feature works by:

1. Examining the filenames of the images in the plate folder
2. Analyzing the directory structure
3. Matching the patterns against known microscope formats

Currently, EZStitcher can auto-detect the following microscope types:

- **ImageXpress**: Detected by filenames like `A01_s001_w1.tif`
- **Opera Phenix**: Detected by filenames like `r01c01f001p01-ch1sk1fk1fl1.tiff`

## Using Auto-Detection

### Command Line

When using the command line, auto-detection is enabled by default:

```bash
# Auto-detection is used by default
ezstitcher /path/to/plate_folder

# Explicitly enable auto-detection
ezstitcher /path/to/plate_folder --microscope-type auto
```

### Python API

When using the Python API, auto-detection is also enabled by default:

```python
from ezstitcher.core.main import process_plate_auto

# Auto-detection is used by default
process_plate_auto(
    'path/to/plate_folder'
)

# Explicitly enable auto-detection
process_plate_auto(
    'path/to/plate_folder',
    microscope_type='auto'
)
```

## Overriding Auto-Detection

If you want to override the auto-detection and explicitly specify the microscope type, you can do so:

### Command Line

```bash
# Explicitly specify ImageXpress
ezstitcher /path/to/plate_folder --microscope-type ImageXpress

# Explicitly specify Opera Phenix
ezstitcher /path/to/plate_folder --microscope-type OperaPhenix
```

### Python API

```python
from ezstitcher.core.main import process_plate_auto

# Explicitly specify ImageXpress
process_plate_auto(
    'path/to/plate_folder',
    microscope_type='ImageXpress'
)

# Explicitly specify Opera Phenix
process_plate_auto(
    'path/to/plate_folder',
    microscope_type='OperaPhenix'
)
```

## Troubleshooting

If auto-detection is not working as expected, you can try the following:

1. Check that your files follow the expected naming conventions for the microscope type
2. Check that your directory structure is correct
3. Explicitly specify the microscope type to see if that resolves the issue
4. Enable debug logging to see more information about the auto-detection process:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from ezstitcher.core.main import process_plate_auto
process_plate_auto('path/to/plate_folder')
```

## Adding Support for New Microscopes

If you want to add support for a new microscope type, you'll need to:

1. Create a new filename parser class that extends `FilenameParser`
2. Implement the required methods for parsing filenames
3. Add the new parser to the `detect_parser` function in `filename_parser.py`

See the [Adding Support for Other Microscopes](opera_phenix_support.md#adding-support-for-other-microscopes) section for more details.
