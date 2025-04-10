# Opera Phenix Support in EZStitcher

EZStitcher now supports Opera Phenix microscopy data in addition to ImageXpress data. This document explains how to use EZStitcher with Opera Phenix data.

## Opera Phenix Filename Format

Opera Phenix uses a different filename format than ImageXpress:

```
rXXcYYfZZZpWW-chVskNfkNflN.tiff
```

Where:
- `rXX` is the row number (e.g., `r01`)
- `cYY` is the column number (e.g., `c03`)
- `fZZZ` is the field/site number (e.g., `f144`)
- `pWW` is the plane/Z-index (e.g., `p05`)
- `chV` is the channel number (e.g., `ch3`)
- `skN`, `fkN`, `flN` are additional parameters

Example: `r03c04f144p05-ch3sk1fk1fl1.tiff`

## Using EZStitcher with Opera Phenix Data

### Automatic Detection

By default, EZStitcher will automatically detect the microscope type based on the filename patterns. You don't need to specify anything special:

```python
from ezstitcher.core.main import process_plate_auto

# EZStitcher will automatically detect Opera Phenix data
result = process_plate_auto(
    "path/to/opera_phenix_data",
    microscope_type="auto",  # This is the default, so you can omit it
    reference_channels=['1'],
    tile_overlap=10.0
)
```

### Explicit Specification

You can also explicitly specify the microscope type:

```python
from ezstitcher.core.main import process_plate_auto

# Explicitly specify Opera Phenix format
result = process_plate_auto(
    "path/to/opera_phenix_data",
    microscope_type='OperaPhenix',
    reference_channels=['1'],
    tile_overlap=10.0
)
```

## Opera Phenix Directory Structure

Opera Phenix typically organizes data in the following structure:

```
ExperimentName_Date-Measurement/
└── Images/
    ├── r01c01f01p01-ch1sk1fk1fl1.tiff
    ├── r01c01f01p01-ch2sk1fk1fl1.tiff
    ├── r01c01f01p01-ch3sk1fk1fl1.tiff
    ├── r01c01f01p02-ch1sk1fk1fl1.tiff
    └── ...
```

For Z-stack data, Opera Phenix typically uses a structure like this:

```
ExperimentName_Date-Measurement/
└── Images/
    ├── ZStep_1/
    │   ├── r01c01f01p01-ch1sk1fk1fl1.tiff
    │   ├── r01c01f01p01-ch2sk1fk1fl1.tiff
    │   └── ...
    ├── ZStep_2/
    │   ├── r01c01f01p01-ch1sk1fk1fl1.tiff
    │   ├── r01c01f01p01-ch2sk1fk1fl1.tiff
    │   └── ...
    └── ...
```

EZStitcher automatically handles Opera Phenix directory structures. When processing Opera Phenix data, EZStitcher will:

1. Detect the Opera Phenix format based on filename patterns
2. Convert Opera Phenix filenames to ImageXpress format internally (e.g., `r01c03f144p01-ch3sk1fk1fl1.tiff` → `A03_s144_w3.tif`)
3. Process Z-stack folders (ZStep_1, ZStep_2, etc.) if present

This conversion ensures compatibility with the core stitching algorithms while preserving all the original metadata.

## Site Number Padding

EZStitcher handles inconsistent site number padding in filenames for both ImageXpress and Opera Phenix formats. For Opera Phenix, the site number is the field number (f) in the filename.

For example, the following Opera Phenix filenames will be sorted correctly:
- `r01c01f1p01-ch1sk1fk1fl1.tiff`
- `r01c01f2p01-ch1sk1fk1fl1.tiff`
- `r01c01f10p01-ch1sk1fk1fl1.tiff`

If you want to rename files to have consistent padding, see the [File Renaming](file_renaming.md) documentation.

## Adding Support for Other Microscopes

EZStitcher is designed to be easily extensible to support other microscope formats. To add support for a new microscope:

1. Create a new subclass of `FilenameParser` in `ezstitcher/core/filename_parser.py`
2. Implement the required methods: `parse_well`, `parse_site`, `parse_z_index`, `parse_channel`, and `construct_filename`
3. Update the `detect_format` method in `FilenameParser` to detect the new format
4. Update the `create_parser` function to create an instance of your new parser

Example:

```python
class NewMicroscopeFilenameParser(FilenameParser):
    def parse_well(self, filename: str) -> Optional[str]:
        # Implement parsing well ID from filename
        pass

    def parse_site(self, filename: str) -> Optional[int]:
        # Implement parsing site number from filename
        pass

    def parse_z_index(self, filename: str) -> Optional[int]:
        # Implement parsing Z-index from filename
        pass

    def parse_channel(self, filename: str) -> Optional[int]:
        # Implement parsing channel from filename
        pass

    def construct_filename(self, well: str, site: int, channel: int,
                          z_index: Optional[int] = None,
                          extension: str = '.tif') -> str:
        # Implement constructing filename from components
        pass
```

Then update the `detect_format` method and `create_parser` function to include your new parser.
