Changelog
=========

This page documents the changes in each version of EZStitcher.

Version 0.1.0 (2023-09-01)
------------------------

Initial release of EZStitcher.

**Features**:

- Basic stitching functionality
- Support for ImageXpress microscope format
- Z-stack processing with max projection
- Command-line interface

Version 0.2.0 (2023-10-15)
------------------------

**Features**:

- Added support for Opera Phenix microscope format
- Added best focus detection for Z-stacks
- Added multiple preprocessing options
- Improved stitching algorithm

**Bug Fixes**:

- Fixed memory leak in Z-stack processing
- Fixed incorrect grid size detection
- Fixed file naming issues

Version 0.3.0 (2023-12-01)
------------------------

**Features**:

- Added support for custom preprocessing functions
- Added parallel processing for wells
- Added multiple projection options
- Improved memory efficiency

**Bug Fixes**:

- Fixed Z-stack ordering issues
- Fixed metadata parsing for Opera Phenix
- Fixed channel selection in composite images

Version 0.4.0 (2024-02-15)
------------------------

**Features**:

- Complete refactoring to object-oriented architecture
- Added comprehensive configuration system
- Added automatic microscope detection
- Added support for custom focus metrics

**Bug Fixes**:

- Fixed stitching artifacts at tile boundaries
- Fixed memory issues with large images
- Fixed Z-stack flattening for Opera Phenix

Version 0.5.0 (2024-04-01)
------------------------

**Features**:

- Added support for dynamic preprocessing based on image properties
- Added support for ROI-based focus detection
- Added support for custom stitching strategies
- Improved documentation and examples

**Bug Fixes**:

- Fixed issues with file path handling on Windows
- Fixed metadata extraction for newer microscope versions
- Fixed memory leaks in image processing

Version 0.6.0 (2024-06-15)
------------------------

**Features**:

- Added support for additional microscope types
- Added advanced focus detection algorithms
- Added support for multi-channel composite images
- Improved performance and memory efficiency

**Bug Fixes**:

- Fixed issues with large Z-stacks
- Fixed metadata parsing for complex directory structures
- Fixed stitching issues with irregular grids

Version 1.0.0 (2024-08-01)
------------------------

First stable release of EZStitcher.

**Features**:

- Comprehensive support for multiple microscope types
- Advanced image processing and stitching capabilities
- Robust Z-stack handling
- Extensive documentation and examples
- Improved performance and stability

**Bug Fixes**:

- Fixed all known issues from previous versions
- Improved error handling and reporting
- Enhanced compatibility with different Python versions
