# User Guide Content Plan

## Status: In Progress
## Progress: 0%
## Last Updated: 2024-05-15
## Dependencies: [plans/documentation-outline.md]

This document outlines the detailed content for the User Guide section of the EZStitcher documentation.

## 2.1 Core Concepts

### Pipeline Architecture
- Overview of the pipeline architecture
- Key components and their responsibilities:
  - PipelineOrchestrator: Coordinates the entire workflow
  - MicroscopeHandler: Handles microscope-specific functionality
  - Stitcher: Performs image stitching
  - FocusAnalyzer: Detects focus quality
  - ImagePreprocessor: Processes images
  - FileSystemManager: Manages file operations
  - ImageLocator: Locates images in various directory structures
- Component interaction diagram

### Processing Workflow
- Step-by-step workflow:
  1. Load and organize images
  2. Process tiles (per well, per site, per channel)
  3. Select or compose channels
  4. Flatten Z-stacks (if present)
  5. Generate stitching positions
  6. Stitch images
- Workflow diagram with data flow

### Input/Output Organization
- Input directory structure requirements
- Output directory structure:
  - `_processed`: Processed individual tiles
  - `_post_processed`: Post-processed images (after channel selection/composition)
  - `_positions`: CSV files with stitching positions
  - `_stitched`: Final stitched images
- File naming conventions

### Microscope Handlers
- Role of microscope handlers
- Composition of FilenameParser and MetadataHandler
- Auto-detection mechanism
- Extension points for new microscope types

### File Naming Conventions
- Standard file naming patterns
- Component extraction from filenames
- Pattern matching and generation
- Handling different microscope naming conventions

## 2.2 File Organization

### Expected Input Structure
- General structure requirements
- ImageXpress structure:
  ```
  plate_folder/
  ├── TimePoint_1/
  │   ├── A01_s1_w1.tif
  │   ├── A01_s1_w2.tif
  │   ├── A01_s2_w1.tif
  │   └── ...
  └── ...
  ```
- Opera Phenix structure:
  ```
  plate_folder/
  ├── Images/
  │   ├── 0101K1F1P1R1.tiff
  │   ├── 0101K1F1P2R1.tiff
  │   └── ...
  ├── Index.xml
  └── ...
  ```
- Z-stack organization:
  ```
  plate_folder/
  ├── TimePoint_1/
  │   ├── ZStep_1/
  │   │   ├── A01_s1_w1.tif
  │   │   └── ...
  │   ├── ZStep_2/
  │   │   ├── A01_s1_w1.tif
  │   │   └── ...
  │   └── ...
  └── ...
  ```

### Output Directory Structure
- Processed directory:
  ```
  plate_folder_processed/
  ├── A01_s1_w1.tif
  ├── A01_s1_w2.tif
  └── ...
  ```
- Post-processed directory:
  ```
  plate_folder_post_processed/
  ├── A01_w1.tif
  ├── A01_w2.tif
  └── ...
  ```
- Positions directory:
  ```
  plate_folder_positions/
  ├── A01_w1.csv
  ├── A01_w2.csv
  └── ...
  ```
- Stitched directory:
  ```
  plate_folder_stitched/
  ├── A01_w1.tif
  ├── A01_w2.tif
  └── ...
  ```

### Naming Conventions
- Well identifiers (A01, B02, etc.)
- Site identifiers (s1, s2, etc.)
- Channel identifiers (w1, w2, etc.)
- Z-stack identifiers (z1, z2, etc.)
- Opera Phenix naming (0101K1F1P1R1, etc.)

### File Formats
- Supported image formats (TIFF, PNG, JPEG)
- Metadata formats (HTD, XML)
- Position CSV format
- Configuration file formats (JSON, YAML)

### Metadata Files
- ImageXpress HTD files
- Opera Phenix XML files
- Extracting pixel size
- Extracting grid dimensions

## 2.3 Configuration System

### Configuration Classes
- PipelineConfig
- StitcherConfig
- FocusAnalyzerConfig
- ImagePreprocessorConfig
- Relationships between configuration classes

### Configuration Presets
- Available presets:
  - default
  - z_stack_best_focus
  - z_stack_per_plane
  - high_resolution
- When to use each preset
- How to customize presets

### Configuration Files (JSON/YAML)
- Creating configuration files
- Loading configuration files
- Example configuration files
- Configuration file schema

### Configuration Validation
- Validation rules
- Error handling
- Default values
- Required vs. optional parameters

### Configuration Inheritance
- Overriding configuration values
- Merging configurations
- Configuration precedence

## 2.4 Microscope Support

### Supported Microscope Types
- ImageXpress
- Opera Phenix
- Auto-detection

### ImageXpress Specifics
- File naming pattern
- Directory structure
- Metadata extraction
- Z-stack handling

### Opera Phenix Specifics
- File naming pattern
- Directory structure
- XML metadata parsing
- Z-stack handling

### Auto-Detection
- How auto-detection works
- Fallback mechanisms
- Configuring auto-detection

### Adding Support for New Microscopes
- Creating a new FilenameParser
- Creating a new MetadataHandler
- Registering the new microscope type
- Testing the new microscope support

## 2.5 Image Processing Pipeline

### Pipeline Stages
- Overview of pipeline stages
- Data flow between stages
- Configuration options for each stage

### Tile Processing
- Per-tile operations
- Available preprocessing functions
- Custom preprocessing functions
- Batch processing of tiles

### Channel Selection/Composition
- Single channel selection
- Multi-channel composition
- Weighting channels
- Custom composition functions

### Z-Stack Flattening
- Maximum intensity projection
- Mean projection
- Best focus selection
- Custom flattening functions

### Position Generation
- Reference channel selection
- Overlap calculation
- Subpixel alignment
- Position CSV format

### Stitching
- Position-based stitching
- Blending options
- Output formats
- Large image handling

## 2.6 Z-Stack Handling

### Z-Stack Organization
- File system organization
- Z-stack detection
- Z-stack grouping

### Z-Stack Loading
- Loading Z-stacks into memory
- Memory-efficient loading
- Filtering Z-stacks

### Z-Stack Processing
- Processing individual Z-planes
- Processing entire Z-stacks
- Custom Z-stack processing functions

### Projections
- Maximum intensity projection
- Mean projection
- Custom projection functions
- Saving projections

### Best Focus Selection
- Focus quality metrics
- Best focus algorithm
- ROI-based focus selection
- Custom focus selection

## 2.7 Focus Detection

### Focus Quality Metrics
- Normalized variance
- Laplacian energy
- Tenengrad variance
- FFT-based metrics
- Combined metrics

### Focus Detection Algorithms
- How focus detection works
- Algorithm comparison
- Performance considerations
- Limitations

### ROI Selection
- Why use ROIs for focus detection
- How to specify ROIs
- Automatic ROI selection
- Multiple ROIs

### Custom Focus Functions
- Creating custom focus metrics
- Integrating custom metrics
- Testing custom metrics
- Examples

### Focus Visualization
- Visualizing focus scores
- Focus quality maps
- Focus curve plotting
- Debugging focus detection

## 2.8 Stitching

### Stitching Algorithms
- Position-based stitching
- Feature-based stitching
- Hybrid approaches
- Algorithm selection

### Position Calculation
- Overlap estimation
- Grid-based positioning
- Subpixel alignment
- Position refinement

### Image Assembly
- Assembling tiles into a mosaic
- Handling large images
- Memory-efficient assembly
- Output formats

### Blending Options
- Linear blending
- Feathering
- No blending
- Custom blending functions

### Subpixel Alignment
- Why subpixel alignment matters
- How subpixel alignment works
- Accuracy considerations
- Performance impact
