# Pipeline Architecture

## Overview

EZStitcher's pipeline architecture has been redesigned to provide a more flexible, modular, and extensible framework for processing microscopy images. The new architecture is composed of three main components:

1. **PipelineOrchestrator**: Coordinates the execution of multiple pipelines across wells
2. **Pipeline**: A sequence of processing steps
3. **Step**: A single processing operation (with specialized subclasses)

This hierarchical design allows for complex workflows to be built from simple, reusable components.

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│            PipelineOrchestrator         │
│                                         │
│  ┌─────────┐    ┌─────────┐             │
│  │ Pipeline│    │ Pipeline│    ...      │
│  │         │    │         │             │
│  │ ┌─────┐ │    │ ┌─────┐ │             │
│  │ │Step │ │    │ │Step │ │             │
│  │ └─────┘ │    │ └─────┘ │             │
│  │ ┌─────┐ │    │ ┌─────┐ │             │
│  │ │Step │ │    │ │Step │ │             │
│  │ └─────┘ │    │ └─────┘ │             │
│  │   ...   │    │   ...   │             │
│  └─────────┘    └─────────┘             │
└─────────────────────────────────────────┘
```

## Key Components

### PipelineOrchestrator

The `PipelineOrchestrator` is the central coordinator that manages the execution of multiple pipelines. It handles:

- Plate and well detection
- Directory structure management
- Multithreaded execution of pipelines
- Error handling and logging

### Pipeline

A `Pipeline` is a sequence of processing steps that are executed in order. It provides:

- Step management (adding, removing, reordering)
- Context passing between steps
- Input/output directory management

### Step

A `Step` is a single processing operation that can be applied to images. The base `Step` class provides:

- Image loading and saving
- Processing function application
- Variable component handling (e.g., channels, z-indices)
- Group-by functionality for processing related images together

Specialized step classes include:

- **PositionGenerationStep**: Generates position files for stitching
- **ImageStitchingStep**: Stitches images using position files

## Function Handling

The pipeline architecture supports three patterns for processing functions:

1. **Single Function**: A callable that takes a list of images and returns a list of processed images
2. **List of Functions**: A sequence of functions applied one after another to the images
3. **Dictionary of Functions**: A mapping from component values (like channel numbers) to functions or lists of functions

This flexibility allows for complex processing workflows to be built from simple, reusable components.

## Example Usage

For examples of using the pipeline architecture, see the [User Guide](../source/user_guide/index.rst) which contains comprehensive usage examples.

## Next Steps

- [PipelineOrchestrator](pipeline_orchestrator.md): Detailed documentation on the PipelineOrchestrator class
- [Pipeline](pipeline.md): Detailed documentation on the Pipeline class
- [Steps](steps.md): Detailed documentation on the Step class and its subclasses
