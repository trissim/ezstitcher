====================
Architecture Overview
====================

Pipeline Architecture
--------------------

EZStitcher is built around a flexible pipeline architecture that allows you to create custom image processing workflows. The architecture consists of three main components:

1. **PipelineOrchestrator**: Coordinates the execution of pipelines across wells
2. **Pipeline**: A sequence of processing steps
3. **Step**: A single processing operation

This hierarchical design allows complex workflows to be built from simple, reusable components:

.. code-block:: text

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

When you run a pipeline, data flows through the steps in sequence. Each step processes the images and passes the results to the next step through a shared context object.

Core Components
--------------

- **PipelineOrchestrator**: Coordinates the entire workflow and manages plate-specific operations
- **Pipeline**: A sequence of processing steps that are executed in order
- **Step**: A single processing operation that can be applied to images
- **ProcessingContext**: Maintains state during pipeline execution
- **MicroscopeHandler**: Handles microscope-specific functionality
- **Stitcher**: Performs image stitching
- **ImagePreprocessor**: Applies processing functions to images
- **ImageLocator**: Locates image files in various directory structures

These components work together to process microscopy images in a flexible and extensible way.

PipelineOrchestrator
------------------

The PipelineOrchestrator serves as the central manager for the entire processing workflow, handling key responsibilities that simplify working with microscopy data:

**Microscope Detection and Workspace Setup**:
- Automatically detects the microscope type based on file patterns
- Instantiates the appropriate MicroscopeHandler, which is crucial for all file loading and writing operations
- Provides the MicroscopeHandler as a service to pipeline steps for consistent file handling
- Creates a workspace directory (`plate_workspace`) with symlinks to protect original data
- Ensures all processing happens on the workspace, preserving original source files

**Multithreaded Processing**:
- Distributes processing across multiple wells in parallel
- Manages thread pools for efficient resource utilization
- Ensures thread safety during concurrent execution
- Collects and aggregates results from all processing threads

By abstracting these complex tasks, the PipelineOrchestrator allows users to focus on defining their processing workflows rather than dealing with low-level setup and execution details.

Processing Workflow and Modularity
-----------------------------

EZStitcher's architecture is designed around a modular, composable API that allows for flexible workflow creation. The interaction between PipelineOrchestrator, Pipeline, and Step components creates a powerful system for building custom image processing workflows:

**Architectural Design**

- **PipelineOrchestrator**: Acts as a plate manager that handles plate-level organization and multithreaded processing. It provides configured services to steps based on the plate being processed, and mirrors the plate folder structure to a workspace using symlinks to protect original source files.

- **Pipeline**: Serves as a container for a sequence of steps, managing their execution order and data flow. Pipelines can be composed, reused, and shared across different projects.

- **Step**: Represents a single processing operation with well-defined inputs and outputs. Steps are highly configurable through parameters like `variable_components` and `group_by`, allowing for flexible function handling patterns.

- **Specialized Steps**: EZStitcher provides specialized steps for common tasks:
  - **PositionGenerationStep**: Analyzes images to generate position files describing how tiles fit together
  - **ImageStitchingStep**: Assembles processed images into a single stitched image using position files

  These specialized steps can be seamlessly mixed with regular processing steps in the same pipeline, allowing you to combine image processing, position generation, and image assembly in a single workflow.

**Workflow Composition**

This modular design allows you to:

1. **Mix and match processing steps**: Combine regular Steps with specialized PositionGenerationStep and ImageStitchingStep in a single pipeline, creating complete workflows from image processing to stitching.
2. **Create end-to-end workflows**: Build pipelines that take raw microscopy images all the way through processing, position generation, and final stitched image assembly.
3. **Reuse common workflows**: Create standard pipelines for common tasks and reuse them across projects.
4. **Customize processing per channel**: Apply different processing to different channels using function dictionaries.
5. **Handle complex data structures**: Process Z-stacks, multi-channel images, and tiled images with consistent patterns.
6. **Scale from simple to complex**: Start with basic workflows and gradually add complexity as needed.

**Typical Processing Flow**

A typical image processing and stitching workflow might include:

1. **Load and organize images**:
   - Detect microscope type
   - Find image directory
   - Organize Z-stack folders
   - Pad filenames for consistent sorting

2. **Process reference images** (for position generation):
   - Flatten Z-stacks if needed
   - Apply channel-specific processing functions
   - Create composite images for better feature detection
   - Save processed reference images

3. **Generate stitching positions**:
   - Calculate relative positions of tiles using reference images
   - Save positions to CSV files

4. **Process final images** (for stitching):
   - Apply channel-specific processing functions to all channels
   - Flatten Z-stacks if needed
   - Save processed images for stitching

5. **Stitch images**:
   - Load processed images
   - Apply positions from reference channels
   - Blend overlapping regions
   - Save final stitched images

A key advantage of EZStitcher's design is that these steps aren't hardcoded—they're composed through the API, allowing you to create custom workflows tailored to your specific microscopy needs. By combining regular processing Steps with specialized PositionGenerationStep and ImageStitchingStep, you can create seamless end-to-end workflows that handle everything from initial image processing to final stitched image assembly.
