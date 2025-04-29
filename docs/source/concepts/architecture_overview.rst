====================
Architecture Overview
====================

Pipeline Architecture
--------------------

EZStitcher is built around a flexible pipeline architecture that allows you to create custom image processing workflows. The architecture consists of three main components:

.. note::
   The EZ module provides a simplified interface that wraps this architecture.
   See :doc:`../user_guide/ez_module` for details.

1. **PipelineOrchestrator**: Coordinates the execution of pipelines across wells
2. **Pipeline**: A sequence of processing steps
3. **Step**: A single processing operation

For detailed information about each component:

* :doc:`pipeline_orchestrator` - Details about the Orchestrator
* :doc:`pipeline` - Details about Pipelines
* :doc:`step` - Details about Steps

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

**Pipeline Management:**

* **PipelineOrchestrator**: Coordinates the entire workflow and manages plate-specific operations
* **Pipeline**: A sequence of processing steps that are executed in order
* **ProcessingContext**: Maintains state during pipeline execution

**Pipeline Factories:**

* Pipeline factories provide a convenient way to create common pipeline configurations
* For detailed information about pipeline factories, see :doc:`pipeline_factory`

**Step Components:**

* **Step**: A single processing operation that can be applied to images
* **SpecializedSteps**: Provides optimized implementations for common operations
* For detailed information about steps, see :doc:`step`

**Image Processing:**

* **ImageProcessor**: Provides static image processing functions
* **FocusAnalyzer**: Provides static focus detection methods for Z-stacks
* **Stitcher**: Performs image stitching

**Infrastructure:**

* **MicroscopeHandler**: Handles microscope-specific functionality
* **FileSystemManager**: Handles file system operations and image loading
* **Config**: Manages configuration settings for various components

These components work together to process microscopy images in a flexible and extensible way. The organization follows the typical workflow:

1. Pipeline setup and management
2. Step definition and execution
3. Image processing operations
4. Supporting infrastructure

Key Component Relationships
------------------------

The relationship between the main components is hierarchical, with the PipelineOrchestrator at the top level, managing Pipelines, which in turn manage Steps:

- **PipelineOrchestrator**: Coordinates execution across wells and provides plate-specific services
- **Pipeline**: Contains and manages a sequence of Steps
- **Step**: Performs specific processing operations

For detailed information about the PipelineOrchestrator, see :doc:`pipeline_orchestrator`.

Workflow Composition and Modularity
-----------------------------

EZStitcher's architecture is designed around a modular, composable API that allows for flexible workflow creation. The interaction between components creates a powerful system for building custom image processing workflows:

**Component Roles**

- **Pipeline**: Serves as a container for a sequence of steps, managing their execution order and data flow. Pipelines can be composed, reused, and shared across different projects. For detailed information, see :doc:`pipeline`.

- **Step**: Represents a single processing operation with well-defined inputs and outputs. Steps are highly configurable through parameters like `variable_components` and `group_by`, allowing for flexible function handling patterns. For detailed information, see :doc:`step`.

**Step Types**: EZStitcher provides various step types for common tasks:
  - **PositionGenerationStep**: Analyzes images to generate position files describing how tiles fit together
  - **ImageStitchingStep**: Assembles processed images into a single stitched image using position files
  - **ZFlatStep**: Handles Z-stack flattening with pre-configured projection methods
  - **FocusStep**: Performs focus-based Z-stack processing using focus detection algorithms
  - **CompositeStep**: Creates composite images from multiple channels with configurable weights

These step types can be seamlessly mixed in the same pipeline, allowing you to combine image processing, Z-stack handling, channel compositing, position generation, and image assembly in a single workflow.

**Workflow Composition**

This modular design allows you to:

1. **Mix and match processing steps**: Combine regular Steps with specialized PositionGenerationStep and ImageStitchingStep in a single pipeline, creating complete workflows from image processing to stitching.
2. **Create end-to-end workflows**: Build pipelines that take raw microscopy images all the way through processing, position generation, and final stitched image assembly.
3. **Reuse common workflows**: Create standard pipelines for common tasks and reuse them across projects.
4. **Customize processing per channel**: Apply different processing to different channels using function dictionaries.
5. **Handle complex data structures**: Process Z-stacks, multi-channel images, and tiled images with consistent patterns.
6. **Scale from simple to complex**: Start with basic workflows and gradually add complexity as needed.

Typical Processing Flow
--------------------

For detailed API documentation, see:

* :doc:`../api/pipeline_orchestrator`
* :doc:`../api/pipeline`
* :doc:`../api/steps`

A typical image processing and stitching workflow includes:

1. **Load and organize images**:

   .. code-block:: python

       from ezstitcher.core import AutoPipelineFactory
       from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

       orchestrator = PipelineOrchestrator(plate_path=plate_path)

2. **Process reference images**:

   .. code-block:: python

       factory = AutoPipelineFactory(
           input_dir=orchestrator.workspace_path,
           output_dir="path/to/output",
           normalize=True
       )
       pipelines = factory.create_pipelines()

3. **Generate stitching positions**:

   This is handled automatically by the pipeline factories.

4. **Process final images**:

   Channel-specific processing is available through:

   .. code-block:: python

       # Create a factory for multi-channel data
       factory = AutoPipelineFactory(
           input_dir=orchestrator.workspace_path,
           output_dir="path/to/output",
           channel_weights=[0.7, 0.3, 0]  # Use only first two channels for reference image
       )
       pipelines = factory.create_pipelines()

5. **Stitch images**:

   The final stitching step is handled automatically by all pipeline factories.

A key advantage of EZStitcher's design is that these steps aren't hardcoded—they're composed through the API, allowing you to create custom workflows tailored to your specific microscopy needs. By combining regular processing Steps with specialized PositionGenerationStep and ImageStitchingStep, you can create seamless end-to-end workflows that handle everything from initial image processing to final stitched image assembly.
