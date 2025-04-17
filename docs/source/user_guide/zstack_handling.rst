Z-Stack Handling
===============

This page explains how EZStitcher handles Z-stacks.

Z-Stack Organization
------------------

Z-stacks are 3D image stacks captured at different focal planes. EZStitcher supports different Z-stack organizations depending on the microscope type:

ImageXpress
~~~~~~~~~~

ImageXpress supports two Z-stack formats:

1. **Folder-based Z-stacks**:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── ZStep_1/
    │   │   ├── A01_s1_w1.tif
    │   │   ├── A01_s1_w2.tif
    │   │   └── ...
    │   ├── ZStep_2/
    │   │   ├── A01_s1_w1.tif
    │   │   ├── A01_s1_w2.tif
    │   │   └── ...
    │   └── ...
    └── ...

2. **Suffix-based Z-stacks**:

.. code-block:: text

    plate_folder/
    ├── TimePoint_1/
    │   ├── A01_s1_w1_z1.tif
    │   ├── A01_s1_w1_z2.tif
    │   ├── A01_s1_w1_z3.tif
    │   ├── A01_s1_w2_z1.tif
    │   ├── A01_s1_w2_z2.tif
    │   ├── A01_s1_w2_z3.tif
    │   └── ...
    └── ...

Opera Phenix
~~~~~~~~~~~

Opera Phenix Z-stacks are identified by the plane identifier (P) in the filename:

.. code-block:: text

    plate_folder/
    ├── Images/
    │   ├── 0101CH1F1P1R1.tiff  # Well A01, Channel 1, Field 1, Plane 1, Round 1
    │   ├── 0101CH1F1P2R1.tiff  # Well A01, Channel 1, Field 1, Plane 2, Round 1
    │   ├── 0101CH1F1P3R1.tiff  # Well A01, Channel 1, Field 1, Plane 3, Round 1
    │   └── ...
    └── ...

EZStitcher automatically detects Z-stacks and organizes them for processing.

Z-Stack Loading
-------------

EZStitcher provides methods for loading Z-stacks into memory:

.. code-block:: python

    from ezstitcher.core.file_system_manager import FileSystemManager
    from ezstitcher.core.image_locator import ImageLocator
    from pathlib import Path

    # Create file system manager
    fs_manager = FileSystemManager()

    # Find Z-stack directories
    plate_path = Path("path/to/plate_folder")
    image_dir = ImageLocator.find_image_directory(plate_path)
    z_stack_dirs = ImageLocator.find_z_stack_dirs(image_dir)

    # Load Z-stack images
    z_stack_images = []
    for z_index, z_dir in sorted(z_stack_dirs):
        # Find images in Z-stack directory
        images = fs_manager.list_image_files(z_dir)

        # Load first image in each Z-stack directory
        if images:
            image = fs_manager.load_image(images[0])
            z_stack_images.append(image)

    print(f"Loaded {len(z_stack_images)} Z-stack images")

For Opera Phenix, the process is similar but uses the plane identifier (P) in the filename instead of directories. For ImageXpress with suffix-based Z-stacks, the process uses the z-index suffix in the filename.

Z-Stack Processing
----------------

EZStitcher provides several options for processing Z-stacks:

Processing Individual Z-Planes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can process each Z-plane separately:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration for per-plane processing
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",  # Use max projection for position generation
        stitch_flatten=None                  # Process each Z-plane separately
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

This will:

1. Use max projection for position generation
2. Process each Z-plane separately
3. Stitch each Z-plane using the positions from the reference

Processing Entire Z-Stacks
~~~~~~~~~~~~~~~~~~~~~~~~

You can also process entire Z-stacks at once:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Define a function to process entire Z-stacks
    def process_zstack(stack):
        """Process an entire Z-stack."""
        # Apply processing to each plane
        processed_stack = []
        for plane in stack:
            processed_plane = ImagePreprocessor.equalize_histogram(plane)
            processed_stack.append(processed_plane)
        return processed_stack

    # Create configuration for Z-stack processing
    config = PipelineConfig(
        reference_channels=["1"],
        reference_processing=process_zstack,
        reference_flatten="max_projection",
        stitch_flatten="best_focus"
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Custom Z-Stack Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can define custom functions for processing Z-stacks:

.. code-block:: python

    import numpy as np
    from skimage import filters

    def stack_equalize_histogram(stack):
        """Apply histogram equalization to each plane in a Z-stack."""
        from ezstitcher.core.image_preprocessor import ImagePreprocessor
        return [ImagePreprocessor.equalize_histogram(plane) for plane in stack]

    def stack_background_subtract(stack, radius=50):
        """Apply background subtraction to each plane in a Z-stack."""
        from ezstitcher.core.image_preprocessor import ImagePreprocessor
        return [ImagePreprocessor.background_subtract(plane, radius) for plane in stack]

    def stack_denoise(stack, sigma=1):
        """Apply denoising to each plane in a Z-stack."""
        return [filters.gaussian(plane, sigma=sigma) for plane in stack]

Projections
----------

EZStitcher provides several methods for creating projections from Z-stacks:

Maximum Intensity Projection
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Create maximum intensity projection
    max_projection = ImagePreprocessor.max_projection(z_stack_images)

Mean Projection
~~~~~~~~~~~~~

.. code-block:: python

    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Create mean projection
    mean_projection = ImagePreprocessor.mean_projection(z_stack_images)

Custom Projection Functions
~~~~~~~~~~~~~~~~~~~~~~~~

You can define custom projection functions:

.. code-block:: python

    import numpy as np

    def median_projection(stack):
        """Create a median projection from a Z-stack."""
        return np.median(stack, axis=0)

    def weighted_projection(stack, weights=None):
        """Create a weighted projection from a Z-stack."""
        if weights is None:
            # Default: emphasize middle planes
            num_planes = len(stack)
            weights = np.ones(num_planes)
            middle = num_planes // 2
            for i in range(num_planes):
                weights[i] = 1.0 - 0.5 * abs(i - middle) / middle

        # Apply weights
        weighted_stack = np.array([stack[i] * weights[i] for i in range(len(stack))])
        return np.sum(weighted_stack, axis=0) / np.sum(weights)

Best Focus Selection
------------------

EZStitcher can select the best focused plane in a Z-stack:

.. code-block:: python

    from ezstitcher.core.focus_analyzer import FocusAnalyzer
    from ezstitcher.core.config import FocusAnalyzerConfig

    # Create focus analyzer
    focus_config = FocusAnalyzerConfig(method="combined")
    focus_analyzer = FocusAnalyzer(focus_config)

    # Find best focus
    best_idx, focus_scores = focus_analyzer.find_best_focus(z_stack_images)
    best_focus_image = z_stack_images[best_idx]

    print(f"Best focus plane: {best_idx}")
    print(f"Focus scores: {focus_scores}")

You can also select the best focus with a region of interest (ROI):

.. code-block:: python

    # Define ROI (x, y, width, height)
    roi = (100, 100, 200, 200)

    # Find best focus with ROI
    best_idx, focus_scores = focus_analyzer.find_best_focus(z_stack_images, roi=roi)
    best_focus_image = z_stack_images[best_idx]

Complete Z-Stack Processing Example
---------------------------------

Here's a complete example of Z-stack processing:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Define Z-stack preprocessing function
    def preprocess_zstack(stack):
        """Preprocess a Z-stack."""
        # Apply histogram equalization to each plane
        return [ImagePreprocessor.equalize_histogram(plane) for plane in stack]

    # Create configuration for Z-stack processing
    config = PipelineConfig(
        reference_channels=["1"],
        reference_processing=preprocess_zstack,
        reference_flatten="max_projection",  # Use max projection for position generation
        stitch_flatten="best_focus",         # Use best focus for final stitching
        focus_method="combined",             # Use combined focus metric
        focus_config=FocusAnalyzerConfig(
            method="combined",
            roi=None,  # Use entire image
            weights={
                "nvar": 0.4,
                "lap": 0.3,
                "ten": 0.2,
                "fft": 0.1
            }
        ),
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        ),
        additional_projections=["max", "mean"]  # Create additional projections
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

This example:

1. Applies histogram equalization to each plane in the Z-stack
2. Uses max projection for position generation
3. Uses best focus for final stitching
4. Uses combined focus metric with custom weights
5. Creates additional max and mean projections
