Stitching
========

This page explains how EZStitcher handles image stitching.

Stitching Algorithms
------------------

EZStitcher uses position-based stitching to combine multiple tiles into a single larger image. The stitching process involves several steps:

1. **Position Generation**: Calculate the relative positions of tiles
2. **Position Refinement**: Refine positions to subpixel precision
3. **Image Assembly**: Assemble tiles into a final image
4. **Blending**: Blend overlapping regions for seamless transitions

Position-Based Stitching
~~~~~~~~~~~~~~~~~~~~~

Position-based stitching uses the relative positions of tiles to assemble them into a larger image:

.. code-block:: python

    from ezstitcher.core.stitcher import Stitcher
    from ezstitcher.core.config import StitcherConfig

    # Create stitcher
    stitcher_config = StitcherConfig(
        tile_overlap=10.0,
        max_shift=50,
        margin_ratio=0.1
    )
    stitcher = Stitcher(stitcher_config)

    # Generate positions
    stitcher.generate_positions(
        image_dir="path/to/processed_images",
        image_pattern="A01_s{iii}_w1.tif",
        output_path="path/to/positions/A01_w1.csv",
        grid_size_x=3,
        grid_size_y=3,
        overlap=10.0,
        pixel_size=0.65,
        max_shift=50
    )

    # Assemble image
    stitcher.assemble_image(
        positions_path="path/to/positions/A01_w1.csv",
        images_dir="path/to/processed_images",
        output_path="path/to/stitched/A01_w1.tif"
    )

Feature-Based Refinement
~~~~~~~~~~~~~~~~~~~~~

EZStitcher can refine positions using feature-based alignment:

.. code-block:: python

    # Generate positions with feature-based refinement
    stitcher.generate_positions(
        image_dir="path/to/processed_images",
        image_pattern="A01_s{iii}_w1.tif",
        output_path="path/to/positions/A01_w1.csv",
        grid_size_x=3,
        grid_size_y=3,
        overlap=10.0,
        pixel_size=0.65,
        max_shift=50,
        feature_based=True  # Enable feature-based refinement
    )

Feature-based refinement can improve alignment accuracy, especially for images with distinct features.

Position Calculation
------------------

Position calculation determines the relative positions of tiles for stitching.

Grid-Based Positioning
~~~~~~~~~~~~~~~~~~~

Grid-based positioning arranges tiles in a regular grid:

.. code-block:: python

    # Calculate grid-based positions
    grid_positions = stitcher.calculate_grid_positions(
        grid_size_x=3,
        grid_size_y=3,
        overlap=10.0,
        image_width=1024,
        image_height=1024
    )

The grid-based positions are calculated based on:

- Grid dimensions (number of tiles in X and Y directions)
- Tile overlap percentage
- Tile dimensions (width and height)

Subpixel Alignment
~~~~~~~~~~~~~~~

Subpixel alignment refines positions to subpixel precision:

.. code-block:: python

    # Refine positions with subpixel alignment
    refined_positions = stitcher.refine_positions(
        image_dir="path/to/processed_images",
        positions=grid_positions,
        max_shift=50
    )

Subpixel alignment uses cross-correlation to find the optimal alignment between overlapping tiles.

Position CSV Format
~~~~~~~~~~~~~~~~

Positions are saved to CSV files with the following format:

.. code-block:: text

    filename,x,y
    A01_s1_w1.tif,0.0,0.0
    A01_s2_w1.tif,1024.5,0.0
    A01_s3_w1.tif,2049.2,0.0
    A01_s4_w1.tif,0.0,1024.3
    ...

The CSV file contains:

- **filename**: The filename of the tile
- **x**: The X coordinate of the tile in the stitched image
- **y**: The Y coordinate of the tile in the stitched image

Image Assembly
------------

Image assembly combines tiles into a single larger image based on their positions.

Canvas Creation
~~~~~~~~~~~~

The first step in image assembly is creating a canvas large enough to hold all tiles:

.. code-block:: python

    # Create canvas
    canvas_width = max([pos[0] + image_width for pos in positions])
    canvas_height = max([pos[1] + image_height for pos in positions])
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint16)

The canvas dimensions are calculated based on:

- Tile positions
- Tile dimensions

Tile Placement
~~~~~~~~~~~

Tiles are placed on the canvas according to their positions:

.. code-block:: python

    # Place tiles on canvas
    for filename, x, y in positions:
        # Load image
        image = load_image(os.path.join(images_dir, filename))
        
        # Calculate placement coordinates
        x_start = int(x)
        y_start = int(y)
        
        # Place image on canvas
        canvas[y_start:y_start+image.shape[0], x_start:x_start+image.shape[1]] = image

For overlapping regions, blending is applied to create seamless transitions.

Memory-Efficient Assembly
~~~~~~~~~~~~~~~~~~~~~

For large images, memory-efficient assembly is used:

.. code-block:: python

    # Memory-efficient assembly
    stitcher.assemble_image(
        positions_path="path/to/positions/A01_w1.csv",
        images_dir="path/to/processed_images",
        output_path="path/to/stitched/A01_w1.tif",
        memory_efficient=True
    )

Memory-efficient assembly processes the image in tiles, reducing memory usage for large stitched images.

Blending Options
--------------

Blending creates seamless transitions between overlapping tiles.

Linear Blending
~~~~~~~~~~~~

Linear blending applies a linear weight mask to overlapping regions:

.. code-block:: python

    # Create linear weight mask
    def create_linear_weight_mask(shape, margin_ratio=0.1):
        """Create a linear weight mask for blending."""
        height, width = shape
        mask = np.ones((height, width), dtype=np.float32)
        
        # Calculate margin width
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)
        
        # Create linear gradients
        for i in range(margin_x):
            mask[:, i] = i / margin_x
            mask[:, width - i - 1] = i / margin_x
        
        for i in range(margin_y):
            mask[i, :] *= i / margin_y
            mask[height - i - 1, :] *= i / margin_y
        
        return mask

    # Apply weight mask to image
    def apply_weight_mask(image, mask):
        """Apply a weight mask to an image."""
        return image * mask

Linear blending creates a smooth transition between tiles, reducing visible seams.

Feathering
~~~~~~~~

Feathering applies a smoother transition between tiles:

.. code-block:: python

    # Create feathering mask
    def create_feathering_mask(shape, margin_ratio=0.1):
        """Create a feathering mask for blending."""
        height, width = shape
        mask = np.ones((height, width), dtype=np.float32)
        
        # Calculate margin width
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)
        
        # Create cosine gradients
        for i in range(margin_x):
            value = 0.5 * (1 - np.cos(np.pi * i / margin_x))
            mask[:, i] = value
            mask[:, width - i - 1] = value
        
        for i in range(margin_y):
            value = 0.5 * (1 - np.cos(np.pi * i / margin_y))
            mask[i, :] *= value
            mask[height - i - 1, :] *= value
        
        return mask

Feathering creates a smoother transition than linear blending, but requires more computation.

No Blending
~~~~~~~~~

No blending simply places tiles on the canvas without blending:

.. code-block:: python

    # Assemble image without blending
    stitcher.assemble_image(
        positions_path="path/to/positions/A01_w1.csv",
        images_dir="path/to/processed_images",
        output_path="path/to/stitched/A01_w1.tif",
        blending=False
    )

No blending is faster but may result in visible seams between tiles.

Custom Blending
~~~~~~~~~~~~

You can define custom blending functions:

.. code-block:: python

    import numpy as np
    from scipy.ndimage import gaussian_filter

    def gaussian_blending(image1, image2, overlap_region):
        """Apply Gaussian blending to overlapping regions."""
        # Create Gaussian weight mask
        mask = np.zeros_like(overlap_region, dtype=np.float32)
        mask[:, :overlap_region.shape[1]//2] = 1.0
        mask = gaussian_filter(mask, sigma=overlap_region.shape[1]/10)
        
        # Apply mask
        blended = image1 * mask + image2 * (1 - mask)
        return blended

Custom blending functions can be tailored to specific image types or requirements.

Subpixel Alignment
----------------

Subpixel alignment refines tile positions to subpixel precision.

Why Subpixel Alignment Matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subpixel alignment is important for several reasons:

- **Accuracy**: Improves alignment accuracy beyond pixel-level precision
- **Quality**: Reduces artifacts in the stitched image
- **Resolution**: Preserves the full resolution of the original images

Without subpixel alignment, stitched images may have visible seams or misalignments.

How Subpixel Alignment Works
~~~~~~~~~~~~~~~~~~~~~~~~~

Subpixel alignment uses cross-correlation to find the optimal alignment between overlapping tiles:

1. **Overlap Extraction**: Extract overlapping regions from adjacent tiles
2. **Cross-Correlation**: Calculate the cross-correlation between overlapping regions
3. **Peak Finding**: Find the peak of the cross-correlation
4. **Subpixel Refinement**: Refine the peak position to subpixel precision
5. **Position Update**: Update tile positions based on the refined alignment

The subpixel refinement uses interpolation to estimate the true peak position with subpixel precision.

Accuracy Considerations
~~~~~~~~~~~~~~~~~~~

Several factors affect subpixel alignment accuracy:

- **Image Quality**: Higher quality images yield better alignment
- **Feature Density**: Images with more features align better
- **Overlap Size**: Larger overlap regions provide more information for alignment
- **Preprocessing**: Image preprocessing can improve alignment accuracy

For best results, use high-quality images with sufficient overlap and apply appropriate preprocessing.

Performance Impact
~~~~~~~~~~~~~~

Subpixel alignment has a performance impact:

- **Computation**: Requires more computation than pixel-level alignment
- **Memory**: May require more memory for intermediate calculations
- **Time**: Takes longer to process, especially for large images

The performance impact is usually outweighed by the quality improvement, especially for scientific applications.

Complete Stitching Example
------------------------

Here's a complete example of the stitching process:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Define preprocessing function
    def enhance_contrast(image):
        """Enhance contrast using histogram equalization."""
        return ImagePreprocessor.equalize_histogram(image)

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1"],
        reference_processing=enhance_contrast,
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50,
            margin_ratio=0.1
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

This example:

1. Applies histogram equalization to enhance contrast
2. Generates positions with 10% overlap and 50-pixel maximum shift
3. Assembles the final stitched image with linear blending

The resulting stitched image will be saved to the output directory.
