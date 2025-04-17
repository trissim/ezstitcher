Focus Detection
===============

This page explains EZStitcher's focus detection capabilities.

Focus Quality Metrics
-------------------

EZStitcher provides several focus quality metrics:

Normalized Variance
~~~~~~~~~~~~~~~~~

Normalized variance measures the variance of pixel intensities normalized by the mean intensity:

.. code-block:: python

    from ezstitcher.core.focus_analyzer import FocusAnalyzer

    # Create focus analyzer
    focus_analyzer = FocusAnalyzer()

    # Calculate normalized variance
    focus_score = focus_analyzer.normalized_variance(image)

Normalized variance is effective for images with high contrast and sharp edges.

Laplacian Energy
~~~~~~~~~~~~~~

Laplacian energy measures the energy of the Laplacian of the image:

.. code-block:: python

    # Calculate Laplacian energy
    focus_score = focus_analyzer.laplacian_energy(image)

Laplacian energy is effective for detecting edges and is less sensitive to noise than other metrics.

Tenengrad Variance
~~~~~~~~~~~~~~~~

Tenengrad variance measures the variance of the gradient magnitude:

.. code-block:: python

    # Calculate Tenengrad variance
    focus_score = focus_analyzer.tenengrad_variance(image)

Tenengrad variance is effective for images with strong edges.

FFT-Based Metric
~~~~~~~~~~~~~~

The FFT-based metric measures the energy in high-frequency components:

.. code-block:: python

    # Calculate FFT-based metric
    focus_score = focus_analyzer.adaptive_fft_focus(image)

The FFT-based metric is effective for images with fine details.

Combined Metric
~~~~~~~~~~~~~

The combined metric combines multiple metrics with weights:

.. code-block:: python

    # Calculate combined metric
    focus_score = focus_analyzer.combined_focus_measure(image)

The default weights are:

- Normalized variance: 0.25
- Laplacian energy: 0.25
- Tenengrad variance: 0.25
- FFT-based metric: 0.25

You can customize the weights:

.. code-block:: python

    from ezstitcher.core.focus_analyzer import FocusAnalyzer
    from ezstitcher.core.config import FocusAnalyzerConfig

    # Create focus analyzer with custom weights
    focus_config = FocusAnalyzerConfig(
        method="combined",
        weights={
            "nvar": 0.4,
            "lap": 0.3,
            "ten": 0.2,
            "fft": 0.1
        }
    )
    focus_analyzer = FocusAnalyzer(focus_config)

    # Calculate combined metric with custom weights
    focus_score = focus_analyzer.combined_focus_measure(image)

Focus Detection Algorithms
------------------------

EZStitcher provides algorithms for detecting the best focused plane in a Z-stack:

Finding Best Focus
~~~~~~~~~~~~~~~~

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

Selecting Best Focus
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Select best focus
    best_image, best_idx, focus_scores = focus_analyzer.select_best_focus(z_stack_images)

Computing Focus Metrics
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compute focus metrics for all planes
    focus_scores = focus_analyzer.compute_focus_metrics(z_stack_images)

    # Print focus scores
    for i, score in enumerate(focus_scores):
        print(f"Plane {i}: {score}")

Algorithm Comparison
~~~~~~~~~~~~~~~~~~

Different focus detection algorithms have different strengths and weaknesses:

- **Normalized variance**: Good for high-contrast images, sensitive to noise
- **Laplacian energy**: Good for edge detection, less sensitive to noise
- **Tenengrad variance**: Good for strong edges, sensitive to noise
- **FFT-based metric**: Good for fine details, less sensitive to illumination changes
- **Combined metric**: Balanced performance across different image types

The best algorithm depends on your specific images and requirements.

ROI Selection
-----------

You can specify a region of interest (ROI) for focus detection:

.. code-block:: python

    # Define ROI (x, y, width, height)
    roi = (100, 100, 200, 200)

    # Find best focus with ROI
    best_idx, focus_scores = focus_analyzer.find_best_focus(z_stack_images, roi=roi)
    best_focus_image = z_stack_images[best_idx]

Using an ROI can improve focus detection by:

- Focusing on a specific region of interest
- Avoiding background or artifacts
- Reducing computation time

You can specify the ROI in the configuration:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with ROI
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="best_focus",
        stitch_flatten="best_focus",
        focus_method="combined",
        focus_config=FocusAnalyzerConfig(
            method="combined",
            roi=(100, 100, 200, 200)  # (x, y, width, height)
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Custom Focus Functions
--------------------

You can create custom focus metrics by extending the FocusAnalyzer class:

.. code-block:: python

    import numpy as np
    from scipy import ndimage
    from ezstitcher.core.focus_analyzer import FocusAnalyzer
    from ezstitcher.core.config import FocusAnalyzerConfig

    # Create a custom focus analyzer with a new metric
    class CustomFocusAnalyzer(FocusAnalyzer):
        def __init__(self, config=None):
            super().__init__(config)
        
        def gradient_magnitude_variance(self, image):
            """Calculate gradient magnitude variance as a focus measure."""
            grad_x = ndimage.sobel(image, axis=0)
            grad_y = ndimage.sobel(image, axis=1)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return np.var(magnitude)
        
        def custom_combined_focus(self, image):
            """Custom combined focus measure."""
            nvar = self.normalized_variance(image)
            lap = self.laplacian_energy(image)
            grad = self.gradient_magnitude_variance(image)
            
            # Custom weighting
            return 0.3 * nvar + 0.3 * lap + 0.4 * grad

    # Create configuration with custom focus detection
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="best_focus",
        stitch_flatten="best_focus",
        focus_method="custom_combined_focus",  # Use our custom method
        focus_config=FocusAnalyzerConfig(
            method="custom_combined_focus"
        )
    )

    # Create and run pipeline with custom focus analyzer
    pipeline = PipelineOrchestrator(config)
    pipeline.focus_analyzer = CustomFocusAnalyzer(config.focus_config)
    pipeline.run("path/to/plate_folder")

Focus Visualization
-----------------

You can visualize focus scores to understand how they vary across a Z-stack:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from ezstitcher.core.focus_analyzer import FocusAnalyzer
    from ezstitcher.core.file_system_manager import FileSystemManager

    # Load a Z-stack
    fs_manager = FileSystemManager()
    z_stack_dir = "path/to/zstack_folder"
    z_stack_files = sorted(fs_manager.list_image_files(z_stack_dir))
    z_stack = [fs_manager.load_image(f) for f in z_stack_files]

    # Create focus analyzer
    focus_analyzer = FocusAnalyzer()

    # Calculate focus scores for different methods
    methods = ["normalized_variance", "laplacian", "tenengrad", "fft", "combined"]
    scores = {}

    for method in methods:
        _, focus_scores = focus_analyzer.find_best_focus(z_stack, method=method)
        scores[method] = [score for _, score in focus_scores]

    # Normalize scores for comparison
    for method in methods:
        max_score = max(scores[method])
        scores[method] = [score / max_score for score in scores[method]]

    # Plot focus scores
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.plot(scores[method], label=method)

    plt.xlabel("Z-Plane")
    plt.ylabel("Normalized Focus Score")
    plt.title("Focus Scores Across Z-Stack")
    plt.legend()
    plt.grid(True)
    plt.savefig("focus_scores.png")
    plt.show()

This will create a plot showing how focus scores vary across the Z-stack for different methods.

You can also create a focus quality map:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from ezstitcher.core.focus_analyzer import FocusAnalyzer

    # Create focus analyzer
    focus_analyzer = FocusAnalyzer()

    # Create focus quality map
    def create_focus_map(image, window_size=32):
        """Create a focus quality map for an image."""
        height, width = image.shape
        map_height = height // window_size
        map_width = width // window_size
        focus_map = np.zeros((map_height, map_width))
        
        for i in range(map_height):
            for j in range(map_width):
                y = i * window_size
                x = j * window_size
                window = image[y:y+window_size, x:x+window_size]
                focus_map[i, j] = focus_analyzer.combined_focus_measure(window)
        
        return focus_map

    # Create focus map for an image
    focus_map = create_focus_map(image)

    # Plot focus map
    plt.figure(figsize=(10, 8))
    plt.imshow(focus_map, cmap='viridis')
    plt.colorbar(label='Focus Score')
    plt.title("Focus Quality Map")
    plt.savefig("focus_map.png")
    plt.show()

This will create a heatmap showing the focus quality across different regions of the image.
