Custom Focus Detection
===================

This example demonstrates how to customize focus detection in EZStitcher.

Focus Detection Methods
--------------------

EZStitcher provides several focus detection methods:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with different focus detection methods
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_config=FocusAnalyzerConfig(
            method="combined"  # Options: "combined", "nvar", "lap", "ten", "fft"
        )
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/zstack_plate")

ROI-Based Focus Detection
----------------------

You can specify a region of interest (ROI) for focus detection:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with ROI-based focus detection
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_config=FocusAnalyzerConfig(
            method="combined",
            roi=(100, 100, 200, 200)  # (x, y, width, height)
        )
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/zstack_plate")

Custom Focus Weights
-----------------

You can customize the weights for the combined focus measure:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create configuration with custom focus weights
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_config=FocusAnalyzerConfig(
            method="combined",
            weights={
                "nvar": 0.4,  # Normalized variance
                "lap": 0.3,   # Laplacian energy
                "ten": 0.2,   # Tenengrad variance
                "fft": 0.1    # FFT-based metric
            }
        )
    )

    # Create and run the pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/zstack_plate")

Custom Focus Analyzer
------------------

You can create a custom focus analyzer with new metrics:

.. code-block:: python

    import numpy as np
    from scipy import ndimage
    from ezstitcher.core.focus_analyzer import FocusAnalyzer
    from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    # Create a custom focus analyzer with a new metric
    class CustomFocusAnalyzer(FocusAnalyzer):
        def __init__(self, config=None):
            super().__init__(config or FocusAnalyzerConfig())
        
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

    # Create configuration
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus"
    )

    # Create pipeline with custom focus analyzer
    pipeline = PipelineOrchestrator(config)
    pipeline.focus_analyzer = CustomFocusAnalyzer()
    pipeline.run("path/to/zstack_plate")

Dynamic Focus Method Selection
---------------------------

You can dynamically select the focus method based on image properties:

.. code-block:: python

    import numpy as np
    from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.file_system_manager import FileSystemManager
    from pathlib import Path

    def select_focus_method(plate_folder):
        """Select the best focus method based on image properties."""
        # Find a sample image
        fs_manager = FileSystemManager()
        sample_files = fs_manager.list_image_files(Path(plate_folder))
        if not sample_files:
            return "combined"
            
        sample_image = fs_manager.load_image(sample_files[0])
        
        # Analyze image properties
        mean_intensity = np.mean(sample_image)
        std_intensity = np.std(sample_image)
        
        # Determine optimal focus method based on image properties
        if std_intensity / mean_intensity < 0.1:
            # Low contrast image - use FFT-based method
            return "fft"
        elif std_intensity / mean_intensity > 0.3:
            # High contrast image - use Laplacian
            return "lap"
        else:
            # Medium contrast - use combined method
            return "combined"

    # Select focus method
    focus_method = select_focus_method("path/to/zstack_plate")

    # Create configuration with selected focus method
    config = PipelineConfig(
        reference_channels=["1"],
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        focus_config=FocusAnalyzerConfig(
            method=focus_method
        )
    )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/zstack_plate")
