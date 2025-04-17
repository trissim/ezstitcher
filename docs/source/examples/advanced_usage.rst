Advanced Usage
==============

This example demonstrates advanced usage of EZStitcher.

Custom Preprocessing Functions
----------------------------

.. code-block:: python

    import numpy as np
    from skimage import exposure
    from ezstitcher.core import process_plate_folder

    # Define custom preprocessing functions
    def enhance_contrast(img):
        """Enhance contrast using histogram equalization."""
        return exposure.equalize_hist(img).astype(np.float32)

    def denoise(img):
        """Apply simple denoising."""
        from scipy import ndimage
        return ndimage.gaussian_filter(img, sigma=1)

    # Process with custom preprocessing functions
    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1", "2"],
        preprocessing_funcs={
            "1": enhance_contrast,
            "2": denoise
        }
    )

Multi-Channel Composite Images
----------------------------

.. code-block:: python

    from ezstitcher.core import process_plate_folder

    # Process with multi-channel composite
    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1", "2", "3"],
        composite_weights={
            "1": 0.6,  # Red channel
            "2": 0.3,  # Green channel
            "3": 0.1   # Blue channel
        }
    )

Using Existing Reference Positions
--------------------------------

.. code-block:: python

    from ezstitcher.core import process_plate_folder

    # Process using existing reference positions
    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1"],
        use_reference_positions=True
    )

Custom Focus ROI
--------------

.. code-block:: python

    from ezstitcher.core.config import FocusAnalyzerConfig, PlateProcessorConfig
    from ezstitcher.core.plate_processor import PlateProcessor

    # Create configuration with custom focus ROI
    focus_config = FocusAnalyzerConfig(
        method="combined",
        roi=(100, 100, 200, 200)  # (x, y, width, height)
    )

    plate_config = PlateProcessorConfig(
        reference_channels=["1"],
        focus_analyzer=focus_config
    )

    # Create and run the plate processor
    processor = PlateProcessor(plate_config)
    processor.run("path/to/plate_folder")

Direct Component Access
---------------------

.. code-block:: python

    import numpy as np
    from ezstitcher.core.focus_analyzer import FocusAnalyzer
    from ezstitcher.core.config import FocusAnalyzerConfig
    from ezstitcher.core.image_preprocessor import ImagePreprocessor
    from ezstitcher.core.config import ImagePreprocessorConfig

    # Create a focus analyzer
    focus_config = FocusAnalyzerConfig(method="combined")
    analyzer = FocusAnalyzer(focus_config)

    # Create a Z-stack (list of 2D images)
    z_stack = [np.random.rand(100, 100) for _ in range(10)]

    # Find the best focused image
    best_index, focus_scores = analyzer.find_best_focus(z_stack)
    print(f"Best focused image is at index {best_index}")

    # Create an image preprocessor
    preprocessor = ImagePreprocessor()

    # Process a brightfield image
    bf_image = np.random.rand(100, 100)
    processed_image = preprocessor.process_bf(bf_image)
