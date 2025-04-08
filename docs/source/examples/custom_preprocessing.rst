Custom Preprocessing
==================

This example demonstrates how to use custom preprocessing functions with EZStitcher.

Basic Preprocessing Functions
---------------------------

.. code-block:: python

    from ezstitcher.core import process_plate_folder
    from skimage import exposure
    from scipy import ndimage
    import numpy as np

    # Define custom preprocessing functions
    def enhance_contrast(images):
        """Enhance contrast using histogram equalization."""
        # Handle list of images
        if isinstance(images, list):
            return [exposure.equalize_hist(img).astype(np.float32) for img in images]
        else:
            return exposure.equalize_hist(images).astype(np.float32)

    def denoise(images):
        """Apply simple denoising."""
        # Handle list of images
        if isinstance(images, list):
            return [ndimage.gaussian_filter(img, sigma=1) for img in images]
        else:
            return ndimage.gaussian_filter(images, sigma=1)

    # Process with custom preprocessing functions
    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1", "2"],
        preprocessing_funcs={
            "1": enhance_contrast,
            "2": denoise
        }
    )

Percentile Normalization
----------------------

.. code-block:: python

    from ezstitcher.core import process_plate_folder
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    # Create an ImagePreprocessor instance
    preprocessor = ImagePreprocessor()

    # Define a custom preprocessing function using percentile normalization
    def percentile_norm(images):
        """Apply percentile-based normalization."""
        # Handle list of images
        if isinstance(images, list):
            return [preprocessor.percentile_normalize(img, 
                                                     low_percentile=2, 
                                                     high_percentile=98) 
                    for img in images]
        else:
            return preprocessor.percentile_normalize(images, 
                                                    low_percentile=2, 
                                                    high_percentile=98)

    # Process with custom preprocessing function
    process_plate_folder(
        'path/to/plate_folder',
        reference_channels=["1"],
        preprocessing_funcs={
            "1": percentile_norm
        }
    )

Stack Percentile Normalization for Z-Stacks
-----------------------------------------

.. code-block:: python

    from ezstitcher.core.config import ZStackProcessorConfig, PlateProcessorConfig
    from ezstitcher.core.plate_processor import PlateProcessor
    from ezstitcher.core.image_preprocessor import ImagePreprocessor
    import numpy as np

    # Create an ImagePreprocessor instance
    preprocessor = ImagePreprocessor()

    # Define a custom projection function using stack percentile normalization
    def percentile_normalized_projection(z_stack):
        """
        Create a percentile-normalized projection of a Z-stack.
        
        This function normalizes the entire stack using percentile-based contrast stretching,
        then creates a maximum intensity projection.
        
        Args:
            z_stack (list): List of images in the Z-stack
            
        Returns:
            numpy.ndarray: Normalized projection image
        """
        # Normalize the stack using percentile-based contrast stretching
        normalized_stack = preprocessor.stack_percentile_normalize(
            z_stack, 
            low_percentile=2, 
            high_percentile=98
        )
        
        # Create a maximum intensity projection
        projection = np.max(normalized_stack, axis=0)
        
        return projection

    # Create Z-stack processor configuration with the custom projection function
    zstack_config = ZStackProcessorConfig(
        create_projections=True,
        projection_types=["max"],  # Standard projections to create
        stitch_z_reference=percentile_normalized_projection,  # Use our custom function for stitching
        stitch_all_z_planes=True  # Stitch each Z-plane using the same positions
    )
    
    # Create plate processor configuration
    plate_config = PlateProcessorConfig(
        reference_channels=["1"],  # Use channel 1 as reference
        z_stack_processor=zstack_config
    )
    
    # Create and run the plate processor
    processor = PlateProcessor(plate_config)
    processor.run("path/to/plate_folder")
