Runtime Issues
=============

This page addresses common runtime issues with EZStitcher.

File Not Found Errors
------------------

**Issue**: EZStitcher can't find image files in the plate folder.

**Solution**:

1. Check that the plate folder path is correct:

   .. code-block:: python

       from pathlib import Path
       
       plate_folder = Path("path/to/plate_folder")
       print(f"Plate folder exists: {plate_folder.exists()}")
       print(f"Plate folder is a directory: {plate_folder.is_dir()}")

2. Check that the plate folder contains image files:

   .. code-block:: python

       from pathlib import Path
       
       plate_folder = Path("path/to/plate_folder")
       image_files = list(plate_folder.glob("**/*.tif"))
       print(f"Number of image files: {len(image_files)}")
       if image_files:
           print(f"First few image files: {image_files[:5]}")

3. Make sure the microscope type is correctly detected:

   .. code-block:: python

       from ezstitcher.core.microscope_interfaces import MicroscopeHandler
       from pathlib import Path
       
       plate_folder = Path("path/to/plate_folder")
       handler = MicroscopeHandler(plate_folder=plate_folder)
       print(f"Detected microscope type: {handler.__class__.__name__}")

4. If the microscope type is not correctly detected, specify it explicitly:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       from ezstitcher.core.microscope_interfaces import MicroscopeHandler
       
       config = PipelineConfig(reference_channels=["1"])
       pipeline = PipelineOrchestrator(config)
       pipeline.microscope_handler = MicroscopeHandler(
           plate_folder="path/to/plate_folder",
           microscope_type="ImageXpress"  # or "OperaPhenix"
       )
       pipeline.run("path/to/plate_folder")

Memory Errors
-----------

**Issue**: EZStitcher runs out of memory when processing large images.

**Solution**:

1. Process fewer wells at a time:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       # Process wells in batches
       all_wells = ["A01", "A02", "A03", "B01", "B02", "B03"]
       batch_size = 2
       
       for i in range(0, len(all_wells), batch_size):
           batch_wells = all_wells[i:i+batch_size]
           print(f"Processing wells: {batch_wells}")
           
           config = PipelineConfig(
               reference_channels=["1"],
               well_filter=batch_wells
           )
           
           pipeline = PipelineOrchestrator(config)
           pipeline.run("path/to/plate_folder")

2. Reduce the image size before processing:

   .. code-block:: python

       import numpy as np
       from skimage.transform import resize
       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       # Define a preprocessing function to resize images
       def resize_image(image, scale=0.5):
           """Resize image to reduce memory usage."""
           new_shape = tuple(int(s * scale) for s in image.shape)
           resized = resize(image, new_shape, preserve_range=True)
           return resized.astype(image.dtype)
       
       # Create configuration with resizing
       config = PipelineConfig(
           reference_channels=["1"],
           reference_processing={
               "1": lambda img: resize_image(img, scale=0.5)
           }
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

3. Use a machine with more memory or enable swap space:

   .. code-block:: bash

       # Linux: Create a swap file
       sudo fallocate -l 8G /swapfile
       sudo chmod 600 /swapfile
       sudo mkswap /swapfile
       sudo swapon /swapfile
       
       # Add to /etc/fstab for persistence
       echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

Stitching Errors
-------------

**Issue**: Stitching fails or produces poor results.

**Solution**:

1. Check that the grid size is correct:

   .. code-block:: python

       from ezstitcher.core.microscope_interfaces import MicroscopeHandler
       from pathlib import Path
       
       plate_folder = Path("path/to/plate_folder")
       handler = MicroscopeHandler(plate_folder=plate_folder)
       grid_x, grid_y = handler.get_grid_dimensions(plate_folder)
       print(f"Detected grid size: {grid_x}x{grid_y}")

2. Adjust the tile overlap parameter:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig, StitcherConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       config = PipelineConfig(
           reference_channels=["1"],
           stitcher=StitcherConfig(
               tile_overlap=15.0,  # Try different values: 5.0, 10.0, 15.0, 20.0
               max_shift=75        # Increase for larger overlaps
           )
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

3. Try using a different reference channel:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       config = PipelineConfig(
           reference_channels=["2"]  # Try a different channel
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

4. Apply preprocessing to improve image quality:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       from ezstitcher.core.image_preprocessor import ImagePreprocessor
       
       config = PipelineConfig(
           reference_channels=["1"],
           reference_processing={
               "1": [
                   ImagePreprocessor.background_subtract,
                   ImagePreprocessor.equalize_histogram
               ]
           }
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

Z-Stack Issues
-----------

**Issue**: Z-stack processing fails or produces poor results.

**Solution**:

1. Check that Z-stack folders are correctly detected:

   .. code-block:: python

       from ezstitcher.core.file_system_manager import FileSystemManager
       from pathlib import Path
       
       plate_folder = Path("path/to/plate_folder")
       fs_manager = FileSystemManager()
       has_zstack, z_folders = fs_manager.detect_zstack_folders(plate_folder)
       print(f"Has Z-stack folders: {has_zstack}")
       if has_zstack:
           print(f"Z-stack folders: {z_folders}")

2. Try different Z-stack flattening methods:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       # Try different flattening methods
       for flatten_method in ["max_projection", "mean_projection", "best_focus"]:
           print(f"Trying {flatten_method}...")
           
           config = PipelineConfig(
               reference_channels=["1"],
               reference_flatten="max_projection",  # For position generation
               stitch_flatten=flatten_method        # For final stitching
           )
           
           pipeline = PipelineOrchestrator(config)
           pipeline.run("path/to/plate_folder")

3. Adjust focus detection parameters:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       config = PipelineConfig(
           reference_channels=["1"],
           reference_flatten="max_projection",
           stitch_flatten="best_focus",
           focus_config=FocusAnalyzerConfig(
               method="combined",  # Try different methods: "nvar", "lap", "ten", "fft"
               roi=(100, 100, 200, 200)  # Specify ROI for focus detection
           )
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

4. Process Z-stacks plane by plane:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       config = PipelineConfig(
           reference_channels=["1"],
           reference_flatten="max_projection",  # For position generation
           stitch_flatten=None                  # Process each Z-plane separately
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

Logging and Debugging
------------------

To enable detailed logging for debugging:

.. code-block:: python

    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='ezstitcher_debug.log'
    )
    
    # Run the pipeline
    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    
    config = PipelineConfig(reference_channels=["1"])
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

You can also use the Python debugger to step through the code:

.. code-block:: python

    import pdb
    
    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    
    config = PipelineConfig(reference_channels=["1"])
    pipeline = PipelineOrchestrator(config)
    
    # Set a breakpoint
    pdb.set_trace()
    
    pipeline.run("path/to/plate_folder")
