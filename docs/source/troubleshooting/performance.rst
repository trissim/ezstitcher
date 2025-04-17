Performance Optimization
======================

This page provides tips for optimizing the performance of EZStitcher.

Memory Usage Optimization
----------------------

EZStitcher can be memory-intensive when processing large images. Here are some tips to reduce memory usage:

1. **Process fewer wells at a time**:

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

2. **Reduce image size**:

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

3. **Use memory-efficient preprocessing**:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       from ezstitcher.core.image_preprocessor import ImagePreprocessor
       
       # Use memory-efficient preprocessing
       config = PipelineConfig(
           reference_channels=["1"],
           reference_processing={
               "1": ImagePreprocessor.percentile_normalize  # More memory-efficient than equalize_histogram
           }
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

4. **Clean up temporary files**:

   .. code-block:: python

       import gc
       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       from ezstitcher.core.file_system_manager import FileSystemManager
       
       # Process wells one by one and clean up after each
       all_wells = ["A01", "A02", "A03", "B01", "B02", "B03"]
       fs_manager = FileSystemManager()
       
       for well in all_wells:
           print(f"Processing well: {well}")
           
           config = PipelineConfig(
               reference_channels=["1"],
               well_filter=[well]
           )
           
           pipeline = PipelineOrchestrator(config)
           pipeline.run("path/to/plate_folder")
           
           # Clean up temporary files
           fs_manager.clean_temp_folders("path/to", "plate_folder", keep_suffixes=["_stitched"])
           
           # Force garbage collection
           gc.collect()

Processing Speed Optimization
--------------------------

Here are some tips to improve processing speed:

1. **Use fewer reference channels**:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       # Use only one reference channel
       config = PipelineConfig(
           reference_channels=["1"]  # Instead of ["1", "2", "3"]
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

2. **Simplify preprocessing**:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       from ezstitcher.core.image_preprocessor import ImagePreprocessor
       
       # Use simpler preprocessing
       config = PipelineConfig(
           reference_channels=["1"],
           reference_processing={
               "1": ImagePreprocessor.normalize  # Faster than equalize_histogram
           }
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

3. **Use max projection instead of best focus**:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       # Use max projection (faster than best focus)
       config = PipelineConfig(
           reference_channels=["1"],
           reference_flatten="max_projection",
           stitch_flatten="max_projection"  # Instead of "best_focus"
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

4. **Process only necessary wells**:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       # Process only necessary wells
       config = PipelineConfig(
           reference_channels=["1"],
           well_filter=["A01", "A02"]  # Only process these wells
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

Parallel Processing
----------------

EZStitcher doesn't natively support parallel processing, but you can implement it manually:

1. **Process wells in parallel**:

   .. code-block:: python

       import concurrent.futures
       from pathlib import Path
       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       def process_well(well):
           """Process a single well."""
           print(f"Processing well: {well}")
           
           config = PipelineConfig(
               reference_channels=["1"],
               well_filter=[well]
           )
           
           pipeline = PipelineOrchestrator(config)
           return pipeline.run("path/to/plate_folder")
       
       # Process wells in parallel
       all_wells = ["A01", "A02", "A03", "B01", "B02", "B03"]
       
       with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
           results = list(executor.map(process_well, all_wells))
           
       print(f"Results: {results}")

2. **Process plates in parallel**:

   .. code-block:: python

       import concurrent.futures
       from pathlib import Path
       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       def process_plate(plate_folder):
           """Process a single plate."""
           print(f"Processing plate: {plate_folder}")
           
           config = PipelineConfig(
               reference_channels=["1"]
           )
           
           pipeline = PipelineOrchestrator(config)
           return pipeline.run(plate_folder)
       
       # Process plates in parallel
       plate_folders = [
           "path/to/plate1",
           "path/to/plate2",
           "path/to/plate3"
       ]
       
       with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
           results = list(executor.map(process_plate, plate_folders))
           
       print(f"Results: {results}")

Disk I/O Optimization
------------------

Disk I/O can be a bottleneck when processing large images. Here are some tips to improve disk I/O performance:

1. **Use a fast storage device**:
   - Use an SSD instead of an HDD
   - Use a local disk instead of a network drive

2. **Reduce disk I/O**:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       # Reduce disk I/O by not saving reference images
       config = PipelineConfig(
           reference_channels=["1"],
           save_reference=False  # Don't save reference images
       )
       
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

3. **Use compression**:

   .. code-block:: python

       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       from ezstitcher.core.file_system_manager import FileSystemManager
       
       # Use compression when saving images
       original_save_image = FileSystemManager.save_image
       
       def save_image_with_compression(file_path, image, compression="zlib"):
           """Save image with compression."""
           return original_save_image(file_path, image, compression=compression)
       
       # Replace the save_image method
       FileSystemManager.save_image = save_image_with_compression
       
       # Run the pipeline
       config = PipelineConfig(reference_channels=["1"])
       pipeline = PipelineOrchestrator(config)
       pipeline.run("path/to/plate_folder")

Profiling
-------

To identify performance bottlenecks, you can use Python's built-in profiling tools:

1. **cProfile**:

   .. code-block:: python

       import cProfile
       import pstats
       from ezstitcher.core.config import PipelineConfig
       from ezstitcher.core.processing_pipeline import PipelineOrchestrator
       
       # Create configuration
       config = PipelineConfig(reference_channels=["1"])
       pipeline = PipelineOrchestrator(config)
       
       # Profile the run method
       cProfile.run('pipeline.run("path/to/plate_folder")', 'ezstitcher_profile.stats')
       
       # Analyze the profile
       p = pstats.Stats('ezstitcher_profile.stats')
       p.sort_stats('cumulative').print_stats(30)

2. **line_profiler**:

   .. code-block:: bash

       pip install line_profiler

   .. code-block:: python

       # Add @profile decorator to the method you want to profile
       @profile
       def process_well(self, well, wavelength_patterns, wavelength_patterns_z, dirs):
           """Process a single well through the pipeline."""
           # Method implementation
       
       # Run the profiler
       kernprof -l -v my_script.py

3. **memory_profiler**:

   .. code-block:: bash

       pip install memory_profiler

   .. code-block:: python

       # Add @profile decorator to the method you want to profile
       @profile
       def process_well(self, well, wavelength_patterns, wavelength_patterns_z, dirs):
           """Process a single well through the pipeline."""
           # Method implementation
       
       # Run the profiler
       python -m memory_profiler my_script.py
