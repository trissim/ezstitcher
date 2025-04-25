.. _best-practices:

==============
Best Practices
==============

This guide provides comprehensive best practices for using EZStitcher effectively.

.. _best-practices-pipeline:

Pipeline Best Practices
--------------------

1. **Parameterize your pipelines**: Make key parameters configurable

   .. code-block:: python

       def create_pipeline(plate_path, num_workers=1, normalize=True):
           """Create a pipeline with configurable parameters."""
           # Create configuration and orchestrator
           config = PipelineConfig(num_workers=num_workers)
           orchestrator = PipelineOrchestrator(config=config, plate_path=plate_path)

           # Build steps based on parameters
           steps = []

           # Add normalization step if requested
           if normalize:
               steps.append(Step(
                   func=IP.stack_percentile_normalize,
                   input_dir=orchestrator.workspace_path
               ))

           # Always add position generation and stitching
           steps.append(PositionGenerationStep())
           steps.append(ImageStitchingStep())

           # Create pipeline with the configured steps
           pipeline = Pipeline(
               steps=steps,
               name="Configurable Pipeline"
           )

           return orchestrator, pipeline

2. **Use functions to create pipelines**: Encapsulate pipeline creation in functions

   .. code-block:: python

       # Create a function for each pipeline type
       def create_basic_pipeline(plate_path, num_workers=1):
           """Create a basic processing pipeline."""
           # Pipeline creation code...
           return orchestrator, pipeline

       def create_advanced_pipeline(plate_path, num_workers=1):
           """Create an advanced processing pipeline."""
           # Pipeline creation code...
           return orchestrator, pipeline

3. **Document your pipelines**: Add comments explaining the purpose of each step

   .. code-block:: python

       pipeline = Pipeline(
           steps=[
               # Normalize images to standardize intensity values
               Step(
                   func=IP.stack_percentile_normalize,
                   input_dir=orchestrator.workspace_path
               ),

               # Generate positions for stitching
               PositionGenerationStep(),

               # Stitch images using the generated positions
               ImageStitchingStep()
           ],
           name="Well-Documented Pipeline"
       )

4. **Leverage dynamic directory resolution**: Set directories at the pipeline level and only override when necessary

   .. code-block:: python

       # Set input and output at the pipeline level
       pipeline = Pipeline(
           input_dir=orchestrator.workspace_path,
           output_dir=orchestrator.plate_path.parent / f"{orchestrator.plate_path.name}_stitched",
           steps=[
               # First step uses pipeline's input_dir automatically
               Step(func=IP.stack_percentile_normalize),

               # Subsequent steps use previous step's output automatically
               PositionGenerationStep(),
               ImageStitchingStep()
           ],
           name="Directory Resolution Pipeline"
       )

5. **Use coherent data flow**: Let each step's output feed into the next step's input

   .. code-block:: python

       pipeline = Pipeline(
           input_dir=orchestrator.workspace_path,
           steps=[
               # Process images
               Step(
                   func=IP.stack_percentile_normalize,
                   # No output_dir specified - uses input_dir by default
               ),

               # Generate positions using processed images
               PositionGenerationStep(),

               # Stitch using processed images
               ImageStitchingStep()
           ],
           name="Coherent Flow Pipeline"
       )

6. **Organize by experiment type**: Create separate scripts for different experiment types

   .. code-block:: python

       # brightfield_pipeline.py
       def create_brightfield_pipeline(plate_path):
           """Create a pipeline optimized for brightfield images."""
           # Brightfield-specific pipeline code...
           return pipeline

       # fluorescence_pipeline.py
       def create_fluorescence_pipeline(plate_path):
           """Create a pipeline optimized for fluorescence images."""
           # Fluorescence-specific pipeline code...
           return pipeline

7. **Version control your scripts**: Keep track of changes to your pipeline configurations

   - Store pipeline scripts in a version control system like Git
   - Use descriptive commit messages when making changes
   - Consider using semantic versioning for pipeline releases

.. _best-practices-directory:

Directory Management Best Practices
--------------------------------

For comprehensive information on directory structure and management in EZStitcher, see :doc:`../concepts/directory_structure`. Here are the key best practices:

1. **Always specify input_dir for the first step**:

   - Use ``orchestrator.workspace_path`` to ensure processing happens on workspace copies
   - This protects original data from modification

   .. code-block:: python

       pipeline = Pipeline(
           steps=[
               Step(
                   func=IP.stack_percentile_normalize,
                   input_dir=orchestrator.workspace_path  # Specify input_dir for first step
               ),
               # Subsequent steps...
           ]
       )

2. **Specify output_dir only when you need a specific directory structure**:

   - For example, when you need to save results in a specific location
   - When you need to reference the output directory from outside the pipeline

   .. code-block:: python

       # Specify output_dir when you need a specific location
       processed_dir = plate_path.parent / f"{plate_path.name}_processed"
       pipeline = Pipeline(
           steps=[
               Step(
                   func=IP.stack_percentile_normalize,
                   input_dir=orchestrator.workspace_path,
                   output_dir=processed_dir  # Specify output_dir for reference later
               ),
               # Subsequent steps...
           ]
       )

3. **Don't specify input_dir for subsequent steps**:

   - Each step's output directory automatically becomes the next step's input directory
   - This reduces code verbosity and potential for errors

   For more details on directory resolution logic, see :ref:`directory-resolution`.

4. **Use consistent directory naming**:

   - Follow the default naming conventions when possible
   - Or configure custom suffixes through PipelineConfig for consistent naming
   - This makes it easier to understand the directory structure

   For more information on custom directory structures, see :ref:`directory-custom-structures`.

5. **Consider performance**:

   - In-place processing (using the same directory for input and output) is more efficient
   - This is the default behavior for steps after the first step
   - Only use separate input and output directories when necessary

.. _best-practices-specialized-steps:

Specialized Steps Best Practices
-----------------------------

For comprehensive information on specialized steps in EZStitcher, see :doc:`../concepts/specialized_steps`. Here are the key best practices:

1. **Directory Resolution**:

   - Let EZStitcher automatically resolve directories when possible
   - Only specify directories when you need a specific directory structure
   - The ``ImageStitchingStep`` follows the standard directory resolution logic, using the previous step's output directory as its input
   - You can explicitly set ``input_dir=orchestrator.workspace_path`` to use original images for stitching instead of processed images

   For more details on specialized step directory resolution, see :ref:`specialized-steps-directory-resolution`.

2. **Step Order**:

   - Place ``PositionGenerationStep`` after image processing steps
   - Place ``ImageStitchingStep`` after ``PositionGenerationStep``
   - This ensures that position generation works with processed images

   For more information on typical stitching workflows, see :ref:`typical-stitching-workflows`.

3. **Pipeline Integration**:

   - Use specialized steps within a pipeline for automatic directory resolution
   - The steps will automatically access the orchestrator through the context

4. **Multi-Channel Processing**:

   - When working with multiple channels, create a composite image before position generation
   - This ensures that position files are generated based on all available information

   .. code-block:: python

       pipeline = Pipeline(
           steps=[
               # Process channels
               Step(
                   func=IP.stack_percentile_normalize,
                   variable_components=['channel'],
                   input_dir=orchestrator.workspace_path
               ),

               # Create composite image for position generation
               Step(
                   func=IP.create_composite,
                   variable_components=['channel']
               ),

               # Generate positions using the composite image
               PositionGenerationStep(),

               # Stitch images
               ImageStitchingStep()
           ]
       )

.. _best-practices-function-handling:

Function Handling Best Practices
-----------------------------

For comprehensive information on function handling patterns in EZStitcher, see :doc:`../concepts/function_handling`. Here are the key best practices:

1. **Use the tuple pattern for function arguments**:

   - Always use ``(func, kwargs)`` to pass arguments to functions
   - This is clearer and more maintainable than other approaches

   .. code-block:: python

       # Good: Use tuple pattern for arguments
       step = Step(
           func=(IP.stack_percentile_normalize, {
               'low_percentile': 1.0,
               'high_percentile': 99.0
           })
       )

       # Avoid: Using separate arguments parameter
       # This approach is not supported and will cause errors
       step = Step(
           func=IP.stack_percentile_normalize,
           args={'low_percentile': 1.0, 'high_percentile': 99.0}  # Don't do this
       )

   For more details on function argument patterns, see :ref:`function-arguments`.

2. **Keep function lists focused**:

   - When using lists of functions, each function should have a clear purpose
   - Avoid overly long lists that are difficult to understand

   .. code-block:: python

       # Good: Focused list with clear purpose for each function
       step = Step(
           func=[
               (stack(IP.tophat), {'size': 15}),          # Remove background
               (stack(IP.sharpen), {'sigma': 1.0}),       # Enhance features
               IP.stack_percentile_normalize              # Normalize intensities
           ]
       )

   For more information on function lists, see :ref:`function-lists`.

3. **Use descriptive variable names in processing functions**:

   - When defining custom processing functions, use descriptive parameter names
   - This makes the code more readable and maintainable

   .. code-block:: python

       # Good: Descriptive parameter names
       def enhance_nuclei(images, blur_sigma=1.0, tophat_size=15):
           """Enhance nuclei in DAPI images."""
           processed = []
           for img in images:
               # Apply gaussian blur to reduce noise
               blurred = gaussian(img, sigma=blur_sigma)
               # Apply tophat to remove background
               bg_removed = tophat(blurred, size=tophat_size)
               processed.append(bg_removed)
           return processed

4. **Document complex processing chains**:

   - Add comments explaining what each function in a chain does
   - This is especially important for complex processing

   .. code-block:: python

       step = Step(
           func=[
               (stack(IP.tophat), {'size': 15}),          # Remove background
               (stack(IP.sharpen), {'sigma': 1.0}),       # Enhance features
               (IP.stack_percentile_normalize, {          # Normalize intensities
                   'low_percentile': 1.0,
                   'high_percentile': 99.0
               })
           ]
       )

   For more information on advanced function patterns, see :ref:`function-dictionaries` and :ref:`function-advanced-patterns`.

.. _best-practices-performance:

Performance Best Practices
-----------------------

1. **Use multithreading for multiple wells**:

   - Set ``num_workers`` in PipelineConfig to process multiple wells in parallel
   - This can significantly improve performance

   .. code-block:: python

       # Create configuration with multithreaded processing
       config = PipelineConfig(
           num_workers=4  # Use 4 worker threads
       )

       # Create orchestrator with multithreading
       orchestrator = PipelineOrchestrator(
           config=config,
           plate_path=plate_path
       )

2. **Minimize disk I/O**:

   - Avoid unnecessary saving and loading of intermediate results
   - Use in-place processing when possible

   .. code-block:: python

       # Good: In-place processing (no output_dir specified)
       step = Step(
           func=IP.stack_percentile_normalize,
           input_dir=orchestrator.workspace_path
           # No output_dir - uses input_dir by default
       )

       # Avoid: Unnecessary separate output directory
       step = Step(
           func=IP.stack_percentile_normalize,
           input_dir=orchestrator.workspace_path,
           output_dir=orchestrator.workspace_path  # Unnecessary - same as input_dir
       )

3. **Balance memory usage and performance**:

   - Processing large images can consume significant memory
   - Consider using smaller tiles or processing in batches for very large datasets

   .. code-block:: python

       # Process wells in batches to manage memory usage
       all_wells = ["A01", "A02", "A03", "B01", "B02", "B03"]
       batch_size = 2

       for i in range(0, len(all_wells), batch_size):
           batch_wells = all_wells[i:i+batch_size]
           orchestrator.run(
               pipelines=[pipeline],
               well_filter=batch_wells
           )

4. **Profile your pipelines**:

   - Use Python profiling tools to identify bottlenecks
   - Focus optimization efforts on the slowest parts of your pipeline

   .. code-block:: python

       import cProfile
       import pstats

       # Profile pipeline execution
       profiler = cProfile.Profile()
       profiler.enable()

       # Run your pipeline
       orchestrator.run(pipelines=[pipeline])

       profiler.disable()
       stats = pstats.Stats(profiler).sort_stats('cumtime')
       stats.print_stats(20)  # Print top 20 time-consuming functions
