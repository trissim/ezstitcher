====
Step
====

Overview
-------

A ``Step`` is a single processing operation that can be applied to images. The base ``Step`` class provides:

* Image loading and saving
* Processing function application
* Variable component handling (e.g., channels, z-indices)
* Group-by functionality for processing related images together

Creating a Basic Step
-------------------

.. code-block:: python

    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_processor import ImageProcessor as IP

    # Create a basic processing step
    step = Step(
        name="Image Enhancement",
        func=IP.stack_percentile_normalize,
        variable_components=['channel'],
        group_by='channel',
        input_dir=orchestrator.workspace_path,  # Specify input_dir for the first step
        # output_dir is automatically determined
    )

.. _step-parameters:

Step Parameters
-------------

* ``name``: Human-readable name for the step
* ``func``: The processing function(s) to apply (see :doc:`function_handling`)
* ``variable_components``: Components that vary across files (e.g., 'z_index', 'channel')
* ``group_by``: How to group files for processing (e.g., 'channel', 'site')
* ``input_dir``: The input directory (optional, can inherit from pipeline)
* ``output_dir``: The output directory (optional, can inherit from pipeline)
* ``well_filter``: Wells to process (optional, can inherit from pipeline)

For practical examples of how to use these parameters in different scenarios, see:

* :doc:`../user_guide/basic_usage` - Basic examples of step parameters
* :doc:`../user_guide/intermediate_usage` - Examples of variable_components and group_by
* :doc:`../user_guide/advanced_usage` - Advanced examples of func parameter
* :doc:`../user_guide/best_practices` - Best practices for step parameters

Processing Arguments
------------------

Processing arguments are passed directly with the function using the tuple pattern ``(func, kwargs)``. For detailed information about function handling patterns, see :ref:`function-handling`.

.. code-block:: python

    # Pass arguments to a function
    step = Step(
        name="Z-Stack Flattening",
        func=(IP.create_projection, {'method': 'max_projection'}),
        variable_components=['z_index'],
        input_dir=orchestrator.workspace_path
    )

This pattern can be used with:
* Single functions (:ref:`function-single`, :ref:`function-with-arguments`)
* Lists of functions (:ref:`function-lists`, :ref:`function-lists-with-arguments`)
* Dictionaries of functions (:ref:`function-dictionaries`, :ref:`function-dictionary-tuples`)
* Mixed function types (:ref:`function-mixed-types`)

.. note::
   Always use the tuple pattern ``(func, kwargs)`` to pass arguments to processing functions.
   This is the recommended approach for all function arguments.

Step Initialization Best Practices
--------------------------------

When initializing steps, it's important to follow best practices for directory specification.

For detailed information on step initialization best practices, directory resolution, and directory flow, see :doc:`directory_structure`.

.. _variable-components:

Variable Components
-----------------

The ``variable_components`` parameter specifies which components will be grouped together for processing. It determines how images are organized into stacks before being passed to the processing function.

**Key concept**: Images that share the same values for all components *except* the variable component will be grouped together into a stack.

In most cases, you don't need to set this explicitly as it defaults to ``['site']``, but there are specific cases where you should change it.

For practical examples of how to use variable_components in different scenarios, see:

* :doc:`../user_guide/intermediate_usage` - Examples for Z-stack processing and channel compositing
* :doc:`../user_guide/advanced_usage` - Advanced examples with custom functions

.. code-block:: python

    # When flattening Z-stacks, set variable_components to 'z_index'
    # This groups images with the same site, channel, etc. but different z_index values
    # The function will receive a stack of images with varying z_index values
    step = Step(
        name="Z-Stack Flattening",
        func=(IP.create_projection, {'method': 'max_projection'}),
        variable_components=['z_index']  # Group images by z_index
    )

    # When creating composite images, set variable_components to 'channel'
    # This groups images with the same site, z_index, etc. but different channel values
    # The function will receive a stack of images with varying channel values

    # Without weights (equal weighting for all channels)
    step = Step(
        func=IP.create_composite,
        variable_components=['channel']  # Group images by channel
    )

    # With custom weights (70% channel 1, 30% channel 2)
    step = Step(
        func=(IP.create_composite, {'weights': [0.7, 0.3]}),  # Pass weights as a list
        variable_components=['channel']  # Group images by channel
    )

    # For most other operations, the default 'site' is appropriate
    # This groups images with the same channel, z_index, etc. but different site values
    # The function will receive a stack of images with varying site values
    step = Step(
        name="Enhance Images",
        func=stack(IP.sharpen)
        # variable_components defaults to ['site']
    )

.. _group-by:

Group By
-------

The ``group_by`` parameter is only used when providing a dictionary of functions. It specifies what component the keys in your function dictionary correspond to.

For practical examples of how to use group_by in different scenarios, see:

* :doc:`../user_guide/intermediate_usage` - Examples for channel-specific processing
* :doc:`../user_guide/advanced_usage` - Advanced examples with dictionaries of functions

.. code-block:: python

    # When using a dictionary of channel-specific functions
    step = Step(
        name="Channel-Specific Processing",
        func={"1": process_dapi, "2": process_calcein},
        # variable_components defaults to ['site']
        group_by='channel'  # Keys "1" and "2" correspond to channel values
    )

**Key concept**: The ``group_by`` parameter tells the Step what the keys in the function dictionary represent.

In this example:
- ``group_by='channel'`` means the keys in the function dictionary ("1" and "2") correspond to channel values
- Images with channel="1" will be processed by ``process_dapi``
- Images with channel="2" will be processed by ``process_calcein``

**Parameter Relationships and Constraints**:

1. ``group_by`` is **only needed when using a dictionary of functions**. It's not needed for single functions or lists of functions.

2. ``group_by`` should **NEVER be the same as** ``variable_components``:

   This is a critical rule that must be followed to avoid logical errors. When ``variable_components=['channel']``, it means we're processing each channel separately. When ``group_by='channel'``, it means we're grouping functions by channel. If these were the same, it would create a logical contradiction in how the images are processed.

   .. code-block:: python

       # CORRECT: variable_components and group_by are different
       step = Step(
           name="Channel-Specific Processing",
           func={"1": process_dapi, "2": process_calcein},
           variable_components=['site'],  # Process each site separately
           group_by='channel'  # Keys "1" and "2" correspond to channel values
       )

       # INCORRECT: variable_components and group_by are the same
       # This will lead to logical errors and should never be done
       step = Step(
           name="Incorrect Setup",
           func={"1": process_dapi, "2": process_calcein},
           variable_components=['channel'],  # Process each channel separately
           group_by='channel'  # Keys "1" and "2" correspond to channel values
       )

3. ``group_by`` is typically only set when ``variable_components`` is left at its default value of ``['site']``:

   .. code-block:: python

       # Typical pattern: variable_components defaults to ['site'], group_by is set to 'channel'
       step = Step(
           name="Channel-Specific Processing",
           func={"1": process_dapi, "2": process_calcein},
           # variable_components defaults to ['site']
           group_by='channel'  # Keys "1" and "2" correspond to channel values
       )

4. ``input_dir`` must be specified for the first step in a pipeline, typically using ``orchestrator.workspace_path``.

5. ``output_dir`` is optional and will be automatically determined if not specified.

6. ``well_filter`` is optional and will inherit from the pipeline's context if not specified.

Step Parameters Best Practices
----------------------------

When configuring step parameters, follow these best practices:

1. **Use Descriptive Names**:
   - Choose clear, descriptive names for your steps
   - This makes pipelines easier to understand and debug

2. **Function Handling**:
   - Use the tuple pattern ``(func, kwargs)`` for passing arguments to functions
   - Use lists of functions for sequential processing
   - Use dictionaries of functions with ``group_by`` for component-specific processing
   - Use the ``stack()`` utility for adapting single-image functions

3. **Variable Components**:
   - Set ``variable_components=['z_index']`` when flattening Z-stacks
   - Set ``variable_components=['channel']`` when creating composite images
   - Leave at default ``['site']`` for most other operations

4. **Directory Management**:
   - Always specify ``input_dir`` for the first step, using ``orchestrator.workspace_path``
   - Let EZStitcher handle directory resolution for subsequent steps
   - Only specify ``output_dir`` when you need a specific directory structure

5. **Parameter Validation**:
   - Ensure ``group_by`` is never the same as ``variable_components``
   - Only use ``group_by`` with dictionary functions
   - Verify that all required parameters are specified
