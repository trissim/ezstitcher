.. _function-handling:

=================
Function Handling
=================

The Step class supports several patterns for processing functions, providing flexibility in how images are processed. This page explains the different patterns available.

.. _function-single:

Single Function
-------------

A callable that takes a list of images and returns a list of processed images:

.. code-block:: python

    # Single function
    step = Step(
        name="Normalize Images",
        func=IP.stack_percentile_normalize
        # variable_components defaults to ['site']
    )

.. _function-with-arguments:
.. _function-arguments:

Function with Arguments
---------------------

A tuple containing a function and its arguments:

.. code-block:: python

    # Function tuple (function, kwargs)
    step = Step(
        name="Normalize Images",
        func=(IP.stack_percentile_normalize, {
            'low_percentile': 0.1,
            'high_percentile': 99.9
        })
        # variable_components defaults to ['site']
    )

.. _function-lists:

List of Functions
---------------

A sequence of functions applied one after another:

.. code-block:: python

    from ezstitcher.core.utils import stack

    # List of functions
    step = Step(
        name="Enhance Images",
        func=[
            stack(IP.sharpen),              # First sharpen the images
            IP.stack_percentile_normalize   # Then normalize the intensities
        ]
        # variable_components defaults to ['site']
    )

The ``stack()`` utility function adapts a single-image function to work with stacks of images. It applies the original function to each image in the stack and returns a new stack of processed images. This is particularly useful when you want to use functions from libraries like scikit-image that operate on single images.

.. _function-lists-with-arguments:

List of Functions with Arguments
-----------------------------

A sequence of function tuples applied in sequence:

.. code-block:: python

    from ezstitcher.core.utils import stack

    # List of function tuples
    step = Step(
        name="Enhance Images",
        func=[
            (stack(IP.sharpen), {'sigma': 1.0, 'amount': 2.0}),  # Sharpen with specific parameters
            (IP.stack_percentile_normalize, {                    # Normalize with specific parameters
                'low_percentile': 0.1,
                'high_percentile': 99.9
            })
        ]
        # variable_components defaults to ['site']
    )

.. _function-dictionaries:

Dictionary of Functions
---------------------

A mapping from component values to functions, allowing different processing for different components:

.. code-block:: python

    # Define channel-specific processing functions
    def process_dapi(stack):
        """Process DAPI channel images."""
        stack = IP.stack_percentile_normalize(stack)
        return [IP.tophat(img) for img in stack]

    def process_calcein(stack):
        """Process Calcein channel images."""
        return [IP.tophat(img) for img in stack]

    # Dictionary of functions
    step = Step(
        name="Channel-Specific Processing",
        func={
            "1": process_dapi,      # Apply process_dapi to channel 1
            "2": process_calcein    # Apply process_calcein to channel 2
        },
        # variable_components defaults to ['site']
        group_by='channel'  # Specifies that keys "1" and "2" refer to channel values
    )

.. _function-dictionary-tuples:

Dictionary of Function Tuples
---------------------------

A mapping from component values to function tuples:

.. code-block:: python

    # Dictionary of function tuples
    step = Step(
        name="Channel-Specific Processing",
        func={
            "1": (process_dapi, {'threshold': 100}),      # Apply process_dapi to channel 1 with args
            "2": (process_calcein, {'radius': 5})         # Apply process_calcein to channel 2 with args
        },
        # variable_components defaults to ['site']
        group_by='channel'  # Specifies that keys "1" and "2" refer to channel values
    )

.. _function-dictionary-lists:

Dictionary of Lists with Mixed Function Types
------------------------------------------

A mapping from component values to lists that can contain both plain functions and function tuples:

.. code-block:: python

    # Dictionary of lists with mixed function types
    step = Step(
        name="Advanced Channel Processing",
        func={
            "1": [  # Process channel 1 with a sequence of functions
                stack(IP.tophat),                              # Function without args
                (stack(IP.sharpen), {'sigma': 1.0}),           # Function with args
                IP.stack_percentile_normalize                  # Function without args
            ],
            "2": [  # Process channel 2 with a different sequence
                (stack(IP.gaussian_blur), {'sigma': 2.0}),     # Function with args
                IP.stack_percentile_normalize                  # Function without args
            ],
            "3": (IP.stack_percentile_normalize, {            # Single function tuple for channel 3
                'low_percentile': 0.5,
                'high_percentile': 99.5
            })
        },
        # variable_components defaults to ['site']
        group_by='channel'  # Specifies that keys "1", "2", and "3" refer to channel values
    )

When using a dictionary of functions:
- The `group_by` parameter is required to specify what component the dictionary keys refer to
- Each key in the dictionary corresponds to a specific value of that component
- Files are processed by the function that matches their component value
- For example, with `group_by='channel'`, files with channel="1" are processed by the function at key "1"

.. _function-mixed-types:
.. _function-advanced-patterns:

Mixed Function Types
------------------

You can mix plain functions and function tuples in the same list or dictionary. The tuple pattern ``(func, args)`` is optional even within lists or dictionaries:

.. code-block:: python

    # Mixed function types in a list
    step = Step(
        name="Mixed Processing",
        func=[
            stack(IP.tophat),                              # Function without args
            (stack(IP.sharpen), {'sigma': 1.0}),           # Function with args
            IP.stack_percentile_normalize                  # Function without args
        ]
        # variable_components defaults to ['site']
    )

    # Mixed function types in a dictionary
    step = Step(
        name="Mixed Channel Processing",
        func={
            "1": stack(IP.tophat),                         # Function without args
            "2": (stack(IP.sharpen), {'sigma': 1.0}),      # Function with args
            "3": IP.stack_percentile_normalize             # Function without args
        },
        # variable_components defaults to ['site']
        group_by='channel'  # Specifies that keys "1", "2", and "3" refer to channel values
    )

.. _function-when-to-use:

When to Use Each Pattern
----------------------

* **Single Function**: When you need to apply the same processing to all images with default parameters
* **Function with Arguments**: When you need to apply a single function with specific parameters
* **List of Functions**: When you need to apply multiple processing steps in sequence with default parameters
* **List of Functions with Arguments**: When you need to apply multiple processing steps with specific parameters
* **Dictionary of Functions**: When you need to apply different processing to different components with default parameters
* **Dictionary of Function Tuples**: When you need to apply different processing to different components with specific parameters

For comprehensive best practices for function handling, see :ref:`best-practices-function-handling` in the :doc:`../user_guide/best_practices` guide.

.. _function-stack-utility:

The stack() Utility Function
--------------------------

The ``stack()`` utility function is a key tool for adapting single-image functions to work with stacks of images:

.. code-block:: python

    from ezstitcher.core.utils import stack
    from skimage.filters import gaussian

    # Use stack() to adapt a single-image function to work with a stack
    step = Step(
        name="Gaussian Blur",
        func=stack(gaussian),  # Apply gaussian blur to each image in the stack
        # variable_components defaults to ['site']
    )

    # You can also use stack() with arguments
    step = Step(
        name="Gaussian Blur with Parameters",
        func=(stack(gaussian), {'sigma': 2.0}),  # Apply gaussian blur with sigma=2.0
        # variable_components defaults to ['site']
    )

    # stack() can be used in lists and dictionaries
    step = Step(
        name="Mixed Processing",
        func=[
            stack(gaussian),                      # Apply gaussian blur to each image
            (stack(IP.sharpen), {'sigma': 1.0}),  # Then sharpen each image
            IP.stack_percentile_normalize         # Then normalize the entire stack
        ]
        # variable_components defaults to ['site']
    )

**When to use stack()**:

* Use ``stack()`` when you have a function that operates on a single image but you need to apply it to a stack of images
* Use ``stack()`` with functions from libraries like scikit-image that operate on single images
* Use ``stack()`` when you want to apply the same operation to each image in a stack independently

**How stack() works**:

1. It takes a function that operates on a single image as input
2. It returns a new function that operates on a stack of images
3. The new function applies the original function to each image in the stack
4. It returns a new stack containing the processed images

This allows you to seamlessly integrate single-image functions into EZStitcher's stack-based processing pipeline.
