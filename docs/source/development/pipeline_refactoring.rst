Pipeline Orchestrator Refactoring
============================

This document describes the recent refactoring of the Pipeline Orchestrator, focusing on the new modular pattern processing approach.

Overview
--------

The Pipeline Orchestrator has been refactored to improve modularity, reduce code duplication, and enhance flexibility in handling different pattern types. The core of this refactoring is the new ``process_patterns_with_variable_components`` method, which provides a unified way to process patterns with variable components.

Key Components
-------------

1. **Pattern Detection**: Automatically detects patterns with variable components in a directory
2. **Pattern Grouping**: Optionally groups patterns by a specific component (e.g., channel, z-index)
3. **Processing Functions**: Applies processing functions to patterns, with support for component-specific functions
4. **Unified Processing Flow**: Handles both grouped and flat patterns with the same code path

The ``process_patterns_with_variable_components`` Method
------------------------------------------------------

This method is the cornerstone of the new approach, providing a flexible way to process patterns with variable components:

.. code-block:: python

    def process_patterns_with_variable_components(self, input_dir, output_dir, well_filter=None,
                                                variable_components=None, group_by=None,
                                                processing_funcs=None, processing_args=None):
        """
        Detect patterns with variable components and process them flexibly.

        Args:
            input_dir (str or Path): Input directory containing images
            output_dir (str or Path): Output directory for processed images
            well_filter (list, optional): List of wells to include
            variable_components (list, optional): Components to make variable (e.g., ['site', 'z_index'])
            group_by (str, optional): How to group patterns (e.g., 'channel', 'z_index', 'well')
            processing_funcs (callable, list, dict, optional): Processing functions to apply
            processing_args (dict, optional): Additional arguments to pass to processing functions

        Returns:
            dict: Dictionary mapping wells to processed file paths
        """

The method works by:

1. Auto-detecting patterns with specified variable components
2. Preparing patterns and functions using the helper method ``_prepare_patterns_and_functions``
3. Processing each group of patterns with its corresponding function
4. Returning a dictionary mapping wells to processed file paths

Helper Methods
-------------

The refactoring introduced several helper methods to support the main processing flow:

``_prepare_patterns_and_functions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method prepares patterns and processing functions for processing:

.. code-block:: python

    def _prepare_patterns_and_functions(self, patterns, processing_funcs, component='default'):
        """
        Prepare patterns and processing functions for processing.
        
        This function handles two main tasks:
        1. Ensuring patterns are in a component-keyed dictionary format
        2. Determining which processing functions to use for each component
        """

It ensures that patterns are in a dictionary format and determines which processing functions to use for each component. The method is optimized to handle cases where patterns and functions are already properly structured.

``process_tiles``
~~~~~~~~~~~~~~~

This method processes tiles using the specified processing functions:

.. code-block:: python

    def process_tiles(self, input_dir, output_dir, patterns, processing_funcs=None, **kwargs):
        """
        Unified processing using zstack_processor.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            patterns: List of file patterns
            processing_funcs: Processing functions to apply (optional)
            **kwargs: Additional arguments to pass to processing functions
        """

It handles loading images, applying processing functions, and saving the processed images.

Workflow Examples
---------------

Here are some examples of how to use the new modular pattern processing approach:

Processing Reference Images
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Process reference images for a well
    processed_files = pipeline.process_patterns_with_variable_components(
        input_dir=dirs['input'],
        output_dir=dirs['processed'],
        well_filter=[well],
        variable_components=['site'],
        group_by='channel',
        processing_funcs=processing_funcs
    )

Creating Composite Images
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create composites from multiple channels
    composite_files = pipeline.process_patterns_with_variable_components(
        input_dir=dirs['processed'],
        output_dir=dirs['processed'],
        well_filter=[well],
        variable_components=['channel'],
        group_by='site',
        processing_funcs=pipeline.image_preprocessor.create_composite,
        processing_args={'weights': weights}
    )

Flattening Z-Stacks
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Flatten Z-stacks using projection
    flattened_files = pipeline.process_patterns_with_variable_components(
        input_dir=dirs['processed'],
        output_dir=dirs['processed'],
        well_filter=[well],
        variable_components=['z_index'],
        processing_funcs=pipeline.image_preprocessor.create_projection,
        processing_args={
            'method': 'max_projection',
            'focus_analyzer': pipeline.focus_analyzer
        }
    )

Benefits of the Refactoring
-------------------------

1. **Modularity**: Each component of the pipeline is now more modular and can be used independently
2. **Reduced Code Duplication**: Common pattern processing logic is now centralized
3. **Flexibility**: The same method can handle different pattern types and processing functions
4. **Simplified API**: Users can accomplish complex tasks with a single method call
5. **Improved Error Handling**: Better error handling and validation throughout the pipeline

The refactoring has significantly improved the maintainability and extensibility of the codebase while preserving backward compatibility.
