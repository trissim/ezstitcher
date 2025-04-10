"""
Z-stack reference adapter module for ezstitcher.

This module provides a class for adapting various functions to work with Z-stacks.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Union, Any, Callable

from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.config import FocusConfig

logger = logging.getLogger(__name__)


class ZStackReferenceAdapter:
    """
    Adapts various functions to work with Z-stacks.

    This class provides utilities for:
    - Creating reference functions from string names or callables
    - Adapting single-image functions to work with Z-stacks
    - Preprocessing Z-stacks before applying reference functions
    """
    def __init__(self, image_preprocessor=None, focus_analyzer=None):
        """
        Initialize the ZStackReferenceAdapter.

        Args:
            image_preprocessor: Image preprocessor for projection creation
            focus_analyzer: Focus analyzer for focus detection
        """
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        self.focus_analyzer = focus_analyzer or FocusAnalyzer(FocusConfig())

    def create_reference_function(self, func_or_name: Union[str, Callable], focus_method='combined') -> Callable:
        """
        Create a reference function from a string name or callable.

        This function adapts various types of functions to the reference function interface,
        which takes a Z-stack (list of images) and returns a single 2D image.

        Args:
            func_or_name: String name of a standard function or a callable
                Standard names: "max_projection", "mean_projection", "best_focus"
                Can also be a custom function that takes a Z-stack and returns a 2D image
            focus_method: Focus detection method to use when using best_focus

        Returns:
            A function that takes a stack and returns a single image
        """
        # Handle string names
        if isinstance(func_or_name, str):
            if func_or_name == "max_projection":
                return self.adapt_function(self.image_preprocessor.max_projection)
            elif func_or_name == "mean_projection":
                return self.adapt_function(self.image_preprocessor.mean_projection)
            elif func_or_name == "best_focus":
                return lambda stack: self.focus_analyzer.select_best_focus(stack, method=focus_method)[0]
            else:
                raise ValueError(f"Unknown reference function name: {func_or_name}")

        # Handle callables
        if callable(func_or_name):
            return self.adapt_function(func_or_name)

        raise ValueError(f"Reference function must be a string or callable, got {type(func_or_name)}")

    def adapt_function(self, func: Callable) -> Callable:
        """
        Adapt a function to the reference function interface.

        This allows both:
        - Functions that take a single image and return a processed image
        - Functions that take a stack and return a single image

        to be used as reference functions.

        Args:
            func: The function to adapt

        Returns:
            A function that takes a stack and returns a single image
        """
        # Try to determine if the function works on stacks or single images
        try:
            # Create a small test stack (2 tiny images)
            test_stack = [np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)]

            # Try calling the function with the stack
            result = func(test_stack)

            # If it returns a single image, it's a stack function
            if isinstance(result, np.ndarray) and result.ndim == 2:
                logger.debug(f"Function {func.__name__} detected as stack function")
                return func

            # If it returns something else, we can't use it directly
            raise ValueError(f"Function {func.__name__} doesn't return a 2D image when given a stack")

        except Exception:
            # If it fails, try with a single image
            try:
                # Try with a single image
                result = func(test_stack[0])

                # If it returns an image, it's an image function
                if isinstance(result, np.ndarray) and result.ndim == 2:
                    logger.debug(f"Function {func.__name__} detected as image function")

                    def adapter(stack):
                        # Apply the function to each image in the stack
                        processed_stack = [func(img) for img in stack]
                        # Return the max projection of the processed stack
                        return np.max(np.array(processed_stack), axis=0)

                    # Copy metadata from original function
                    adapter.__name__ = f"{func.__name__}_adapted"
                    adapter.__doc__ = f"Adapted version of {func.__name__} that works on stacks."
                    return adapter

                # If it returns something else, we can't use it
                raise ValueError(f"Function {func.__name__} doesn't return a 2D image when given a single image")

            except Exception as e:
                # If both attempts fail, raise an error
                raise ValueError(f"Cannot adapt function {func.__name__}: {str(e)}")

    def preprocess_stack(self, stack, channel, preprocessing_funcs=None):
        """
        Apply preprocessing to each image in a Z-stack.

        Args:
            stack: List of images in the Z-stack
            channel: Channel identifier for selecting the preprocessing function
            preprocessing_funcs: Dictionary mapping channels to preprocessing functions

        Returns:
            List of preprocessed images
        """
        if preprocessing_funcs is None:
            preprocessing_funcs = {}

        if channel in preprocessing_funcs:
            func = preprocessing_funcs[channel]
            return [func(img) for img in stack]
        return stack
