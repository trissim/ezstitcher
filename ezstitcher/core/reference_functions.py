"""
Reference functions for Z-stack processing.

This module provides functions for converting 3D Z-stacks to 2D reference images,
as well as utilities for adapting various function types to the reference function interface.
"""

import numpy as np
from typing import List, Callable, Union, Optional, Any
import inspect
import logging

logger = logging.getLogger(__name__)

def max_projection(stack: List[np.ndarray]) -> np.ndarray:
    """
    Create a maximum intensity projection from a Z-stack.
    
    Args:
        stack: List of 2D images representing a Z-stack
        
    Returns:
        2D image representing the maximum intensity projection
    """
    return np.max(np.array(stack), axis=0)

def mean_projection(stack: List[np.ndarray]) -> np.ndarray:
    """
    Create a mean intensity projection from a Z-stack.
    
    Args:
        stack: List of 2D images representing a Z-stack
        
    Returns:
        2D image representing the mean intensity projection
    """
    return np.mean(np.array(stack), axis=0).astype(stack[0].dtype)

def create_reference_function(func: Callable, function_type: Optional[str] = None, 
                             projection_type: str = 'max') -> Callable:
    """
    Create a reference function from any image processing function.
    
    This function adapts various types of functions to the reference function interface,
    which takes a Z-stack (list of images) and returns a single 2D image.
    
    Args:
        func: The function to adapt
        function_type: 'image' for single-image functions, 'stack' for stack functions.
                      If None, will try to determine automatically.
        projection_type: For image functions, how to combine the processed images.
                        Options: 'max', 'mean', 'first', 'last'
    
    Returns:
        A function that takes a stack and returns a single image
    """
    # Define projection functions
    projections = {
        'max': lambda stack: np.max(np.array(stack), axis=0),
        'mean': lambda stack: np.mean(np.array(stack), axis=0).astype(stack[0].dtype),
        'first': lambda stack: stack[0],
        'last': lambda stack: stack[-1]
    }
    
    # If projection_type is invalid, use max
    if projection_type not in projections:
        logger.warning(f"Invalid projection_type '{projection_type}'. Using 'max' instead.")
        projection_type = 'max'
    
    # If function type is explicitly provided, use it
    if function_type == 'stack':
        return func
    
    if function_type == 'image':
        def adapter(stack):
            processed_stack = [func(img) for img in stack]
            return projections[projection_type](processed_stack)
        
        # Copy metadata from original function
        adapter.__name__ = f"{func.__name__}_adapted"
        adapter.__doc__ = f"Adapted version of {func.__name__} that works on stacks."
        return adapter
    
    # Try to determine automatically
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
                    processed_stack = [func(img) for img in stack]
                    return projections[projection_type](processed_stack)
                
                # Copy metadata from original function
                adapter.__name__ = f"{func.__name__}_adapted"
                adapter.__doc__ = f"Adapted version of {func.__name__} that works on stacks."
                return adapter
            
            # If it returns something else, we can't use it
            raise ValueError(f"Function {func.__name__} doesn't return a 2D image when given a single image")
            
        except Exception as e:
            # If both attempts fail, raise an error
            raise ValueError(f"Cannot adapt function {func.__name__}: {str(e)}")

def create_best_focus_function(method: str = "combined") -> Callable:
    """
    Create a function that selects the best focused image from a Z-stack.
    
    Args:
        method: Focus detection method to use
        
    Returns:
        Function that takes a Z-stack and returns the best focused image
    """
    # Import here to avoid circular imports
    from ezstitcher.core.config import FocusAnalyzerConfig
    from ezstitcher.core.focus_analyzer import FocusAnalyzer
    
    # Create a FocusAnalyzer with the specified method
    config = FocusAnalyzerConfig(method=method)
    analyzer = FocusAnalyzer(config)
    
    def best_focus(stack):
        """
        Select the best focused image from a Z-stack.
        
        Args:
            stack: List of 2D images representing a Z-stack
            
        Returns:
            2D image representing the best focused plane
        """
        # Find the best focused image
        best_img, _, _ = analyzer.select_best_focus(stack, method=method)
        return best_img
    
    return best_focus

# Standard reference functions
standard_reference_functions = {
    'max_projection': max_projection,
    'mean_projection': mean_projection,
    'best_focus': create_best_focus_function()
}

def get_reference_function(reference_method: Union[str, Callable]) -> Callable:
    """
    Get a reference function from a string name or callable.
    
    Args:
        reference_method: Name of a standard reference function or a callable
        
    Returns:
        A function that takes a stack and returns a single image
    """
    if isinstance(reference_method, str):
        if reference_method in standard_reference_functions:
            return standard_reference_functions[reference_method]
        elif reference_method == 'best_focus':
            return create_best_focus_function()
        else:
            raise ValueError(f"Unknown reference method: {reference_method}")
    elif callable(reference_method):
        return create_reference_function(reference_method)
    else:
        raise ValueError(f"reference_method must be a string or callable, got {type(reference_method)}")
