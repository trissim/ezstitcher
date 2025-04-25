"""
Test the new function tuple feature in the Step class.
"""

import numpy as np
from ezstitcher.core.steps import Step

def test_function_tuple():
    """Test that a function tuple (function, kwargs) works correctly."""
    # Create a simple test function
    def add_value(images, value=0):
        """Add a value to all images."""
        return [img + value for img in images]
    
    # Create test images
    images = [np.ones((10, 10)) for _ in range(3)]
    
    # Test with a function tuple
    step = Step(func=(add_value, {'value': 5}))
    result = step._apply_processing(images)
    
    # Check that the value was added
    assert np.all(result[0] == 6)
    
def test_function_list_with_tuples():
    """Test that a list of function tuples works correctly."""
    # Create simple test functions
    def add_value(images, value=0):
        """Add a value to all images."""
        return [img + value for img in images]
    
    def multiply_value(images, factor=1):
        """Multiply all images by a factor."""
        return [img * factor for img in images]
    
    # Create test images
    images = [np.ones((10, 10)) for _ in range(3)]
    
    # Test with a list of function tuples
    step = Step(func=[
        (add_value, {'value': 5}),
        (multiply_value, {'factor': 2})
    ])
    result = step._apply_processing(images)
    
    # Check that the operations were applied in sequence
    # First add 5, then multiply by 2, so result should be (1+5)*2 = 12
    assert np.all(result[0] == 12)
    
def test_mixed_function_list():
    """Test that a list with both plain functions and tuples works correctly."""
    # Create simple test functions
    def add_one(images):
        """Add 1 to all images."""
        return [img + 1 for img in images]
    
    def multiply_value(images, factor=1):
        """Multiply all images by a factor."""
        return [img * factor for img in images]
    
    # Create test images
    images = [np.ones((10, 10)) for _ in range(3)]
    
    # Test with a mixed list
    step = Step(func=[
        add_one,  # No kwargs
        (multiply_value, {'factor': 3})
    ])
    result = step._apply_processing(images)
    
    # Check that the operations were applied in sequence
    # First add 1, then multiply by 3, so result should be (1+1)*3 = 6
    assert np.all(result[0] == 6)
