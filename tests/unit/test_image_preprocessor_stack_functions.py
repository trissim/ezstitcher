import unittest
import numpy as np
from ezstitcher.core.image_preprocessor import ImagePreprocessor


class TestImagePreprocessorStackFunctions(unittest.TestCase):
    """Test the ImagePreprocessor's ability to handle both single-image and stack functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        
        # Create a simple test stack
        self.test_stack = [
            np.ones((10, 10), dtype=np.uint16) * 100,  # First image
            np.ones((10, 10), dtype=np.uint16) * 200,  # Second image
            np.ones((10, 10), dtype=np.uint16) * 300   # Third image
        ]

    def test_single_image_function(self):
        """Test applying a function that expects a single image."""
        # Define a function that only works on single images
        def double_intensity(image):
            """Double the intensity of a single image."""
            return image * 2
        
        # Apply to stack
        result = self.preprocessor.apply_function_to_stack(self.test_stack, double_intensity)
        
        # Check that we got a list back with the right number of images
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(self.test_stack))
        
        # Check that each image was processed correctly
        self.assertEqual(np.mean(result[0]), 200)
        self.assertEqual(np.mean(result[1]), 400)
        self.assertEqual(np.mean(result[2]), 600)

    def test_stack_function(self):
        """Test applying a function that expects a stack."""
        # Define a function that works on stacks
        def normalize_stack(stack):
            """Normalize a stack of images to the range [0, 1]."""
            stack_array = np.array(stack)
            min_val = np.min(stack_array)
            max_val = np.max(stack_array)
            return (stack_array - min_val) / (max_val - min_val)
        
        # Apply to stack
        result = self.preprocessor.apply_function_to_stack(self.test_stack, normalize_stack)
        
        # Check that we got a numpy array back
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 10, 10))
        
        # Check that the stack was normalized correctly
        self.assertAlmostEqual(np.min(result), 0.0)
        self.assertAlmostEqual(np.max(result), 1.0)
        self.assertAlmostEqual(np.mean(result[0]), 0.0)
        self.assertAlmostEqual(np.mean(result[1]), 0.5)
        self.assertAlmostEqual(np.mean(result[2]), 1.0)

    def test_built_in_stack_function(self):
        """Test applying a built-in stack function."""
        # Apply stack_percentile_normalize
        result = self.preprocessor.apply_function_to_stack(
            self.test_stack, 
            lambda stack: self.preprocessor.stack_percentile_normalize(stack, low_percentile=0, high_percentile=100)
        )
        
        # Check that we got a numpy array back
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 10, 10))
        
        # Check that the stack was normalized correctly
        self.assertEqual(np.min(result), 0)
        self.assertEqual(np.max(result), 65535)

    def test_mixed_function_list(self):
        """Test applying a list of functions with mixed types."""
        # Define a list of functions
        def double_intensity(image):
            """Double the intensity of a single image."""
            return image * 2
        
        def normalize_stack(stack):
            """Normalize a stack of images to the range [0, 1]."""
            stack_array = np.array(stack)
            min_val = np.min(stack_array)
            max_val = np.max(stack_array)
            return (stack_array - min_val) / (max_val - min_val)
        
        # Apply functions in sequence
        intermediate = self.preprocessor.apply_function_to_stack(self.test_stack, double_intensity)
        final = self.preprocessor.apply_function_to_stack(intermediate, normalize_stack)
        
        # Check that we got a numpy array back
        self.assertIsInstance(final, np.ndarray)
        self.assertEqual(final.shape, (3, 10, 10))
        
        # Check that the stack was processed correctly
        self.assertAlmostEqual(np.min(final), 0.0)
        self.assertAlmostEqual(np.max(final), 1.0)

    def test_function_with_exception(self):
        """Test applying a function that raises an exception for stacks."""
        # Define a function that raises an exception for stacks
        def fail_on_stack(input_data):
            """Raise an exception if input is a stack, otherwise double the intensity."""
            if isinstance(input_data, list) or isinstance(input_data, np.ndarray) and input_data.ndim > 2:
                raise ValueError("Cannot process stack")
            return input_data * 2
        
        # Apply to stack
        result = self.preprocessor.apply_function_to_stack(self.test_stack, fail_on_stack)
        
        # Check that we got a list back with the right number of images
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(self.test_stack))
        
        # Check that each image was processed correctly
        self.assertEqual(np.mean(result[0]), 200)
        self.assertEqual(np.mean(result[1]), 400)
        self.assertEqual(np.mean(result[2]), 600)

    def test_multiple_stack_functions(self):
        """Test applying multiple stack functions in sequence."""
        # Define a list of functions that work on stacks
        def normalize_stack(stack):
            """Normalize a stack of images to the range [0, 1]."""
            stack_array = np.array(stack)
            min_val = np.min(stack_array)
            max_val = np.max(stack_array)
            return (stack_array - min_val) / (max_val - min_val)
        
        def invert_stack(stack):
            """Invert a stack of images."""
            return 1.0 - stack
        
        # Apply functions in sequence
        intermediate = self.preprocessor.apply_function_to_stack(self.test_stack, normalize_stack)
        final = self.preprocessor.apply_function_to_stack(intermediate, invert_stack)
        
        # Check that we got a numpy array back
        self.assertIsInstance(final, np.ndarray)
        self.assertEqual(final.shape, (3, 10, 10))
        
        # Check that the stack was processed correctly
        self.assertAlmostEqual(np.min(final), 0.0)
        self.assertAlmostEqual(np.max(final), 1.0)
        self.assertAlmostEqual(np.mean(final[0]), 1.0)
        self.assertAlmostEqual(np.mean(final[1]), 0.5)
        self.assertAlmostEqual(np.mean(final[2]), 0.0)


if __name__ == '__main__':
    unittest.main()
