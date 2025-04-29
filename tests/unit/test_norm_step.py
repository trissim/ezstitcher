"""
Unit tests for the NormStep class.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from ezstitcher.core.steps import NormStep
from ezstitcher.core.image_processor import ImageProcessor as IP


class TestNormStep(unittest.TestCase):
    """Test the NormStep class."""

    def test_init(self):
        """Test initialization with default parameters."""
        step = NormStep()
        self.assertEqual(step.name, "Percentile Normalization")
        # Default variable_components should be ['site']
        self.assertEqual(step.variable_components, ['site'])
        self.assertIsNone(step.group_by)

    def test_init_with_custom_percentiles(self):
        """Test initialization with custom percentile values."""
        step = NormStep(low_percentile=1.0, high_percentile=99.0)
        self.assertEqual(step.name, "Percentile Normalization")
        # Check that the function arguments are correctly set
        self.assertTrue(isinstance(step.func, tuple))
        self.assertEqual(step.func[0], IP.stack_percentile_normalize)
        self.assertEqual(step.func[1]['low_percentile'], 1.0)
        self.assertEqual(step.func[1]['high_percentile'], 99.0)

    def test_init_with_kwargs(self):
        """Test initialization with additional kwargs."""
        step = NormStep(
            low_percentile=1.0,
            high_percentile=99.0,
            input_dir="/input",
            output_dir="/output",
            well_filter=["A01", "B02"],
            variable_components=["channel"],
            group_by="channel"
        )
        self.assertEqual(step.name, "Percentile Normalization")
        self.assertEqual(step.input_dir, "/input")
        self.assertEqual(step.output_dir, "/output")
        self.assertEqual(step.well_filter, ["A01", "B02"])
        self.assertEqual(step.variable_components, ["channel"])
        self.assertEqual(step.group_by, "channel")

    @patch.object(IP, 'stack_percentile_normalize')
    def test_process(self, mock_normalize):
        """Test the process method."""
        # Create a mock image stack
        images = [np.zeros((10, 10)) for _ in range(3)]
        
        # Set up the mock to return the input images
        mock_normalize.return_value = images
        
        # Create a step and process the images
        step = NormStep(low_percentile=1.0, high_percentile=99.0)
        result = step.process(images)
        
        # Check that stack_percentile_normalize was called with the correct arguments
        mock_normalize.assert_called_once_with(
            images, 
            low_percentile=1.0, 
            high_percentile=99.0
        )
        
        # Check that the result is what we expect
        self.assertEqual(result, images)


if __name__ == '__main__':
    unittest.main()
