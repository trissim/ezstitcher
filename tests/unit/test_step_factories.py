"""
Unit tests for the step factory classes.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from ezstitcher.core.step_factories import ZFlatStep, FocusStep, CompositeStep
from ezstitcher.core.image_processor import ImageProcessor as IP
from ezstitcher.core.focus_analyzer import FocusAnalyzer


class TestZFlatStep(unittest.TestCase):
    """Test the ZFlatStep class."""

    def test_init(self):
        """Test initialization with default parameters."""
        step = ZFlatStep()
        self.assertEqual(step.variable_components, ['z_index'])
        self.assertIsNone(step.group_by)
        self.assertEqual(step.name, "Max Projection")

    def test_init_with_method(self):
        """Test initialization with a specific method."""
        step = ZFlatStep(method="mean")
        self.assertEqual(step.variable_components, ['z_index'])
        self.assertIsNone(step.group_by)
        self.assertEqual(step.name, "Mean Projection")

    def test_init_with_invalid_method(self):
        """Test initialization with an invalid method."""
        with self.assertRaises(ValueError):
            ZFlatStep(method="invalid")

    @patch.object(IP, 'create_projection')
    def test_process(self, mock_create_projection):
        """Test the process method."""
        # Create a mock image stack
        images = [np.zeros((10, 10)) for _ in range(3)]

        # Create a step and process the images
        step = ZFlatStep(method="max")
        step.process(images)

        # Check that create_projection was called with the correct arguments
        mock_create_projection.assert_called_once_with(images, method="max_projection")


class TestFocusStep(unittest.TestCase):
    """Test the FocusStep class."""

    def test_init(self):
        """Test initialization with default parameters."""
        step = FocusStep()
        self.assertEqual(step.variable_components, ['z_index'])
        self.assertIsNone(step.group_by)
        self.assertEqual(step.name, "Best Focus (combined)")

    def test_init_with_metric(self):
        """Test initialization with a specific metric."""
        step = FocusStep(focus_options={'metric': 'laplacian'})
        self.assertEqual(step.variable_components, ['z_index'])
        self.assertIsNone(step.group_by)
        self.assertEqual(step.name, "Best Focus (laplacian)")

    @patch.object(IP, 'create_projection')
    @patch.object(FocusAnalyzer, '__init__', return_value=None)
    def test_process(self, mock_focus_analyzer_init, mock_create_projection):
        """Test the process method."""
        # Create a mock image stack
        images = [np.zeros((10, 10)) for _ in range(3)]

        # Create a step and process the images
        step = FocusStep(focus_options={'metric': 'laplacian'})
        step.process(images)

        # Check that create_projection was called with the correct arguments
        mock_create_projection.assert_called_once()
        args, kwargs = mock_create_projection.call_args
        self.assertEqual(args[0], images)
        self.assertEqual(kwargs['method'], 'best_focus')
        self.assertIn('focus_analyzer', kwargs)


class TestCompositeStep(unittest.TestCase):
    """Test the CompositeStep class."""

    def test_init(self):
        """Test initialization with default parameters."""
        step = CompositeStep()
        self.assertEqual(step.variable_components, ['channel'])
        self.assertIsNone(step.group_by)
        self.assertEqual(step.name, "Channel Composite")

    def test_init_with_weights(self):
        """Test initialization with specific weights."""
        weights = [0.7, 0.3]
        step = CompositeStep(weights=weights)
        self.assertEqual(step.variable_components, ['channel'])
        self.assertIsNone(step.group_by)
        self.assertEqual(step.name, "Channel Composite")

    @patch.object(IP, 'create_composite')
    def test_process(self, mock_create_composite):
        """Test the process method."""
        # Create a mock image stack
        images = [np.zeros((10, 10)) for _ in range(2)]

        # Create a step and process the images
        weights = [0.7, 0.3]
        step = CompositeStep(weights=weights)
        step.process(images)

        # Check that create_composite was called with the correct arguments
        mock_create_composite.assert_called_once_with(images, weights=weights)
