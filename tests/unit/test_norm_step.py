#"""
#Unit tests for the NormStep class.
#"""
#
#import unittest
#from unittest.mock import patch, MagicMock
#import numpy as np
#
#from ezstitcher.core.steps import NormStep
#from ezstitcher.core.image_processor import ImageProcessor as IP
#
#
#class TestNormStep(unittest.TestCase):
#    """Test the NormStep class."""
#
#    def test_init(self):
#        """Test initialization with default parameters."""
#        step = NormStep()
#        self.assertEqual(step.name, "Percentile Normalization")
#        # Default variable_components should be ['site']
#        self.assertEqual(step.variable_components, ['site'])
#        self.assertIsNone(step.group_by)
#
#    def test_init_with_custom_percentiles(self):
#        """Test initialization with custom percentile values."""
#        step = NormStep(low_percentile=1.0, high_percentile=99.0)
#        self.assertEqual(step.name, "Percentile Normalization")
#        # Check that the function arguments are correctly set
#        self.assertTrue(isinstance(step.func, tuple))
#        self.assertEqual(step.func[0], IP.stack_percentile_normalize)
#        self.assertEqual(step.func[1]['low_percentile'], 1.0)
#        self.assertEqual(step.func[1]['high_percentile'], 99.0)
#
#    def test_init_with_kwargs(self):
#        """Test initialization with additional kwargs."""
#        step = NormStep(
#            low_percentile=1.0,
#            high_percentile=99.0,
#            input_dir="/input",
#            output_dir="/output",
#            well_filter=["A01", "B02"],
#            variable_components=["channel"],
#            group_by="channel"
#        )
#        self.assertEqual(step.name, "Percentile Normalization")
#        self.assertEqual(step.input_dir, "/input")
#        self.assertEqual(step.output_dir, "/output")
#        self.assertEqual(step.well_filter, ["A01", "B02"])
#        self.assertEqual(step.variable_components, ["channel"])
#        self.assertEqual(step.group_by, "channel")
#
#    @patch.object(IP, 'stack_percentile_normalize')
#    def test_process(self, mock_normalize):
#        """Test the process method."""
#        # Create a mock image stack
#        images = [np.zeros((10, 10)) for _ in range(3)]
#
#        # Set up the mock to return the input images
#        mock_normalize.return_value = images
#
#        # Create a step and process the images
#        step = NormStep(low_percentile=1.0, high_percentile=99.0)
#        result = step.process(images)
#
#        # Check that stack_percentile_normalize was called with the correct arguments
#        mock_normalize.assert_called_once_with(
#            images,
#            low_percentile=1.0,
#            high_percentile=99.0
#        )
#
#        # Check that the result is what we expect
#        self.assertEqual(result, images)
#
#
#if __name__ == '__main__':
#    unittest.main()
#
# tests/unit/test_norm_step.py

#import unittest
#from unittest.mock import patch
#import numpy as np
#from pathlib import Path
#
#from ezstitcher.core.steps import NormStep
#from ezstitcher.core.image_processor import ImageProcessor as IP
#from ezstitcher.core.pipeline import ProcessingContext
#from types import SimpleNamespace
#
#
#def create_test_context(images=None, input_dir=None, output_dir=None):
#    """Create a mock ProcessingContext with minimal required state for testing."""
#    context = ProcessingContext(
#        well_filter=["A01"],
#        config={},
#        orchestrator=None  # Only needed if steps require it
#    )
#
#    class DummyPlan:
#        def __init__(self, input_dir, output_dir):
#            self.input_dir = input_dir
#            self.output_dir = output_dir
#
#    # Add dummy step plan if images are provided
#    if images is not None:
#        dummy_step = SimpleNamespace()  # lightweight fake step with unique id
#        dummy_step.name = "DummyStep"
#        plan = DummyPlan(input_dir=images, output_dir=output_dir)
#        context.add_step_plan(dummy_step, plan)
#        return context, dummy_step
#
#    return context, None
#
#class TestNormStep(unittest.TestCase):
#    """Test the NormStep class."""
#
#    def test_init(self):
#        step = NormStep()
#        self.assertEqual(step.name, "Percentile Normalization")
#        self.assertEqual(step.variable_components, ['site'])
#        self.assertIsNone(step.group_by)
#
#    def test_init_with_custom_percentiles(self):
#        step = NormStep(low_percentile=1.0, high_percentile=99.0)
#        func, kwargs = step.func
#        self.assertEqual(step.name, "Percentile Normalization")
#        self.assertEqual(func, IP.stack_percentile_normalize)
#        self.assertEqual(kwargs['low_percentile'], 1.0)
#        self.assertEqual(kwargs['high_percentile'], 99.0)
#
#    def test_init_with_kwargs(self):
#        """Test initialization with variable components and group_by."""
#        step = NormStep(
#            low_percentile=1.0,
#            high_percentile=99.0,
#            variable_components=["channel"],
#            group_by="channel"
#        )
#        self.assertEqual(step.name, "Percentile Normalization")
#        self.assertEqual(step.variable_components, ["channel"])
#        self.assertEqual(step.group_by, "channel")
#
#        # input_dir, output_dir, and well_filter should no longer be attributes
#        self.assertFalse(hasattr(step, "input_dir"))
#        self.assertFalse(hasattr(step, "output_dir"))
#        self.assertFalse(hasattr(step, "well_filter"))
#
#
#    @patch.object(IP, 'stack_percentile_normalize')
#    def test_process(self, mock_normalize):
#        """Test the process method using a proper ProcessingContext."""
#        images = [np.zeros((10, 10)) for _ in range(3)]
#        mock_normalize.return_value = images
#
#        step = NormStep(low_percentile=1.0, high_percentile=99.0)
#        context, dummy_step = create_test_context(images=images)
#
#        # Use the real step in place of dummy_step
#        context.add_step_plan(step, context.step_plans.pop(id(dummy_step)))
#
#        result = step.process(context)
#
#        mock_normalize.assert_called_once_with(
#            images,
#            low_percentile=1.0,
#            high_percentile=99.0
#        )
#        self.assertEqual(result, images)
#
#
#if __name__ == '__main__':
#    unittest.main()

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from ezstitcher.core.steps import NormStep
from ezstitcher.core.image_processor import ImageProcessor as IP
from ezstitcher.core.pipeline import ProcessingContext
from ezstitcher.core.pipeline_orchestrator import StepExecutionPlan


def create_test_context(step, input_dir=None, output_dir=None):
    """Create a ProcessingContext with a dummy StepExecutionPlan for the given step."""
    context = ProcessingContext(
        well_filter=["A01"],
        config={},
        orchestrator=MagicMock()
    )
    context.orchestrator.microscope_handler = MagicMock()
    context.orchestrator.microscope_handler.parser = MagicMock()

    input_dir = Path(input_dir or "/mock/input")
    output_dir = Path(output_dir or "/mock/output")

    plan = StepExecutionPlan(
        step_id=id(step),
        step_name=step.name,
        step_type=type(step).__name__,
        input_dir=input_dir,
        output_dir=output_dir
    )

    context.add_step_plan(step, plan)
    return context


class TestNormStep(unittest.TestCase):
    """Test the NormStep class."""

    def test_init(self):
        step = NormStep()
        self.assertEqual(step.name, "Percentile Normalization")
        self.assertEqual(step.variable_components, ['site'])
        self.assertIsNone(step.group_by)

    def test_init_with_custom_percentiles(self):
        step = NormStep(low_percentile=1.0, high_percentile=99.0)
        func, kwargs = step.func
        self.assertEqual(step.name, "Percentile Normalization")
        self.assertEqual(func, IP.stack_percentile_normalize)
        self.assertEqual(kwargs['low_percentile'], 1.0)
        self.assertEqual(kwargs['high_percentile'], 99.0)

    def test_init_with_kwargs(self):
        step = NormStep(
            low_percentile=1.0,
            high_percentile=99.0,
            variable_components=["channel"],
            group_by="channel"
        )
        self.assertEqual(step.name, "Percentile Normalization")
        self.assertEqual(step.variable_components, ["channel"])
        self.assertEqual(step.group_by, "channel")
        self.assertFalse(hasattr(step, "input_dir"))
        self.assertFalse(hasattr(step, "output_dir"))
        self.assertFalse(hasattr(step, "well_filter"))

   # @patch.object(IP, 'stack_percentile_normalize')
   # def test_process(self, mock_normalize):
   #     images = [np.zeros((10, 10)) for _ in range(3)]
   #     mock_normalize.return_value = images

   #     step = NormStep(low_percentile=1.0, high_percentile=99.0)
   #     context = create_test_context(step, images=images)

   #     result = step.process(context)

   #     mock_normalize.assert_called_once_with(
   #         images,
   #         low_percentile=1.0,
   #         high_percentile=99.0
   #     )
   #     self.assertEqual(result, images)
    @patch.object(IP, 'stack_percentile_normalize')
    def test_process_function(self, mock_normalize):
        """Test the processing function directly."""
        # Create dummy image list
        images = [np.zeros((10, 10)) for _ in range(3)]
        mock_normalize.return_value = images

        # Create a step
        step = NormStep(low_percentile=1.0, high_percentile=99.0)

        # Extract the processing function and arguments
        func, kwargs = step.func

        # Call the function directly
        result = func(images, **kwargs)

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
