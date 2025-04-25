"""
Test the Step class with function tuples.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os

from ezstitcher.core.steps import Step
from ezstitcher.core.pipeline import ProcessingContext
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
from ezstitcher.core.config import PipelineConfig


# Define test functions
def add_value(images, value=0):
    """Add a value to all images."""
    return [img + value for img in images]

def multiply_value(images, value=1):
    """Multiply all images by a value."""
    return [img * value for img in images]

def subtract_value(images, value=0):
    """Subtract a value from all images."""
    return [img - value for img in images]


class TestStepFunctionTuples:
    """Test the Step class with function tuples."""

    @pytest.fixture
    def test_images(self):
        """Create test images."""
        return [np.ones((10, 10), dtype=np.uint16) * 10]  # Single image with all pixels = 10

    @pytest.fixture
    def test_context(self):
        """Create a test context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create input directory with a test image
            input_dir = temp_path / "input"
            input_dir.mkdir()

            # Create output directory
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Create a minimal orchestrator
            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(
                config=config,
                plate_path=str(input_dir)
            )

            # Create context
            context = ProcessingContext(
                input_dir=input_dir,
                output_dir=output_dir,
                orchestrator=orchestrator
            )

            yield context

            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_single_function_single_args(self, test_images):
        """Test a single function with a single args dictionary."""
        # Create step with single function and args as a tuple
        step = Step(
            func=(add_value, {"value": 5})
        )

        # Process images
        result = step._apply_processing(test_images)

        # Check result (10 + 5 = 15)
        assert np.array_equal(result[0], np.ones((10, 10), dtype=np.uint16) * 15)

    def test_list_functions_single_args(self, test_images):
        """Test a list of functions with a single args dictionary."""
        # Create step with list of functions where each function has the same args
        step = Step(
            func=[
                (add_value, {"value": 2}),
                (multiply_value, {"value": 2})
            ]
        )

        # Process images
        result = step._apply_processing(test_images)

        # Check result (10 + 2 = 12, then 12 * 2 = 24)
        assert np.array_equal(result[0], np.ones((10, 10), dtype=np.uint16) * 24)

    def test_list_functions_list_args(self, test_images):
        """Test a list of functions with a matching list of args dictionaries."""
        # Create step with list of function tuples with different args
        step = Step(
            func=[
                (add_value, {"value": 5}),         # Add 5
                (multiply_value, {"value": 3}),    # Multiply by 3
                (subtract_value, {"value": 10})    # Subtract 10
            ]
        )

        # Process images
        result = step._apply_processing(test_images)

        # Check result (10 + 5 = 15, then 15 * 3 = 45, then 45 - 10 = 35)
        assert np.array_equal(result[0], np.ones((10, 10), dtype=np.uint16) * 35)

    def test_list_functions_partial_list_args(self, test_images):
        """Test a list of functions with a partial list of args dictionaries."""
        # Create step with list of function tuples and one function without args
        step = Step(
            func=[
                (add_value, {"value": 5}),         # Add 5
                (multiply_value, {"value": 3}),    # Multiply by 3
                subtract_value                     # Use default value=0
            ]
        )

        # Process images
        result = step._apply_processing(test_images)

        # Check result (10 + 5 = 15, then 15 * 3 = 45, then 45 - 0 = 45)
        assert np.array_equal(result[0], np.ones((10, 10), dtype=np.uint16) * 45)

    def test_single_function_tuple(self, test_images):
        """Test a single function tuple."""
        # Create step with a single function tuple
        step = Step(
            func=(add_value, {"value": 5})
        )

        # Process images
        result = step._apply_processing(test_images)

        # Check result (10 + 5 = 15)
        assert np.array_equal(result[0], np.ones((10, 10), dtype=np.uint16) * 15)
