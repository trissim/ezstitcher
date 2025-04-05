# Testing EZStitcher

This document provides instructions on how to run tests for the EZStitcher package.

## Prerequisites

Before running the tests, make sure you have installed the package in development mode:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-cov
```

## Running Tests

### Running All Tests

To run all tests, use the `run_tests.py` script:

```bash
python run_tests.py
```

This will discover and run all tests in the `tests` directory.

### Running Specific Tests

To run specific test modules, use the `unittest` module:

```bash
# Run a specific test module
python -m unittest tests.test_image_processor

# Run a specific test class
python -m unittest tests.test_image_processor.TestImageProcessor

# Run a specific test method
python -m unittest tests.test_image_processor.TestImageProcessor.test_blur
```

### Running Tests with Coverage

To run tests with coverage reporting, use `pytest` with the `pytest-cov` plugin:

```bash
# Run all tests with coverage
pytest --cov=ezstitcher tests/

# Generate an HTML coverage report
pytest --cov=ezstitcher --cov-report=html tests/
```

The HTML coverage report will be generated in the `htmlcov` directory. Open `htmlcov/index.html` in a web browser to view the report.

## Test Structure

The tests are organized as follows:

- `tests/test_image_processor.py`: Tests for the `ImageProcessor` class
- `tests/test_focus_detector.py`: Tests for the `FocusDetector` class
- `tests/test_z_stack_manager.py`: Tests for the `ZStackManager` class
- `tests/test_stitcher_manager.py`: Tests for the `StitcherManager` class
- `tests/test_main.py`: Tests for the main module functions
- `tests/test_imports_new.py`: Tests for imports from the new class-based API
- `tests/test_synthetic_workflow.py`: Integration tests using synthetic data

## Writing New Tests

When writing new tests, follow these guidelines:

1. Create a new test module in the `tests` directory with a name starting with `test_`.
2. Use the `unittest` framework for consistency with existing tests.
3. Create a test class that inherits from `unittest.TestCase`.
4. Write test methods with names starting with `test_`.
5. Use descriptive method names that indicate what is being tested.
6. Include docstrings that describe the purpose of each test.
7. Use assertions to verify expected behavior.
8. Clean up any temporary files or resources in the `tearDown` method.

Example:

```python
import unittest
import numpy as np
from ezstitcher.core import ImageProcessor

class TestImageProcessor(unittest.TestCase):
    """Test the ImageProcessor class."""
    
    def setUp(self):
        """Set up test data."""
        self.test_image = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    
    def test_blur(self):
        """Test the blur method."""
        # Apply blur
        blurred = ImageProcessor.blur(self.test_image, sigma=1)
        
        # Check that the output has the same shape and dtype
        self.assertEqual(blurred.shape, self.test_image.shape)
        self.assertEqual(blurred.dtype, self.test_image.dtype)
        
        # Check that the blur reduced the variance
        self.assertLess(np.var(blurred), np.var(self.test_image))
```

## Continuous Integration

The tests are automatically run on GitHub Actions when changes are pushed to the repository. The workflow is defined in `.github/workflows/tests.yml`.

## Troubleshooting

If you encounter issues running the tests, try the following:

1. Make sure you have installed the package in development mode.
2. Make sure you have installed all dependencies.
3. Check that your Python version is compatible with the package.
4. Try running the tests with increased verbosity: `python -m unittest -v tests.test_module`.
5. If a specific test is failing, try running it in isolation to see if it's a dependency issue.

If you still have issues, please open an issue on the GitHub repository.
