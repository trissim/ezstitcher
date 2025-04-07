# EZStitcher Tests

This directory contains tests for the EZStitcher package. The tests cover core functionality including:

1. **Image Processing** - Tests for image processing functions like blur, edge detection, and normalization
2. **Focus Detection** - Tests for focus quality detection algorithms
3. **Stitching** - Tests for image stitching and position calculation
4. **Z-Stack Handling** - Tests for Z-stack organization and projection creation
5. **Integration** - Tests for the full workflow

## Running the Tests

### Setup

Before running the tests, make sure you have installed the package in development mode:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-cov
```

### Running All Tests

You can run all tests using the provided script:

```bash
python run_tests.py
```

Or using pytest directly:

```bash
pytest tests/
```

### Running Specific Tests

To run a specific test file:

```bash
pytest tests/test_image_processing.py
```

To run a specific test class:

```bash
pytest tests/test_image_processing.py::TestImageProcessing
```

To run a specific test method:

```bash
pytest tests/test_image_processing.py::TestImageProcessing::test_blur
```

### Test Coverage

To generate a test coverage report:

```bash
pytest --cov=ezstitcher tests/
```

For a detailed HTML report:

```bash
pytest --cov=ezstitcher --cov-report=html tests/
```

## Troubleshooting

If you encounter issues with NumPy or other dependencies:

1. Make sure you're using the correct Python version (3.6+)
2. Try reinstalling NumPy: `pip uninstall numpy && pip install numpy`
3. Check that all dependencies are installed: `pip install -r requirements.txt`

## Adding New Tests

When adding new tests:

1. Follow the existing pattern of test files
2. Use descriptive test method names that explain what is being tested
3. Include assertions that verify both the functionality and edge cases
4. Add the new test file to this README if it tests a new component
