# EZStitcher Tests

This directory contains tests for the EZStitcher package. The tests cover core functionality including:

1. **Image Processing** - Tests for image processing functions like blur, edge detection, and normalization
2. **Focus Detection** - Tests for focus quality detection algorithms
3. **Stitching** - Tests for image stitching and position calculation
4. **Z-Stack Handling** - Tests for Z-stack organization and projection creation
5. **Integration** - Tests for the full workflow

## Test Organization

### Unit Tests

These tests focus on testing individual components in isolation:

- **`test_file_system_manager.py`**: Tests for the FileSystemManager class, which handles file system operations.
- **`test_stitcher.py`**: Tests for the Stitcher class, which implements the core stitching algorithms.
- **`test_zstack_processor.py`**: Tests for the ZStackProcessor class, which handles Z-stack processing.

### Configuration Tests

These tests focus on the configuration system:

- **`test_config.py`**: Tests for the legacy dataclass-based configuration system. May be deprecated in the future.
- **`test_pydantic_config.py`**: Tests for the newer Pydantic-based configuration system, which provides validation, serialization, and hierarchical configuration management.

### Integration Tests

These tests focus on testing how components work together:

- **`test_integration.py`**: Integration tests for the full workflow using the legacy configuration system.
- **`test_config_integration.py`**: Integration tests for the configuration system's integration with the processing pipeline.

### Documentation and Example Tests

These tests verify that the examples in the documentation work correctly:

- **`test_documentation_examples.py`**: Tests for the examples in the documentation, using synthetic microscopy data.

### Synthetic Workflow Tests

These tests use synthetic microscopy data to test the library's functionality in realistic scenarios:

- **`test_synthetic_workflow_class_based.py`**: Comprehensive tests using synthetic microscopy data with the class-based implementation.

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

## Test Organization Guidelines

1. **Isolation**: Each test should be isolated from other tests. Tests should not depend on the state created by other tests.
2. **Cleanup**: Tests should clean up after themselves, removing any temporary files or directories they create.
3. **Documentation**: Tests should include docstrings explaining what they're testing and how they're testing it.
4. **Synthetic Data**: Tests should use synthetic data rather than real data to ensure reproducibility and to avoid dependencies on external data.
5. **Test Directory Structure**: Each test should create its own test data directory with a unique name to avoid interference between tests.

## Known Test Redundancies

There are some redundancies in the test suite that should be addressed in future refactoring:

1. **Configuration Tests**: `test_config.py` and `test_pydantic_config.py` test similar functionality but for different configuration implementations. As the codebase transitions to Pydantic models, `test_config.py` may be deprecated.

2. **Overlapping Test Coverage**: `test_documentation_examples.py` and `test_synthetic_workflow_class_based.py` have some overlapping test cases for Z-stack processing. These could potentially be consolidated.

3. **Integration Test Redundancy**: `test_integration.py` and `test_config_integration.py` both test aspects of the configuration system's integration with the processing pipeline.
