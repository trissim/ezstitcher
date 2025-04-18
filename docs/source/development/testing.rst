Testing
=======

This guide explains how to test EZStitcher.

Test Organization
--------------

The tests are organized into the following directories:

- **`unit/`**: Unit tests for individual components
- **`integration/`**: Integration tests for the full workflow
- **`generators/`**: Synthetic data generators for testing

Running Tests
-----------

To run all tests:

.. code-block:: bash

    pytest

To run a specific test file:

.. code-block:: bash

    pytest tests/unit/test_image_preprocessor.py

To run a specific test class:

.. code-block:: bash

    pytest tests/unit/test_image_preprocessor.py::TestImagePreprocessor

To run a specific test method:

.. code-block:: bash

    pytest tests/unit/test_image_preprocessor.py::TestImagePreprocessor::test_blur

Test Coverage
-----------

To generate a test coverage report:

.. code-block:: bash

    pytest --cov=ezstitcher tests/

For a detailed HTML report:

.. code-block:: bash

    pytest --cov=ezstitcher --cov-report=html tests/

Writing Tests
-----------

When writing tests for EZStitcher, follow these guidelines:

1. **Use pytest fixtures**: Use fixtures to set up test data and dependencies
2. **Test one thing at a time**: Each test should test one specific functionality
3. **Use descriptive names**: Test names should describe what is being tested
4. **Use assertions**: Use assertions to verify expected behavior
5. **Clean up after tests**: Clean up any temporary files or directories created during tests

Here's an example of a unit test:

.. code-block:: python

    import pytest
    import numpy as np
    from ezstitcher.core.image_preprocessor import ImagePreprocessor

    class TestImagePreprocessor:
        """Tests for the ImagePreprocessor class."""

        def test_blur(self):
            """Test the blur method."""
            # Create a test image
            image = np.ones((100, 100), dtype=np.uint16) * 1000
            image[40:60, 40:60] = 5000  # Add a bright square

            # Apply blur
            blurred = ImagePreprocessor.blur(image, sigma=2.0)

            # Verify that the image was blurred
            assert blurred.shape == image.shape
            assert blurred.dtype == image.dtype
            assert np.mean(blurred[40:60, 40:60]) < 5000  # Blurring should reduce the intensity
            assert np.mean(blurred[40:60, 40:60]) > 1000  # But it should still be brighter than the background

        def test_normalize(self):
            """Test the normalize method."""
            # Create a test image
            image = np.ones((100, 100), dtype=np.uint16) * 1000
            image[40:60, 40:60] = 5000  # Add a bright square

            # Apply normalization
            normalized = ImagePreprocessor.normalize(image, target_min=0, target_max=65535)

            # Verify that the image was normalized
            assert normalized.shape == image.shape
            assert normalized.dtype == image.dtype
            assert np.min(normalized) == 0
            assert np.max(normalized) == 65535

Here's an example of an integration test:

.. code-block:: python

    import pytest
    import os
    import numpy as np
    from pathlib import Path
    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator

    @pytest.fixture
    def test_data_dir():
        """Fixture for test data directory."""
        return Path("tests/data/test_plate")

    def test_pipeline_basic(test_data_dir):
        """Test basic pipeline functionality."""
        # Create configuration
        config = PipelineConfig(
            reference_channels=["1"],
            reference_flatten="max_projection",
            stitch_flatten="max_projection"
        )

        # Create pipeline
        pipeline = PipelineOrchestrator(config)

        # Run pipeline
        result = pipeline.run(test_data_dir)

        # Verify that the pipeline ran successfully
        assert result is True

        # Verify that output files were created
        processed_dir = test_data_dir.parent / f"{test_data_dir.name}{config.processed_dir_suffix}"
        stitched_dir = test_data_dir.parent / f"{test_data_dir.name}{config.stitched_dir_suffix}"

        assert processed_dir.exists()
        assert stitched_dir.exists()

        # Verify that stitched images were created
        stitched_files = list(stitched_dir.glob("*.tif"))
        assert len(stitched_files) > 0

Generating Test Data
-----------------

EZStitcher includes a synthetic data generator for testing:

.. code-block:: bash

    python -m ezstitcher.tests.generators.generate_synthetic_data output_dir --grid-size 3 3 --wavelengths 2 --z-stack 3

You can also generate test data programmatically:

.. code-block:: python

    from ezstitcher.tests.generators.generate_synthetic_data import generate_plate

    # Generate a synthetic plate
    generate_plate(
        output_dir="tests/data/synthetic_plate",
        grid_size=(3, 3),
        wavelengths=2,
        z_stack=3,
        wells=["A01", "A02"],
        image_size=(512, 512),
        overlap=0.1,
        noise_level=0.05
    )

Mocking
------

When testing components that depend on external resources, use mocking to isolate the component being tested:

.. code-block:: python

    import pytest
    from unittest.mock import Mock, patch
    from pathlib import Path
    from ezstitcher.core.stitcher import Stitcher
    from ezstitcher.core.config import StitcherConfig

    def test_generate_positions_with_mock():
        """Test generate_positions with mocked dependencies."""
        # Create a mock filename parser
        mock_parser = Mock()
        mock_parser.path_list_from_pattern.return_value = [
            "A01_s1_w1.tif",
            "A01_s2_w1.tif",
            "A01_s3_w1.tif",
            "A01_s4_w1.tif"
        ]

        # Create a stitcher with the mock parser
        stitcher = Stitcher(StitcherConfig(), filename_parser=mock_parser)

        # Mock the _generate_positions_ashlar method
        with patch.object(stitcher, '_generate_positions_ashlar', return_value=True) as mock_method:
            # Call the method being tested
            result = stitcher.generate_positions(
                image_dir="path/to/images",
                image_pattern="A01_s{iii}_w1.tif",
                positions_path="path/to/positions.csv",
                grid_size_x=2,
                grid_size_y=2
            )

            # Verify that the method was called with the expected arguments
            mock_method.assert_called_once_with(
                "path/to/images",
                "A01_s{iii}_w1.tif",
                "path/to/positions.csv",
                2,
                2
            )

            # Verify the result
            assert result is True

Debugging Tests
------------

To debug tests, you can use the `--pdb` option to drop into the debugger when a test fails:

.. code-block:: bash

    pytest --pdb

You can also use the `breakpoint()` function to set a breakpoint in your test:

.. code-block:: python

    def test_something():
        # Some test code
        breakpoint()  # Debugger will stop here
        # More test code
