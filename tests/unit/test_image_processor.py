"""
Unit tests for the ImageProcessor class.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ezstitcher.core.image_processor import ImageProcessor as IP
from ezstitcher.core.image_processor import create_linear_weight_mask
from ezstitcher.core.focus_analyzer import FocusAnalyzer


class TestImageProcessor:
    """Tests for the ImageProcessor class."""

    @pytest.fixture
    def test_image_2d(self):
        """Create a 2D test image with a bright square in the center."""
        image = np.ones((100, 100), dtype=np.uint16) * 1000
        image[40:60, 40:60] = 5000  # Add a bright square
        return image

    @pytest.fixture
    def test_image_3d(self):
        """Create a 3D test image with a bright square in the center."""
        image = np.ones((3, 100, 100), dtype=np.uint16) * 1000
        image[:, 40:60, 40:60] = 5000  # Add a bright square
        return image

    @pytest.fixture
    def test_stack(self):
        """Create a stack of test images."""
        stack = []
        for i in range(3):
            img = np.ones((100, 100), dtype=np.uint16) * 1000
            img[40:60, 40:60] = 4000 + i * 1000  # Increasing brightness
            stack.append(img)
        return stack

    def test_sharpen(self, test_image_2d):
        """Test the sharpen method."""
        # Apply sharpening
        sharpened = IP.sharpen(test_image_2d, radius=1.0, amount=1.5)

        # Verify that the image was sharpened
        assert sharpened.shape == test_image_2d.shape
        assert sharpened.dtype == test_image_2d.dtype

        # The bright square should have higher contrast after sharpening
        bright_square_original = np.mean(test_image_2d[40:60, 40:60])
        bright_square_sharpened = np.mean(sharpened[40:60, 40:60])

        # The edge of the square should be enhanced
        edge_original = np.mean(test_image_2d[39:41, 40:60])
        edge_sharpened = np.mean(sharpened[39:41, 40:60])

        assert bright_square_sharpened >= bright_square_original
        assert edge_sharpened != edge_original  # Edge should change

    def test_percentile_normalize(self, test_image_2d):
        """Test the percentile_normalize method."""
        # Apply percentile normalization
        normalized = IP.percentile_normalize(
            test_image_2d,
            low_percentile=1,
            high_percentile=99,
            target_min=0,
            target_max=65535
        )

        # Verify that the image was normalized
        assert normalized.shape == test_image_2d.shape
        assert normalized.dtype == test_image_2d.dtype

        # Check that values are within the target range
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 65535

        # The bright square should still be brighter than the background
        assert np.mean(normalized[40:60, 40:60]) > np.mean(normalized[0:40, 0:40])

    def test_stack_percentile_normalize(self, test_stack):
        """Test the stack_percentile_normalize method."""
        # Apply stack percentile normalization
        normalized = IP.stack_percentile_normalize(
            test_stack,
            low_percentile=1,
            high_percentile=99,
            target_min=0,
            target_max=65535
        )

        # Verify that the stack was normalized
        assert len(normalized) == len(test_stack)
        assert normalized[0].shape == test_stack[0].shape
        assert normalized[0].dtype == test_stack[0].dtype

        # Check that values are within the target range
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 65535

        # The relative brightness between images should be preserved
        assert np.mean(normalized[2]) > np.mean(normalized[1]) > np.mean(normalized[0])

    def test_create_composite_equal_weights(self, test_stack):
        """Test the create_composite method with equal weights."""
        # Create composite with equal weights
        composite = IP.create_composite(test_stack)

        # Verify the composite
        assert composite.shape == test_stack[0].shape
        assert composite.dtype == test_stack[0].dtype

        # The composite should be between the min and max of the stack
        assert np.mean(composite) > np.mean(test_stack[0])
        assert np.mean(composite) < np.mean(test_stack[2])





    def test_create_composite_empty_list(self):
        """Test the create_composite method with an empty list."""
        # Try to create a composite from an empty list
        with pytest.raises(ValueError):
            IP.create_composite([])

    def test_create_composite_invalid_input(self):
        """Test the create_composite method with invalid input."""
        # Try to create a composite from a non-list
        with pytest.raises(TypeError):
            IP.create_composite(np.ones((10, 10)))

    def test_apply_mask(self, test_image_2d):
        """Test the apply_mask method."""
        # Create a mask
        mask = np.zeros_like(test_image_2d, dtype=np.float32)
        mask[30:70, 30:70] = 1.0  # Mask the center region

        # Apply the mask
        masked = IP.apply_mask(test_image_2d, mask)

        # Verify the masked image
        assert masked.shape == test_image_2d.shape
        assert masked.dtype == test_image_2d.dtype

        # The masked region should have values, the rest should be zero
        assert np.all(masked[0:30, 0:30] == 0)
        assert np.all(masked[30:70, 30:70] > 0)

    def test_apply_mask_shape_mismatch(self, test_image_2d):
        """Test the apply_mask method with a shape mismatch."""
        # Create a mask with a different shape
        mask = np.ones((50, 50), dtype=np.float32)

        # Try to apply the mask
        with pytest.raises(ValueError):
            IP.apply_mask(test_image_2d, mask)

    def test_create_weight_mask(self):
        """Test the create_weight_mask method."""
        # Create a weight mask
        height, width = 100, 100
        mask = IP.create_weight_mask((height, width), margin_ratio=0.1)

        # Verify the mask
        assert mask.shape == (height, width)
        assert mask.dtype == np.float32

        # The center should be close to 1.0, the edges should be less than 1.0
        assert mask[50, 50] > 0.9  # Center should be close to 1.0
        assert mask[0, 0] < 0.5    # Corners should be less than 0.5
        assert mask[0, 50] < 0.5   # Edges should be less than 0.5
        assert mask[50, 0] < 0.5
        assert mask[99, 99] < 0.5

    def test_max_projection(self, test_stack):
        """Test the max_projection method."""
        # Create a max projection
        projection = IP.max_projection(test_stack)

        # Verify the projection
        assert projection.shape == test_stack[0].shape
        assert projection.dtype == test_stack[0].dtype

        # The projection should have the maximum value at each pixel
        for i in range(len(test_stack)):
            assert np.all(projection >= test_stack[i])

        # The bright square should have the value from the brightest image
        assert np.mean(projection[40:60, 40:60]) == np.mean(test_stack[2][40:60, 40:60])

    def test_mean_projection(self, test_stack):
        """Test the mean_projection method."""
        # Create a mean projection
        projection = IP.mean_projection(test_stack)

        # Verify the projection
        assert projection.shape == test_stack[0].shape
        assert projection.dtype == test_stack[0].dtype

        # The projection should have the mean value at each pixel
        expected_mean = np.mean(np.array(test_stack), axis=0)
        assert np.allclose(projection, expected_mean)

    def test_stack_equalize_histogram(self, test_stack):
        """Test the stack_equalize_histogram method."""
        # Apply histogram equalization
        equalized = IP.stack_equalize_histogram(test_stack)

        # Verify the equalized stack
        assert len(equalized) == len(test_stack)
        assert equalized[0].shape == test_stack[0].shape
        assert equalized[0].dtype == test_stack[0].dtype

        # The histogram equalization should change the image values
        # Check that the equalized stack is different from the original
        stack_array = np.array(test_stack)
        assert not np.array_equal(stack_array, equalized)

        # Check that the output is in the expected range for uint16
        assert np.min(equalized) >= 0
        assert np.max(equalized) <= 65535

    def test_create_projection_max(self, test_stack):
        """Test the create_projection method with max_projection."""
        # Create a max projection
        projection = IP.create_projection(test_stack, method="max_projection")

        # Verify the projection
        assert projection.shape == test_stack[0].shape
        assert projection.dtype == test_stack[0].dtype

        # Should be the same as max_projection
        max_proj = IP.max_projection(test_stack)
        assert np.array_equal(projection, max_proj)

    def test_create_projection_mean(self, test_stack):
        """Test the create_projection method with mean_projection."""
        # Create a mean projection
        projection = IP.create_projection(test_stack, method="mean_projection")

        # Verify the projection
        assert projection.shape == test_stack[0].shape
        assert projection.dtype == test_stack[0].dtype

        # Should be the same as mean_projection
        mean_proj = IP.mean_projection(test_stack)
        assert np.array_equal(projection, mean_proj)

    def test_create_projection_best_focus(self, test_stack):
        """Test the create_projection method with best_focus."""
        # Create a mock focus analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.find_best_focus.return_value = (2, 0.8)  # Return index 2 as best focus

        # Create a best focus projection
        projection = IP.create_projection(test_stack, method="best_focus", focus_analyzer=mock_analyzer)

        # Verify the projection
        assert projection.shape == test_stack[0].shape
        assert projection.dtype == test_stack[0].dtype

        # Should be the same as the best focus image (index 2)
        assert np.array_equal(projection, test_stack[2])

        # Verify that find_best_focus was called
        mock_analyzer.find_best_focus.assert_called_once_with(test_stack)

    def test_create_projection_best_focus_no_analyzer(self, test_stack):
        """Test the create_projection method with best_focus but no analyzer."""
        # Create a best focus projection without an analyzer
        projection = IP.create_projection(test_stack, method="best_focus")

        # Verify the projection
        assert projection.shape == test_stack[0].shape
        assert projection.dtype == test_stack[0].dtype

        # Should fall back to max_projection
        max_proj = IP.max_projection(test_stack)
        assert np.array_equal(projection, max_proj)

    def test_create_projection_unknown_method(self, test_stack):
        """Test the create_projection method with an unknown method."""
        # Create a projection with an unknown method
        projection = IP.create_projection(test_stack, method="unknown_method")

        # Verify the projection
        assert projection.shape == test_stack[0].shape
        assert projection.dtype == test_stack[0].dtype

        # Should fall back to max_projection
        max_proj = IP.max_projection(test_stack)
        assert np.array_equal(projection, max_proj)

    def test_tophat(self, test_image_2d):
        """Test the tophat method."""
        # Apply tophat filter
        filtered = IP.tophat(test_image_2d, selem_radius=10, downsample_factor=2)

        # Verify the filtered image
        assert filtered.shape == test_image_2d.shape
        assert filtered.dtype == test_image_2d.dtype

        # The background should be reduced
        assert np.mean(filtered[0:40, 0:40]) < np.mean(test_image_2d[0:40, 0:40])

        # The bright features should be preserved
        assert np.mean(filtered[40:60, 40:60]) > np.mean(filtered[0:40, 0:40])


class TestCreateLinearWeightMask:
    """Tests for the create_linear_weight_mask function."""

    def test_create_linear_weight_mask(self):
        """Test the create_linear_weight_mask function."""
        # Create a weight mask
        height, width = 100, 100
        margin_ratio = 0.1
        mask = create_linear_weight_mask(height, width, margin_ratio)

        # Verify the mask
        assert mask.shape == (height, width)
        assert mask.dtype == np.float32

        # The center should be close to 1.0, the edges should be less than 1.0
        assert mask[50, 50] > 0.9  # Center should be close to 1.0
        assert mask[0, 0] < 0.5    # Corners should be less than 0.5
        assert mask[0, 50] < 0.5   # Edges should be less than 0.5
        assert mask[50, 0] < 0.5
        assert mask[99, 99] < 0.5

    def test_create_linear_weight_mask_zero_margin(self):
        """Test the create_linear_weight_mask function with zero margin."""
        # Create a weight mask with zero margin
        height, width = 100, 100
        margin_ratio = 0.0
        mask = create_linear_weight_mask(height, width, margin_ratio)

        # Verify the mask
        assert mask.shape == (height, width)
        assert mask.dtype == np.float32

        # All values should be close to 1.0
        assert np.all(mask > 0.99)

    def test_create_linear_weight_mask_full_margin(self):
        """Test the create_linear_weight_mask function with full margin."""
        # Create a weight mask with full margin
        height, width = 100, 100
        margin_ratio = 0.5
        mask = create_linear_weight_mask(height, width, margin_ratio)

        # Verify the mask
        assert mask.shape == (height, width)
        assert mask.dtype == np.float32

        # The center should be close to 1.0, the corners should be close to 0.0
        assert mask[50, 50] > 0.9  # Center should be close to 1.0
        assert mask[0, 0] < 0.1    # Corners should be close to 0.0
        assert mask[0, 99] < 0.1
        assert mask[99, 0] < 0.1
        assert mask[99, 99] < 0.1
