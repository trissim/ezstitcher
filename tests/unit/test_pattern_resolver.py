import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from ezstitcher.core.pattern_resolver import get_patterns_for_well, PatternDetector
from ezstitcher.core.microscope_interfaces import MicroscopeHandler

class TestPatternResolver:
    """Tests for pattern_resolver module."""

    def test_get_patterns_for_well_with_grouped_patterns(self):
        """Test flattening of grouped patterns."""
        mock_detector = MagicMock(spec=PatternDetector)
        mock_detector.auto_detect_patterns.return_value = {
            'A01': {
                'ch1': ['pattern1.tif', 'pattern2.tif'],
                'ch2': ['pattern3.tif', 'pattern4.tif']
            }
        }

        patterns = get_patterns_for_well('A01', '/path/to/dir', mock_detector)

        assert patterns == ['pattern1.tif', 'pattern2.tif', 'pattern3.tif', 'pattern4.tif']
        mock_detector.auto_detect_patterns.assert_called_once_with(
            Path('/path/to/dir'), well_filter=['A01'], variable_components=['site'], flat=False
        )

    def test_get_patterns_for_well_with_flat_patterns(self):
        """Test handling of flat pattern lists."""
        mock_detector = MagicMock(spec=PatternDetector)
        mock_detector.auto_detect_patterns.return_value = {
            'A01': ['pattern1.tif', 'pattern2.tif']
        }

        patterns = get_patterns_for_well('A01', '/path/to/dir', mock_detector)

        assert patterns == ['pattern1.tif', 'pattern2.tif']

    def test_get_patterns_for_well_with_custom_variable_components(self):
        """Test custom variable components."""
        mock_detector = MagicMock(spec=PatternDetector)
        mock_detector.auto_detect_patterns.return_value = {'A01': {'z1': ['pattern1.tif']}}

        get_patterns_for_well('A01', '/path/to/dir', mock_detector, variable_components=['z_index'])

        mock_detector.auto_detect_patterns.assert_called_once_with(
            Path('/path/to/dir'), well_filter=['A01'], variable_components=['z_index'], flat=False
        )

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        mock_detector = MagicMock(spec=PatternDetector)

        # Empty patterns
        mock_detector.auto_detect_patterns.return_value = {}
        assert get_patterns_for_well('A01', '/path/to/dir', mock_detector) == []

        # Well not found
        mock_detector.auto_detect_patterns.return_value = {'B02': {'ch1': ['pattern1.tif']}}
        assert get_patterns_for_well('A01', '/path/to/dir', mock_detector) == []

        # Exception handling
        mock_detector.auto_detect_patterns.side_effect = ValueError("Test error")
        assert get_patterns_for_well('A01', '/path/to/dir', mock_detector) == []

    def test_path_handling_and_microscope_handler_compatibility(self):
        """Test Path objects and MicroscopeHandler compatibility."""
        # Path objects
        mock_detector = MagicMock(spec=PatternDetector)
        mock_detector.auto_detect_patterns.return_value = {'A01': ['pattern1.tif']}
        directory = Path('/path/to/dir')

        patterns = get_patterns_for_well('A01', directory, mock_detector)

        assert patterns == ['pattern1.tif']
        mock_detector.auto_detect_patterns.assert_called_with(
            directory, well_filter=['A01'], variable_components=['site'], flat=False
        )

        # MicroscopeHandler compatibility
        with patch('ezstitcher.core.microscope_interfaces.MicroscopeHandler.auto_detect_patterns') as mock_auto_detect:
            mock_auto_detect.return_value = {'A01': {'ch1': ['pattern1.tif']}}

            handler = MagicMock(spec=MicroscopeHandler)
            handler.auto_detect_patterns = mock_auto_detect

            patterns = get_patterns_for_well('A01', '/path/to/dir', handler)

            assert patterns == ['pattern1.tif']
            mock_auto_detect.assert_called_once()