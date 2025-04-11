"""
Test natural sorting of filenames.
"""

import pytest
from pathlib import Path
import re

from ezstitcher.core.pattern_matcher import PatternMatcher
from ezstitcher.core.filename_parser import ImageXpressFilenameParser


def test_pattern_matcher_natural_sort():
    """Test that PatternMatcher._natural_sort correctly sorts filenames with mixed padding."""
    # Create a list of filenames with mixed padding
    filenames = [
        "A01_s10_w1.tif",
        "A01_s1_w1.tif",
        "A01_s2_w1.tif",
        "A01_s11_w1.tif",
        "A01_s001_w1.tif",  # Padded version of s1
    ]

    # Create a PatternMatcher instance
    pattern_matcher = PatternMatcher()

    # Sort the filenames using natural sorting
    sorted_filenames = pattern_matcher._natural_sort(filenames)

    # Expected order: s1/s001, s2, s10, s11
    expected_order = [
        "A01_s1_w1.tif",  # or "A01_s001_w1.tif" (either is fine as they're the same site)
        "A01_s2_w1.tif",
        "A01_s10_w1.tif",
        "A01_s11_w1.tif",
    ]

    # Check that the sorted filenames match the expected order
    # We need to handle the fact that s1 and s001 are the same site
    assert len(sorted_filenames) == len(filenames)

    # Extract site numbers for comparison
    def extract_site(filename):
        match = re.search(r'_s(\d+)_', filename)
        return int(match.group(1)) if match else None

    # Get unique site numbers in the correct order
    sorted_sites = []
    seen_sites = set()
    for f in sorted_filenames:
        site = extract_site(f)
        if site not in seen_sites:
            sorted_sites.append(site)
            seen_sites.add(site)

    expected_sites = [extract_site(f) for f in expected_order]

    assert sorted_sites == expected_sites


def test_filename_parser_pad_site_number():
    """Test that FilenameParser.pad_site_number correctly pads site numbers."""
    # Create a FilenameParser instance
    parser = ImageXpressFilenameParser()

    # Test padding for ImageXpress format
    assert parser.pad_site_number("A01_s1_w1.tif") == "A01_s001_w1.tif"
    assert parser.pad_site_number("A01_s10_w1.tif") == "A01_s010_w1.tif"
    assert parser.pad_site_number("A01_s100_w1.tif") == "A01_s100_w1.tif"  # Already padded

    # Test with path
    assert parser.pad_site_number("/path/to/A01_s1_w1.tif") == "/path/to/A01_s001_w1.tif"

    # Test with Z-index
    assert parser.pad_site_number("A01_s1_w1_z1.tif") == "A01_s001_w1_z1.tif"

    # Test with custom width
    assert parser.pad_site_number("A01_s1_w1.tif", width=4) == "A01_s0001_w1.tif"

    # Test with invalid filename (should return original)
    assert parser.pad_site_number("invalid_filename.tif") == "invalid_filename.tif"


def test_integration_natural_sorting_and_padding():
    """Test the integration of natural sorting and site number padding."""
    # Create a list of filenames with mixed padding
    filenames = [
        "A01_s10_w1.tif",
        "A01_s1_w1.tif",
        "A01_s2_w1.tif",
        "A01_s11_w1.tif",
    ]

    # Create instances
    pattern_matcher = PatternMatcher()
    parser = ImageXpressFilenameParser()

    # Sort the filenames using natural sorting
    sorted_filenames = pattern_matcher._natural_sort(filenames)

    # Pad the site numbers in the sorted filenames
    padded_filenames = [parser.pad_site_number(f) for f in sorted_filenames]

    # Expected padded filenames
    expected_padded = [
        "A01_s001_w1.tif",
        "A01_s002_w1.tif",
        "A01_s010_w1.tif",
        "A01_s011_w1.tif",
    ]

    assert padded_filenames == expected_padded
