"""
Pytest configuration file for EZStitcher tests.
"""
import os
import sys
import pytest
from pathlib import Path

# Add the parent directory to sys.path to allow importing from ezstitcher
sys.path.insert(0, str(Path(__file__).parent.parent))

# Define common fixtures that can be used across all tests

@pytest.fixture(scope="session")
def tests_root_dir():
    """Return the root directory of the tests."""
    return Path(__file__).parent

@pytest.fixture(scope="session")
def tests_data_dir():
    """Return the directory for test data."""
    data_dir = Path(__file__).parent / "tests_data"
    data_dir.mkdir(exist_ok=True, parents=True)
    return data_dir
