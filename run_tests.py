#!/usr/bin/env python3
"""
Test runner script for the ezstitcher package.

This script runs all the tests in the tests directory.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the package
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def run_tests():
    """Run all tests in the tests directory."""
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
