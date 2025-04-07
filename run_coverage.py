#!/usr/bin/env python3
"""
Run test coverage analysis on the synthetic workflow test.
"""

import sys
import os
import unittest

# Try to import coverage
try:
    import coverage
    has_coverage = True
except ImportError:
    has_coverage = False
    print("Coverage module not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage"])
    import coverage
    has_coverage = True

if has_coverage:
    # Start coverage
    cov = coverage.Coverage()
    cov.start()

    # Run the tests
    from tests.test_synthetic_workflow import TestSyntheticWorkflow
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSyntheticWorkflow)
    unittest.TextTestRunner().run(suite)

    # Stop coverage and generate report
    cov.stop()
    cov.save()
    
    # Print report to console
    print("\nCoverage Report:")
    cov.report()
    
    # Generate HTML report
    html_dir = 'htmlcov'
    cov.html_report(directory=html_dir)
    print(f"\nHTML report generated in {html_dir}")
else:
    print("Failed to import or install coverage module.")
