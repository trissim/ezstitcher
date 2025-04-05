"""
Test imports from the new class-based API.
"""

import unittest

class TestImportsNew(unittest.TestCase):
    """Test imports from the new class-based API."""
    
    def test_import_main_functions(self):
        """Test that main functions can be imported successfully."""
        try:
            from ezstitcher.core.main import (
                process_plate_folder,
                modified_process_plate_folder,
                process_bf,
                find_best_focus
            )
            # If we get here, the import was successful
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import error: {e}")
    
    def test_import_classes(self):
        """Test that classes can be imported successfully."""
        try:
            from ezstitcher.core import (
                ImageProcessor,
                FocusDetector,
                ZStackManager,
                StitcherManager
            )
            # If we get here, the import was successful
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import error: {e}")
    
    def test_import_from_package(self):
        """Test that imports from the package work."""
        try:
            from ezstitcher import (
                process_plate_folder,
                modified_process_plate_folder,
                process_bf,
                find_best_focus,
                ImageProcessor,
                FocusDetector,
                ZStackManager,
                StitcherManager
            )
            # If we get here, the import was successful
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import error: {e}")


if __name__ == "__main__":
    unittest.main()
