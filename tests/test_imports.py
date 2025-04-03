"""
Basic import tests to verify package structure is correct.
"""

import unittest


class TestImports(unittest.TestCase):
    def test_import_core_modules(self):
        """Test that core modules can be imported successfully."""
        try:
            from ezstitcher.core import image_process
            from ezstitcher.core import stitcher
            from ezstitcher.core import z_stack_handler
            from ezstitcher.core import focus_detect
            self.assertTrue(True)  # If we get here, imports worked
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_import_main_functions(self):
        """Test that main functions can be imported successfully."""
        try:
            from ezstitcher.core.stitcher import process_plate_folder
            from ezstitcher.core.z_stack_handler import modified_process_plate_folder
            from ezstitcher.core.image_process import process_bf
            from ezstitcher.core.focus_detect import find_best_focus
            self.assertTrue(True)  # If we get here, imports worked
        except ImportError as e:
            self.fail(f"Import error: {e}")


if __name__ == "__main__":
    unittest.main()