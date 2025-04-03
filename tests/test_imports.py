"""
Basic import tests to verify package structure is correct.
"""

import unittest


class TestImports(unittest.TestCase):
    def test_import_core_modules(self):
        """Test that core modules can be imported successfully."""
        try:
            from axon_quant.core import image_process
            from axon_quant.core import stitcher
            from axon_quant.core import z_stack_handler
            self.assertTrue(True)  # If we get here, imports worked
        except ImportError as e:
            self.fail(f"Import error: {e}")

    def test_import_main_functions(self):
        """Test that main functions can be imported successfully."""
        try:
            from axon_quant.core.stitcher import process_plate_folder
            from axon_quant.core.z_stack_handler import modified_process_plate_folder
            from axon_quant.core.image_process import process_bf
            self.assertTrue(True)  # If we get here, imports worked
        except ImportError as e:
            self.fail(f"Import error: {e}")


if __name__ == "__main__":
    unittest.main()