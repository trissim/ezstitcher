"""
Test for circular imports in the virtual path modules.
"""

import unittest


class TestCircularImports(unittest.TestCase):
    """Test for circular imports in the virtual path modules."""

    def test_import_virtual_path(self):
        """Test importing the virtual_path module."""
        try:
            import ezstitcher.io.virtual_path
            # If we get here, the import succeeded
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import virtual_path: {e}")

    def test_import_virtual_path_factory(self):
        """Test importing the virtual_path_factory module."""
        try:
            import ezstitcher.io.virtual_path_factory
            # If we get here, the import succeeded
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import virtual_path_factory: {e}")

    def test_import_order(self):
        """Test importing the modules in different orders."""
        # First import virtual_path, then virtual_path_factory
        try:
            import ezstitcher.io.virtual_path
            import ezstitcher.io.virtual_path_factory
            # If we get here, the imports succeeded
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import in order virtual_path, virtual_path_factory: {e}")

        # Reset the modules
        import sys
        if "ezstitcher.io.virtual_path" in sys.modules:
            del sys.modules["ezstitcher.io.virtual_path"]
        if "ezstitcher.io.virtual_path_factory" in sys.modules:
            del sys.modules["ezstitcher.io.virtual_path_factory"]

        # First import virtual_path_factory, then virtual_path
        try:
            import ezstitcher.io.virtual_path_factory
            import ezstitcher.io.virtual_path
            # If we get here, the imports succeeded
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import in order virtual_path_factory, virtual_path: {e}")


if __name__ == "__main__":
    unittest.main()
