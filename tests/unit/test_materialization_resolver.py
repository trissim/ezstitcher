"""
Unit tests for the MaterializationResolver.
"""

import unittest
from unittest.mock import Mock

from ezstitcher.io.materialization_resolver import MaterializationResolver
from ezstitcher.io.overlay import OverlayMode


class TestMaterializationResolver(unittest.TestCase):
    """Test the MaterializationResolver class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock step
        self.step = Mock()
        self.step.requires_fs_input = False
        self.step.requires_fs_output = False
        self.step.force_disk_output = False
        self.step.requires_legacy_fs = False
        self.step.needs_materialization.return_value = False

        # Create a mock orchestrator
        self.orchestrator = Mock()
        self.orchestrator.storage_mode = "memory"
        self.orchestrator.overlay_mode = OverlayMode.AUTO
        self.orchestrator.materialization_manager = Mock()
        self.orchestrator.materialization_manager.needs_materialization.return_value = False

    def test_no_orchestrator(self):
        """Test that materialization is not needed when no orchestrator is provided."""
        result = MaterializationResolver.needs_materialization(self.step)
        self.assertFalse(result)

    def test_materialization_manager(self):
        """Test that the materialization manager is used when available."""
        self.orchestrator.materialization_manager.needs_materialization.return_value = True
        result = MaterializationResolver.needs_materialization(self.step, self.orchestrator)
        self.assertTrue(result)
        self.orchestrator.materialization_manager.needs_materialization.assert_called_once_with(self.step)

    def test_direct_step_check(self):
        """Test that step flags are checked directly when materialization manager is not available."""
        # Remove materialization manager
        delattr(self.orchestrator, 'materialization_manager')
        # Set a flag on the step
        self.step.requires_fs_input = True
        result = MaterializationResolver.needs_materialization(self.step, self.orchestrator)
        self.assertTrue(result)

    def test_legacy_storage_mode(self):
        """Test that materialization is not needed when storage mode is legacy."""
        # Remove materialization manager
        delattr(self.orchestrator, 'materialization_manager')
        self.orchestrator.storage_mode = "legacy"
        # Set a flag on the step
        self.step.requires_fs_input = True
        result = MaterializationResolver.needs_materialization(self.step, self.orchestrator)
        self.assertFalse(result)

    def test_overlay_disabled(self):
        """Test that materialization is not needed when overlay mode is disabled."""
        # Remove materialization manager
        delattr(self.orchestrator, 'materialization_manager')
        self.orchestrator.overlay_mode = OverlayMode.DISABLED
        # Set a flag on the step
        self.step.requires_fs_input = True
        result = MaterializationResolver.needs_materialization(self.step, self.orchestrator)
        self.assertFalse(result)

    def test_step_flags_checked_directly(self):
        """Test that step flags are checked directly when materialization manager is not available."""
        # Remove materialization manager
        delattr(self.orchestrator, 'materialization_manager')
        # Set flags on the step
        self.step.requires_fs_input = False
        self.step.requires_fs_output = False
        self.step.force_disk_output = True
        result = MaterializationResolver.needs_materialization(self.step, self.orchestrator)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
