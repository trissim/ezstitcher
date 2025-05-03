"""Tests for the ProcessingContext.update_from_step_result method."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from ezstitcher.core.pipeline import ProcessingContext, StepResult


class TestProcessingContextUpdate:
    """Tests for the ProcessingContext.update_from_step_result method."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a context
        self.context = ProcessingContext()
        
        # Create a step result
        self.step_result = StepResult()
        
    def test_update_from_step_result_with_results(self):
        """Test update_from_step_result with results."""
        # Set up
        self.step_result.add_result("key1", "value1")
        self.step_result.add_result("key2", 42)
        self.step_result.add_result("key3", np.array([1, 2, 3]))
        
        # Test
        self.context.update_from_step_result(self.step_result)
        
        # Verify
        assert self.context.results["key1"] == "value1"
        assert self.context.results["key2"] == 42
        assert np.array_equal(self.context.results["key3"], np.array([1, 2, 3]))
        
    def test_update_from_step_result_with_context_updates(self):
        """Test update_from_step_result with context updates."""
        # Set up
        self.step_result.update_context("attr1", "value1")
        self.step_result.update_context("attr2", 42)
        self.step_result.update_context("attr3", np.array([1, 2, 3]))
        
        # Test
        self.context.update_from_step_result(self.step_result)
        
        # Verify
        assert self.context.attr1 == "value1"
        assert self.context.attr2 == 42
        assert np.array_equal(self.context.attr3, np.array([1, 2, 3]))
        
    def test_update_from_step_result_with_both(self):
        """Test update_from_step_result with both results and context updates."""
        # Set up
        self.step_result.add_result("key1", "value1")
        self.step_result.update_context("attr1", "value1")
        
        # Test
        self.context.update_from_step_result(self.step_result)
        
        # Verify
        assert self.context.results["key1"] == "value1"
        assert self.context.attr1 == "value1"
        
    def test_update_from_step_result_with_empty_result(self):
        """Test update_from_step_result with empty result."""
        # Test
        self.context.update_from_step_result(self.step_result)
        
        # Verify
        assert self.context.results == {}
        
    def test_update_from_step_result_with_existing_results(self):
        """Test update_from_step_result with existing results."""
        # Set up
        self.context.results = {"key1": "old_value", "key3": "value3"}
        self.step_result.add_result("key1", "new_value")
        self.step_result.add_result("key2", "value2")
        
        # Test
        self.context.update_from_step_result(self.step_result)
        
        # Verify
        assert self.context.results["key1"] == "new_value"
        assert self.context.results["key2"] == "value2"
        assert self.context.results["key3"] == "value3"
        
    def test_update_from_step_result_with_existing_attributes(self):
        """Test update_from_step_result with existing attributes."""
        # Set up
        self.context.attr1 = "old_value"
        self.context.attr3 = "value3"
        self.step_result.update_context("attr1", "new_value")
        self.step_result.update_context("attr2", "value2")
        
        # Test
        self.context.update_from_step_result(self.step_result)
        
        # Verify
        assert self.context.attr1 == "new_value"
        assert self.context.attr2 == "value2"
        assert self.context.attr3 == "value3"
        
    def test_update_from_step_result_with_storage_operations(self):
        """Test update_from_step_result with storage operations."""
        # Set up
        self.step_result.store("key1", np.array([1, 2, 3]))
        self.step_result.store("key2", np.array([4, 5, 6]))
        
        # Test
        self.context.update_from_step_result(self.step_result)
        
        # Verify - storage operations are not directly reflected in the context
        assert "key1" not in self.context.results
        assert "key2" not in self.context.results
        
    def test_update_from_step_result_with_complex_objects(self):
        """Test update_from_step_result with complex objects."""
        # Set up
        class ComplexObject:
            def __init__(self, value):
                self.value = value
                
        obj1 = ComplexObject("value1")
        obj2 = ComplexObject("value2")
        
        self.step_result.add_result("obj1", obj1)
        self.step_result.update_context("obj2", obj2)
        
        # Test
        self.context.update_from_step_result(self.step_result)
        
        # Verify
        assert self.context.results["obj1"] is obj1
        assert self.context.results["obj1"].value == "value1"
        assert self.context.obj2 is obj2
        assert self.context.obj2.value == "value2"
