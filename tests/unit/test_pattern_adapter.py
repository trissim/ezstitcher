"""
Tests for pattern_adapter module.
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ezstitcher.io.pattern_adapter import (
    PatternFormatRegistry,
    NoOpAdapter,
    AshlarPatternAdapter
)


class TestPatternFormatAdapter:
    """Tests for pattern format adapters."""

    def test_noop_adapter(self):
        """Test NoOpAdapter."""
        adapter = NoOpAdapter()
        pattern = "test_{iii}_pattern.tif"
        
        # Should return pattern unchanged
        assert adapter.to_internal(pattern) == pattern
        assert adapter.from_internal(pattern) == pattern
        
    def test_ashlar_adapter(self):
        """Test AshlarPatternAdapter."""
        adapter = AshlarPatternAdapter()
        
        # Test to_internal
        ashlar_pattern = "test_{series}_pattern.tif"
        internal_pattern = "test_{iii}_pattern.tif"
        assert adapter.to_internal(ashlar_pattern) == internal_pattern
        
        # Test from_internal
        assert adapter.from_internal(internal_pattern) == ashlar_pattern
        
        # Test with multiple placeholders
        complex_pattern = "test_{series}_{series}_pattern.tif"
        expected = "test_{iii}_{iii}_pattern.tif"
        assert adapter.to_internal(complex_pattern) == expected
        assert adapter.from_internal(expected) == complex_pattern
        
    def test_registry_singleton(self):
        """Test that PatternFormatRegistry is a singleton."""
        registry1 = PatternFormatRegistry()
        registry2 = PatternFormatRegistry()
        assert registry1 is registry2
        
    def test_registry_default_adapters(self):
        """Test that registry initializes with default adapters."""
        registry = PatternFormatRegistry()
        
        # Should have internal and ashlar adapters by default
        assert isinstance(registry.get_adapter("internal"), NoOpAdapter)
        assert isinstance(registry.get_adapter("ashlar"), AshlarPatternAdapter)
        
        # Default adapter should be internal
        assert isinstance(registry.get_adapter(), NoOpAdapter)
        
    def test_registry_case_insensitivity(self):
        """Test that registry is case-insensitive."""
        registry = PatternFormatRegistry()
        
        # Should work with any case
        assert isinstance(registry.get_adapter("ASHLAR"), AshlarPatternAdapter)
        assert isinstance(registry.get_adapter("Ashlar"), AshlarPatternAdapter)
        assert isinstance(registry.get_adapter("ashlar"), AshlarPatternAdapter)
        
    def test_registry_custom_adapter(self):
        """Test registering a custom adapter."""
        registry = PatternFormatRegistry()
        
        # Create a custom adapter
        class CustomAdapter:
            def to_internal(self, pattern):
                return pattern.replace("{custom}", "{iii}")
                
            def from_internal(self, pattern):
                return pattern.replace("{iii}", "{custom}")
        
        # Register the custom adapter
        registry.register_adapter("custom", CustomAdapter())
        
        # Should be able to retrieve it
        adapter = registry.get_adapter("custom")
        assert adapter.to_internal("test_{custom}_pattern.tif") == "test_{iii}_pattern.tif"
        assert adapter.from_internal("test_{iii}_pattern.tif") == "test_{custom}_pattern.tif"
        
    def test_registry_invalid_adapter(self):
        """Test registering an invalid adapter."""
        registry = PatternFormatRegistry()
        
        # Create an invalid adapter (missing methods)
        class InvalidAdapter:
            pass
        
        # Should raise TypeError
        with pytest.raises(TypeError):
            registry.register_adapter("invalid", InvalidAdapter())
            
    def test_registry_unknown_adapter(self):
        """Test retrieving an unknown adapter."""
        registry = PatternFormatRegistry()
        
        # Should raise KeyError
        with pytest.raises(KeyError):
            registry.get_adapter("unknown")


class TestPatternFormatRegistry:
    def test_singleton(self):
        """Test that PatternFormatRegistry is a singleton."""
        registry1 = PatternFormatRegistry()
        registry2 = PatternFormatRegistry()
        assert registry1 is registry2
    
    def test_default_adapters(self):
        """Test that default adapters are registered."""
        registry = PatternFormatRegistry()
        assert isinstance(registry.get_adapter("internal"), NoOpAdapter)
        assert isinstance(registry.get_adapter("ashlar"), AshlarPatternAdapter)
    
    def test_register_adapter(self):
        """Test registering a new adapter."""
        registry = PatternFormatRegistry()
        
        # Create a custom adapter
        class CustomAdapter:
            def to_internal(self, pattern: str) -> str:
                return pattern + "_internal"
            
            def from_internal(self, pattern: str) -> str:
                return pattern + "_external"
        
        # Register the adapter
        custom_adapter = CustomAdapter()
        registry.register_adapter("custom", custom_adapter)
        
        # Get the adapter and test it
        retrieved_adapter = registry.get_adapter("custom")
        assert retrieved_adapter is custom_adapter
        assert retrieved_adapter.to_internal("test") == "test_internal"
        assert retrieved_adapter.from_internal("test") == "test_external"
    
    def test_freeze_registry(self):
        """Test freezing the registry."""
        registry = PatternFormatRegistry()
        
        # Create a custom adapter
        class CustomAdapter:
            def to_internal(self, pattern: str) -> str:
                return pattern
            
            def from_internal(self, pattern: str) -> str:
                return pattern
        
        # Freeze the registry
        registry.freeze()
        assert registry.is_frozen() is True
        
        # Try to register a new adapter
        with pytest.raises(RuntimeError):
            registry.register_adapter("custom", CustomAdapter())
    
    def test_thread_safety(self):
        """Test thread safety of the registry."""
        registry = PatternFormatRegistry()
        
        # Create a custom adapter
        class CustomAdapter:
            def to_internal(self, pattern: str) -> str:
                return pattern
            
            def from_internal(self, pattern: str) -> str:
                return pattern
        
        # Function to register adapters in parallel
        def register_adapter(index):
            try:
                registry.register_adapter(f"custom{index}", CustomAdapter())
                return True
            except Exception:
                return False
        
        # Register adapters in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(register_adapter, range(100)))
        
        # Check that all registrations succeeded
        assert all(results)
        
        # Check that all adapters are registered
        registered_adapters = registry.get_registered_adapters()
        assert len(registered_adapters) >= 102  # 100 custom + 2 default
        
        # Freeze the registry
        registry.freeze()
        
        # Try to register more adapters in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(register_adapter, range(100, 200)))
        
        # Check that all registrations failed
        assert not any(results)

    def test_registry_frozen_by_default(self):
        """Test that the registry is frozen by default."""
        registry = PatternFormatRegistry()
        assert registry.is_frozen() is True
        
        # Try to register a new adapter
        class CustomAdapter:
            def to_internal(self, pattern: str) -> str:
                return pattern
            
            def from_internal(self, pattern: str) -> str:
                return pattern
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError):
            registry.register_adapter("custom", CustomAdapter())
