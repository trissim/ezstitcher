"""
Pattern format adapter system for EZStitcher.

This module provides a clean abstraction layer for pattern format translation,
allowing EZStitcher to work with different tool-specific pattern formats while
maintaining a consistent internal format.
"""

import threading
from typing import Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class PatternFormatAdapter(Protocol):
    """Protocol for pattern format adapters."""
    
    def to_internal(self, pattern: str) -> str:
        """
        Convert a tool-specific pattern to EZStitcher's internal format.
        
        Args:
            pattern: Pattern string in tool-specific format
            
        Returns:
            Pattern string in EZStitcher's internal format
        """
        ...
        
    def from_internal(self, pattern: str) -> str:
        """
        Convert EZStitcher's internal pattern to a tool-specific format.
        
        Args:
            pattern: Pattern string in EZStitcher's internal format
            
        Returns:
            Pattern string in tool-specific format
        """
        ...


class NoOpAdapter:
    """
    Default adapter that performs no translation.
    Used for internal EZStitcher patterns.
    """
    
    def to_internal(self, pattern: str) -> str:
        """Return pattern unchanged."""
        return pattern
        
    def from_internal(self, pattern: str) -> str:
        """Return pattern unchanged."""
        return pattern


class AshlarPatternAdapter:
    """
    Adapter for Ashlar pattern format.
    Translates between Ashlar's {series} and EZStitcher's {iii}.
    """
    
    def to_internal(self, pattern: str) -> str:
        """Convert Ashlar pattern to internal format."""
        return pattern.replace("{series}", "{iii}")
        
    def from_internal(self, pattern: str) -> str:
        """Convert internal pattern to Ashlar format."""
        return pattern.replace("{iii}", "{series}")


class PatternFormatRegistry:
    """Registry for pattern format adapters."""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern to ensure a single registry instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PatternFormatRegistry, cls).__new__(cls)
                cls._instance._adapters = {}
                cls._instance._frozen = False
                cls._instance._initialize_defaults()
            return cls._instance
    
    def _initialize_defaults(self):
        """Initialize default adapters."""
        self.register_adapter("internal", NoOpAdapter())
        self.register_adapter("ashlar", AshlarPatternAdapter())
    
    def register_adapter(self, name: str, adapter: PatternFormatAdapter) -> None:
        """
        Register a pattern format adapter.
        
        Args:
            name: Name to register the adapter under
            adapter: The adapter instance
            
        Raises:
            TypeError: If the adapter doesn't implement PatternFormatAdapter protocol
            RuntimeError: If the registry is frozen
        """
        with self._lock:
            if self._frozen:
                raise RuntimeError("Pattern format registry is frozen and cannot be modified")
                
            if not isinstance(adapter, PatternFormatAdapter):
                raise TypeError(f"Adapter must implement PatternFormatAdapter protocol")
                
            self._adapters[name.lower()] = adapter
    
    def get_adapter(self, name: Optional[str] = None) -> PatternFormatAdapter:
        """
        Get a registered adapter by name.
        
        Args:
            name: Name of the adapter to retrieve
            
        Returns:
            The requested adapter, or the default adapter if name is None
            
        Raises:
            KeyError: If the requested adapter is not registered
        """
        with self._lock:
            if name is None:
                return self._adapters["internal"]
            
            name = name.lower()
            if name not in self._adapters:
                raise KeyError(f"No adapter registered for '{name}'")
            
            return self._adapters[name]
    
    def freeze(self) -> None:
        """
        Freeze the registry to prevent further modifications.
        
        This should be called after all adapters are registered during application startup.
        """
        with self._lock:
            self._frozen = True
    
    def is_frozen(self) -> bool:
        """
        Check if the registry is frozen.
        
        Returns:
            True if the registry is frozen, False otherwise
        """
        with self._lock:
            return self._frozen
    
    def get_registered_adapters(self) -> Dict[str, PatternFormatAdapter]:
        """
        Get a copy of all registered adapters.
        
        Returns:
            A dictionary of adapter names to adapter instances
        """
        with self._lock:
            # Return a copy to prevent modification of the internal dictionary
            return dict(self._adapters)


# Freeze the registry after startup to prevent further modifications.
PatternFormatRegistry().freeze()
