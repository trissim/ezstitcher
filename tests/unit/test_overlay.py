"""
Tests for the overlay module.
"""

import pytest
import numpy as np
from pathlib import Path

from ezstitcher.io.overlay import OverlayMode, OverlayOperation, ArraySupplier

@pytest.mark.skip(reason="Await full plan block approval")
def test_overlay_operation_lazy_load():
    called = False
    def supplier():
        nonlocal called
        called = True
        return np.ones((5, 5), dtype=np.uint8)
    op = OverlayOperation(key="test", data_supplier=supplier, disk_path=Path("/tmp/test.tif"))
    
    # Verify the supplier hasn't been called yet
    assert called is False
    
    # Access the data through the supplier
    data = op.data_supplier()
    
    # Verify the supplier was called
    assert called is True
    assert data[0, 0] == 1