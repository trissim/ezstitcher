# ezstitcher/io/types.py
from typing import TypeAlias
import numpy as np
# Potentially add other heavy types used in I/O signatures if needed
# from some_other_heavy_library import HeavyObject

# Define a type alias for NumPy arrays representing images
ImageArray: TypeAlias = np.ndarray
# Example of another potential alias:
# MetadataDict: TypeAlias = Dict[str, Any]