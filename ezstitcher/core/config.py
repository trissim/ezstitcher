from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class StitchingConfig:
    reference_channels: List[str] = field(default_factory=lambda: ["1"])
    tile_overlap: float = 10.0
    max_shift: int = 50
    focus_detect: bool = False
    focus_method: str = "combined"
    create_projections: bool = False
    stitch_z_reference: str = "best_focus"
    save_projections: bool = True
    stitch_all_z_planes: bool = False
    well_filter: Optional[List[str]] = None
    composite_weights: Optional[Dict] = None
    preprocessing_funcs: Optional[Dict] = None
    margin_ratio: float = 0.1

@dataclass
class ZStackConfig:
    focus_method: str = "combined"
    projection_types: List[str] = field(default_factory=lambda: ["max"])
    select_best_focus: bool = True

@dataclass
class FocusConfig:
    method: str = "combined"
    roi: Optional[List[int]] = None  # [x, y, width, height]

@dataclass
class PlateConfig:
    plate_folder: str = ""
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    zstack: ZStackConfig = field(default_factory=ZStackConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
