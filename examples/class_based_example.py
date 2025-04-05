#!/usr/bin/env python3
"""
Example script demonstrating the class-based API of ezstitcher.
"""

import logging
from pathlib import Path

# Import classes from ezstitcher
from ezstitcher.core import (
    ImageProcessor,
    FocusDetector,
    ZStackManager,
    StitcherManager
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function demonstrating the class-based API."""
    # Path to the plate folder
    plate_folder = "/path/to/your/plate/folder"
    
    # 1. Preprocess the plate folder to detect and organize Z-stacks
    has_zstack, z_info = ZStackManager.preprocess_plate_folder(plate_folder)
    
    if has_zstack:
        print(f"Z-stack detected in {plate_folder}")
        
        # 2. Create projections from Z-stacks
        projections_dir = Path(plate_folder).parent / f"{Path(plate_folder).name}_projections" / "TimePoint_1"
        projections_dir.mkdir(parents=True, exist_ok=True)
        
        projections = ZStackManager.create_zstack_projections(
            Path(plate_folder) / "TimePoint_1",
            projections_dir,
            projection_types=["max", "mean"]
        )
        
        # 3. Find best focused images
        best_focus_dir = Path(plate_folder).parent / f"{Path(plate_folder).name}_best_focus" / "TimePoint_1"
        best_focus_dir.mkdir(parents=True, exist_ok=True)
        
        best_focus_results = ZStackManager.select_best_focus_zstack(
            Path(plate_folder) / "TimePoint_1",
            best_focus_dir,
            focus_method="combined",
            focus_wavelength="1"
        )
        
        # 4. Process the plate folder using best focused images
        StitcherManager.process_plate_folder(
            plate_folder,
            reference_channels=["1", "2"],
            composite_weights={"1": 0.3, "2": 0.7},
            preprocessing_funcs={"1": ImageProcessor.process_bf},
            tile_overlap=10,
            max_shift=50,
            focus_detect=True,
            focus_method="combined",
            create_projections=True,
            projection_types=["max", "mean"],
            stitch_z_reference="best_focus"
        )
    else:
        print(f"No Z-stack detected in {plate_folder}")
        
        # Process without Z-stack handling
        StitcherManager.process_plate_folder(
            plate_folder,
            reference_channels=["1"],
            preprocessing_funcs={"1": ImageProcessor.process_bf},
            tile_overlap=10,
            max_shift=50
        )

if __name__ == "__main__":
    main()
