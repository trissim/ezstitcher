#!/usr/bin/env python3
"""
Command-line interface for the Axon Quant package.
"""

import argparse
import sys
from pathlib import Path

from axon_quant.core.stitcher import process_plate_folder
from axon_quant.core.z_stack_handler import modified_process_plate_folder
from axon_quant.core.image_process import process_bf


def main():
    parser = argparse.ArgumentParser(
        description="Axon Quant - Microscopy Image Stitching for Axon Quantification"
    )
    
    parser.add_argument(
        "plate_folder", 
        type=str,
        help="Path to the plate folder containing microscopy images"
    )
    
    parser.add_argument(
        "--reference-channels", 
        type=str, 
        nargs="+", 
        default=["1"],
        help="Channel(s) to use as reference for alignment (default: 1)"
    )
    
    parser.add_argument(
        "--margin-ratio", 
        type=float, 
        default=0.1,
        help="Blending margin ratio for stitching (default: 0.1)"
    )
    
    parser.add_argument(
        "--tile-overlap", 
        type=float, 
        default=10.0,
        help="Percentage of overlap between tiles (default: 10.0)"
    )
    
    parser.add_argument(
        "--max-shift", 
        type=int, 
        default=50,
        help="Maximum shift allowed between tiles in microns (default: 50)"
    )
    
    parser.add_argument(
        "--bf-process", 
        action="store_true",
        help="Apply brightfield processing to channel 1"
    )

    parser.add_argument(
        "--wells", 
        type=str,
        nargs="+",
        help="Specific wells to process (e.g., A01 B02). If not specified, all wells are processed."
    )
    
    parser.add_argument(
        "--z-stack", 
        action="store_true",
        help="Enable Z-stack handling"
    )
    
    args = parser.parse_args()
    
    # Validate input
    plate_path = Path(args.plate_folder)
    if not plate_path.exists():
        print(f"Error: Plate folder '{args.plate_folder}' does not exist", file=sys.stderr)
        return 1
    
    # Build preprocessing functions dictionary
    preprocessing_funcs = {}
    if args.bf_process:
        preprocessing_funcs["1"] = process_bf
    
    # Build channel weights for composite
    if len(args.reference_channels) > 1:
        # Equal weights for all channels by default
        weight = 1.0 / len(args.reference_channels)
        composite_weights = {channel: weight for channel in args.reference_channels}
    else:
        composite_weights = None
    
    # Process the plate folder
    try:
        if args.z_stack:
            # Use the version that handles Z-stacks
            modified_process_plate_folder(
                args.plate_folder,
                reference_channels=args.reference_channels,
                preprocessing_funcs=preprocessing_funcs,
                composite_weights=composite_weights,
                margin_ratio=args.margin_ratio,
                tile_overlap=args.tile_overlap,
                max_shift=args.max_shift,
                well_filter=args.wells
            )
        else:
            # Use the standard version
            process_plate_folder(
                args.plate_folder,
                reference_channels=args.reference_channels,
                preprocessing_funcs=preprocessing_funcs,
                composite_weights=composite_weights,
                margin_ratio=args.margin_ratio,
                tile_overlap=args.tile_overlap,
                max_shift=args.max_shift,
                well_filter=args.wells
            )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())