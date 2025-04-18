#!/usr/bin/env python3
"""
Command-line interface for the EZStitcher package.
"""

import argparse
import sys
import logging
from pathlib import Path

# Import from main module for consistent API
from ezstitcher.core.main import (
    process_plate_folder,
    modified_process_plate_folder,
    process_bf,
    find_best_focus
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description="EZStitcher - Easy Microscopy Image Stitching Tool"
    )

    # Core arguments
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

    # Z-stack related arguments
    z_stack_group = parser.add_argument_group('Z-Stack Processing')

    z_stack_group.add_argument(
        "--focus-detect",
        action="store_true",
        help="Enable best focus detection for Z-stacks"
    )

    z_stack_group.add_argument(
        "--focus-wavelength",
        type=str,
        default="1",
        help="Wavelength to use for focus detection (default: 1, use 'all' for all wavelengths)"
    )

    z_stack_group.add_argument(
        "--focus-method",
        type=str,
        choices=["combined", "laplacian", "normalized_variance", "tenengrad", "fft", "adaptive_fft"],
        default="combined",
        help="Focus detection method to use (default: combined)"
    )

    z_stack_group.add_argument(
        "--create-projections",
        action="store_true",
        help="Create 3D projections from Z-stacks"
    )

    z_stack_group.add_argument(
        "--projection-types",
        type=str,
        default="max,mean",
        help="Comma-separated list of projection types to create (default: max,mean)"
    )

    z_stack_group.add_argument(
        "--stitch-method",
        type=str,
        default="best_focus",
        help="Method for Z-stack stitching (default: best_focus, or specific z-index)"
    )

    # Legacy Z-stack option for backward compatibility
    parser.add_argument(
        "--z-stack",
        action="store_true",
        help="Enable basic Z-stack handling (legacy option)"
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

    # Parse projection types
    if args.projection_types:
        projection_types = args.projection_types.split(',')
    else:
        projection_types = ['max', 'mean']

    # Determine if we should use Z-stack processing
    use_zstack_handler = (args.z_stack or args.focus_detect or
                         args.create_projections or args.stitch_method != "best_focus")

    # Process the plate folder
    try:
        if use_zstack_handler:
            # Use the enhanced version that handles Z-stacks
            modified_process_plate_folder(
                args.plate_folder,
                reference_channels=args.reference_channels,
                preprocessing_funcs=preprocessing_funcs,
                composite_weights=composite_weights,
                margin_ratio=args.margin_ratio,
                tile_overlap=args.tile_overlap,
                max_shift=args.max_shift,
                well_filter=args.wells,
                focus_detect=args.focus_detect,
                focus_method=args.focus_method,
                create_projections=args.create_projections,
                projection_types=projection_types,
                stitch_z_reference=args.stitch_method
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