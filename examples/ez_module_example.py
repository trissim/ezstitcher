"""
Example demonstrating the use of the EZ module.

This example shows how to use the simplified interface provided by the EZ module
to stitch microscopy images with minimal code.
"""

from pathlib import Path
from ezstitcher import stitch_plate, EZStitcher

# Path to your microscopy data
# Replace with your actual data path
plate_path = Path("path/to/your/microscopy/data")

# Example 1: One-liner stitching
print("Example 1: One-liner stitching")
print("------------------------------")
print("Code: stitch_plate(plate_path)")
# Uncomment to run:
# output_path = stitch_plate(plate_path)
# print(f"Stitching complete! Output saved to: {output_path}")
print()

# Example 2: Stitching with custom options
print("Example 2: Stitching with custom options")
print("---------------------------------------")
print("Code: stitch_plate(plate_path, normalize=True, z_method='focus')")
# Uncomment to run:
# output_path = stitch_plate(
#     plate_path,
#     normalize=True,
#     z_method="focus"
# )
# print(f"Stitching complete! Output saved to: {output_path}")
print()

# Example 3: Using the EZStitcher class
print("Example 3: Using the EZStitcher class")
print("------------------------------------")
print("""Code:
stitcher = EZStitcher(plate_path)
stitcher.set_options(channel_weights=[0.7, 0.3, 0])
output_path = stitcher.stitch()
""")
# Uncomment to run:
# stitcher = EZStitcher(plate_path)
# stitcher.set_options(channel_weights=[0.7, 0.3, 0])
# output_path = stitcher.stitch()
# print(f"Stitching complete! Output saved to: {output_path}")
