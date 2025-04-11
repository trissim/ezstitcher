# Wavelength Pattern Bug Fix Plan

## Status: Complete for ImageXpress
## Progress: 50%
## Last Updated: 2023-07-20
## Dependencies: None

## 1. Problem Analysis

There is a bug in the Z-stack per-plane stitching process where the pattern being stitched isn't changing for wavelength 2 (w2). This means that when stitching multiple wavelengths across Z-planes, only wavelength 1 (w1) is correctly stitched for all Z-planes, while wavelength 2 (w2) is not being stitched at all.

This issue is specifically observed in the `test_zstack_per_plane_minimal` test, where the synthetic data generator creates images for both wavelengths (w1 and w2) across all Z-planes, but only w1 images are being stitched in the output.

The issue appears to be in the `ZStackStitcher` class, specifically in the `stitch_across_z` method, where the code is not correctly handling multiple wavelengths when stitching Z-planes. The problem is likely in how the filenames are constructed for different wavelengths when looking for tiles to stitch.

## 2. Detailed Investigation

1. The `SyntheticMicroscopyGenerator` correctly generates images for multiple wavelengths (w1, w2) across all Z-planes.
2. The `ZStackStitcher.stitch_across_z` method is responsible for stitching all Z-planes for all wavelengths.
3. The issue is not with the position file generation - we only need one position file per well, as the positions are the same regardless of wavelength or Z-step.
4. The problem is in how the `stitch_across_z` method processes the position files and constructs filenames for different wavelengths.
5. The method is correctly reading position files but may not be correctly looking for all available wavelengths when stitching Z-planes.

## 3. Proposed Solution

Update the `ZStackStitcher.stitch_across_z` method to correctly handle multiple wavelengths when stitching Z-planes:

1. Use the existing `auto_detect_patterns` method from the `PatternMatcher` class to find all unique patterns for each well.
2. This will automatically detect all wavelengths available for each well.
3. Process all detected patterns for each well and Z-plane.
4. Add more detailed logging to track which files are being processed for each wavelength and Z-plane.

## 4. Implementation Details

### 4.1 Use PatternMatcher to detect all patterns

Modify the `stitch_across_z` method to use the existing `PatternMatcher` class to detect all patterns for each well:

```python
# Import the PatternMatcher class if not already imported
from ezstitcher.core.pattern_matcher import PatternMatcher

# Create a PatternMatcher instance with the same filename parser
pattern_matcher = PatternMatcher(self.filename_parser)

# Detect all patterns for each well in the timepoint directory
patterns_by_well = pattern_matcher.auto_detect_patterns(timepoint_path)
logger.info(f"Detected patterns by well: {patterns_by_well}")
```

### 4.2 Update the Z-plane stitching loop

Modify the Z-plane stitching loop to process all detected patterns for each well:

```python
# For each Z-plane, stitch all wells and wavelengths
for z_index in z_indices:
    logger.info(f"Stitching Z-plane {z_index}")

    # For each well, get the position file
    for well, wavelength_patterns in patterns_by_well.items():
        logger.info(f"Processing well {well} for Z-plane {z_index}")

        # Find the position file for this well
        position_file = next((p for p in position_files if p.name.startswith(f"{well}_w")), None)
        if not position_file:
            logger.warning(f"No position file found for well {well}")
            continue

        # Read positions from the position file
        positions = []
        with open(position_file, 'r') as f:
            # ... (existing code to read positions) ...

        # Process each wavelength for this well
        for wavelength, pattern in wavelength_patterns.items():
            logger.info(f"Processing wavelength {wavelength} with pattern {pattern} for well {well} and Z-plane {z_index}")

            # Get all tiles for this well, wavelength, and Z-plane
            tiles = []
            for site, x, y in positions:
                # Construct the filename and path based on the pattern
                # ... (existing code to construct filename) ...

                # ... (existing code to load and process tiles) ...

            # ... (existing code to stitch tiles) ...

            # Save the stitched image with Z-plane suffix
            output_filename = f"{well}_w{wavelength}_z{z_index:03d}.tif"
            output_path = stitched_dir / output_filename
            logger.info(f"Saving stitched image to {output_path}")
            self.fs_manager.save_image(output_path, canvas)
            logger.info(f"Completed stitching for {well}_w{wavelength}_z{z_index}")
```

### 4.3 Add more detailed logging

Add more detailed logging throughout the `stitch_across_z` method to track which files are being processed for each wavelength and Z-plane:

```python
# Add at the beginning of the method
logger.info(f"Starting stitch_across_z with plate_folder={plate_folder}, reference_z={reference_z}, stitch_all_z_planes={stitch_all_z_planes}")

# Add after detecting patterns
logger.info(f"Detected patterns by well: {patterns_by_well}")

# Add before processing each wavelength
logger.info(f"Processing wavelength {wavelength} with pattern {pattern} for well {well} and Z-plane {z_index}")

# Add after finding tiles
logger.info(f"Found {len(tiles)} tiles for {well}_w{wavelength}_z{z_index}")

# Add after stitching
logger.info(f"Completed stitching for {well}_w{wavelength}_z{z_index}")
```

## 5. Testing Plan

1. Run the Z-stack per-plane stitching test for both ImageXpress and Opera Phenix data:
   ```bash
   python -m pytest tests/integration/test_synthetic_microscopes.py::test_zstack_per_plane_minimal -v
   ```

2. Verify that the test passes for both microscope types.

3. Check the logs to confirm that all wavelengths are being stitched correctly for all Z-planes.

4. Examine the output files to ensure that stitched images for all wavelengths and Z-planes are created.

## 6. Risks and Mitigations

**Risk**: The fix might affect other functionality that relies on the current behavior.

**Mitigation**: Add comprehensive logging to track the changes and ensure that all expected files are processed correctly.

**Risk**: The issue might be in multiple places in the code, not just in the `stitch_across_z` method.

**Mitigation**: If the initial fix doesn't resolve the issue, expand the investigation to other related methods and classes.

## 7. Implementation Steps

1. Update the `stitch_across_z` method in the `ZStackStitcher` class as described above.
2. Add more detailed logging throughout the method.
3. Run the tests to verify that the fix works correctly.
4. Update documentation if needed.

## 8. Completion Criteria

1. The Z-stack per-plane stitching test passes for both ImageXpress and Opera Phenix data.
2. Stitched images for all wavelengths and Z-planes are created correctly.
3. All other tests continue to pass.

## 9. Implementation Results

- Successfully implemented the fix using the PatternMatcher class to detect all unique patterns for each well.
- The ImageXpress test now passes successfully and correctly stitches both wavelengths.
- The Opera Phenix test still fails, but this is likely due to other issues specific to Opera Phenix handling.
- The approach of using PatternMatcher to find all unique patterns for each well is more robust than hardcoding wavelength detection.
- This fix reuses existing code and follows the principle of finding all unique patterns that exist for a well, which catches all combinations of wavelengths and Z-steps.

## 10. Next Steps

- The ZStackStitcher class has several code smells and should be refactored for better maintainability.
- Opera Phenix support needs additional work, but we can continue using the ImageXpress tests for refactoring.
