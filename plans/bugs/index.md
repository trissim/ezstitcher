# EZStitcher Bug Fixes Index

## Status: In Progress
## Last Updated: 2023-07-20

This index tracks the bugs identified in the EZStitcher codebase and their fix plans.

## Active Bug Fixes

1. **Grid Size Mismatch Fix** - [[plans/bugs/grid-size-mismatch-fix-plan.md]]
   - Status: Complete for ImageXpress, In Progress for Opera Phenix
   - Progress: 50%
   - Description: Fix for the mismatch between the grid size used in the data generator (4,4) and the grid size passed to the stitcher (2,2).
   - Priority: High

2. **Wavelength Pattern Bug Fix** - [[plans/bugs/wavelength-pattern-bug-fix-plan.md]]
   - Status: Complete for ImageXpress, In Progress for Opera Phenix
   - Progress: 50%
   - Description: Fix for the issue where the pattern being stitched isn't changing for wavelength 2 (w2) during Z-stack per-plane stitching.
   - Priority: High

## Completed Bug Fixes

None yet.

## Implementation Order

1. Grid Size Mismatch Fix (partially complete)
2. Wavelength Pattern Bug Fix (partially complete)
3. ZStackStitcher Refactoring (planned)

## Notes

- The Grid Size Mismatch fix has been implemented and works for ImageXpress data, but Opera Phenix data still has issues that need to be addressed separately.
- The Wavelength Pattern bug has been fixed for ImageXpress data by using the PatternMatcher class to detect all unique patterns for each well.
- Both bugs are related to the Z-stack per-plane stitching functionality.
- The ZStackStitcher class has several code smells and should be refactored for better maintainability.