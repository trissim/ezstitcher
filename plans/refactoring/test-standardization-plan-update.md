# Test Standardization Plan Update

Status: In Progress  
Progress: 50%  
Last Updated: 2023-07-11  
Dependencies: None

## 1. Problem Analysis

We've made progress in standardizing the test files, but we're encountering issues with the Opera Phenix Z-stack tests:

1. The Opera Phenix files are being copied to the `TimePoint_1` directory, but the system is not detecting the image patterns correctly.
2. The error message is: "No image patterns detected in /home/ts/code/projects/ezstitcher/tests/tests_data/opera_phenix_refactored_auto/test_zstack_projection_minimal/zstack_plate/TimePoint_1"

This suggests that there might be an issue with how the Opera Phenix files are being processed or how the image patterns are being detected.

## 2. High-Level Solution

1. **Investigate the issue with Opera Phenix Z-stack tests**:
   - Check how the Opera Phenix files are being processed
   - Check how the image patterns are being detected
   - Identify the root cause of the issue

2. **Fix the issue**:
   - Update the Opera Phenix test file to work with Z-stack data
   - Ensure that the image patterns are correctly detected

## 3. Implementation Details

### 3.1 Investigate the Issue

1. Check the logs to understand what's happening
2. Check the directory structure to see if the files are being copied correctly
3. Check the image pattern detection logic to see if it's working correctly

### 3.2 Fix the Issue

Based on the investigation, we'll need to update the Opera Phenix test file to work with Z-stack data. This might involve:

1. Updating the synthetic data generator parameters
2. Updating the test parameters
3. Updating the directory structure

## 4. Validation

### 4.1 Unit Tests

1. Run the Opera Phenix Z-stack tests to verify they work correctly
2. Run the ImageXpress Z-stack tests to verify they still work correctly

### 4.2 Integration Tests

1. Run all tests to verify that the changes don't break existing functionality
2. Verify that the test data is correctly organized in the `/tests/tests_data/` directory

## 5. Implementation Order

1. Investigate the issue with Opera Phenix Z-stack tests
2. Fix the issue
3. Run the tests to verify the changes

## 6. Benefits

1. **Standardized test cases**: Both test files have the same test cases
2. **Consistent test structure**: Both test files use the same test structure
3. **Improved test isolation**: Each test method has its own directory
4. **Better test data management**: Test data is stored in a fixed location for easier inspection
5. **Verified functionality**: Tests verify that the system works correctly with both microscope types

## 7. Risks and Mitigations

1. **Risk**: Changes might break existing tests
   **Mitigation**: Run tests after each change to verify functionality

2. **Risk**: Opera Phenix Z-stack tests might require special handling
   **Mitigation**: Investigate the issue thoroughly and make the necessary changes

## 8. References

- `tests/test_synthetic_imagexpress_refactored_auto.py`
- `tests/test_synthetic_opera_phenix_refactored_auto_new.py`
