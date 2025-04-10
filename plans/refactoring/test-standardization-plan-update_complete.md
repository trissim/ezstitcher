# Test Standardization Plan Update

Status: Complete  
Progress: 100%  
Last Updated: 2023-07-11  
Dependencies: None

## 1. Problem Analysis

We've made progress in standardizing the test files, but we encountered issues with the Opera Phenix Z-stack tests:

1. The Opera Phenix files were being copied to the `TimePoint_1` directory, but the system was not detecting the image patterns correctly.
2. The error message was: "No image patterns detected in /home/ts/code/projects/ezstitcher/tests/tests_data/opera_phenix_refactored_auto/test_zstack_projection_minimal/zstack_plate/TimePoint_1"

This suggested that there was an issue with how the Opera Phenix files were being processed or how the image patterns were being detected.

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

We investigated the issue and found that:

1. The synthetic data generator was correctly creating the Opera Phenix files in the `Images` directory
2. The test was trying to use the root directory instead of the `Images` directory
3. The system was not finding the image files because they were in the `Images` directory

### 3.2 Fix the Issue

We updated the Opera Phenix test file to use the `Images` directory:

```python
def test_flat_plate_minimal(flat_plate_dir):
    """Test processing a flat plate with minimal configuration."""
    # For Opera Phenix, we need to use the Images directory
    images_dir = flat_plate_dir / "Images"
    success = process_plate_auto(
        images_dir,
        microscope_type="OperaPhenix"  # Use explicit microscope type
    )
    assert success, "Flat plate processing failed"
```

We applied this change to all test methods in the Opera Phenix test file.

## 4. Validation

### 4.1 Unit Tests

We ran the Opera Phenix tests to verify they work correctly:

```
tests/test_synthetic_opera_phenix_refactored_auto_new.py::test_flat_plate_minimal PASSED
tests/test_synthetic_opera_phenix_refactored_auto_new.py::test_zstack_projection_minimal PASSED
tests/test_synthetic_opera_phenix_refactored_auto_new.py::test_zstack_per_plane_minimal PASSED
tests/test_synthetic_opera_phenix_refactored_auto_new.py::test_multi_channel_minimal PASSED
```

We also ran the ImageXpress tests to verify they still work correctly:

```
tests/test_synthetic_imagexpress_refactored_auto.py::test_flat_plate_minimal PASSED
tests/test_synthetic_imagexpress_refactored_auto.py::test_zstack_projection_minimal PASSED
tests/test_synthetic_imagexpress_refactored_auto.py::test_zstack_per_plane_minimal PASSED
tests/test_synthetic_imagexpress_refactored_auto.py::test_multi_channel_minimal PASSED
```

### 4.2 Integration Tests

We ran all tests to verify that the changes don't break existing functionality:

```
tests/test_synthetic_opera_phenix_refactored_auto_new.py: 4 passed, 16 warnings
tests/test_synthetic_imagexpress_refactored_auto.py: 4 passed, 16 warnings
```

We also verified that the test data is correctly organized in the `/tests/tests_data/` directory.

## 5. Implementation Order

1. ✅ Investigate the issue with Opera Phenix Z-stack tests
2. ✅ Fix the issue
3. ✅ Run the tests to verify the changes

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

## 9. Completion Summary

We successfully standardized the test files for both ImageXpress and Opera Phenix microscope types. The key changes were:

1. Updated both test files to use the same test cases
2. Updated both test files to use the same test structure
3. Updated both test files to output test data to `/tests/tests_data/` with a folder for each test file and a subfolder for each test method
4. Fixed the issue with Opera Phenix Z-stack tests by using the `Images` directory

All tests are now passing, confirming that our changes work correctly.

Date: 2023-07-11
