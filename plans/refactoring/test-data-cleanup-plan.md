# Test Data Cleanup Plan

Status: Complete
Progress: 100%
Last Updated: 2023-07-11
Dependencies: None

## 1. Problem Analysis

Currently, test data directories are not being deleted before starting a new test, which can lead to:

1. **File conflicts**: When a test tries to create a directory that already exists
2. **Inconsistent test results**: When a test uses data from a previous test run
3. **Disk space usage**: Accumulation of test data over time

We need to ensure that all test data directories are deleted before starting a new test to avoid these issues.

## 2. High-Level Solution

1. **Modify the test fixtures** to delete the test data directories before creating new ones
2. **Implement a cleanup function** that can be called at the beginning of each test
3. **Use pytest's `tmpdir` or `tmp_path` fixtures** to automatically handle cleanup

## 3. Implementation Details

### 3.1 Modify Test Fixtures

For both ImageXpress and Opera Phenix test files, we'll modify the `base_test_dir` fixture to delete the test data directory if it exists:

```python
@pytest.fixture(scope="module")
def base_test_dir():
    """Create base test directory for tests."""
    base_dir = Path(__file__).parent / "tests_data" / "test_file_name"

    # Delete the directory if it exists
    if base_dir.exists():
        shutil.rmtree(base_dir)

    # Create the directory
    base_dir.mkdir(parents=True, exist_ok=True)

    yield base_dir

    # Optionally clean up after tests
    # shutil.rmtree(base_dir)
```

### 3.2 Implement Cleanup Function

We'll also implement a cleanup function that can be called at the beginning of each test:

```python
def clean_test_dir(test_dir):
    """Clean test directory."""
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
```

### 3.3 Use pytest's `tmpdir` or `tmp_path` Fixtures

Alternatively, we can use pytest's `tmpdir` or `tmp_path` fixtures to automatically handle cleanup:

```python
@pytest.fixture
def test_dir(tmp_path):
    """Create test-specific directory."""
    # tmp_path is automatically created and deleted by pytest
    return tmp_path
```

## 4. Validation

### 4.1 Unit Tests

1. Run the tests to verify that they work correctly with the new cleanup logic
2. Verify that the tests don't fail with "directory already exists" errors
3. Verify that the tests don't use data from previous test runs

### 4.2 Integration Tests

1. Run all tests to verify that the changes don't break existing functionality
2. Verify that the test data is correctly organized in the `/tests/tests_data/` directory

## 5. Implementation Order

1. ✅ Modify the `base_test_dir` fixture in both test files
2. ✅ Implement the cleanup function
3. ✅ Update the test fixtures to use the cleanup function
4. ✅ Run the tests to verify the changes

## 6. Benefits

1. **Consistent test results**: Each test runs with a clean slate
2. **No file conflicts**: Tests don't fail because directories already exist
3. **Reduced disk space usage**: Test data is cleaned up after each test run
4. **Improved test isolation**: Each test has its own clean directory

## 7. Risks and Mitigations

1. **Risk**: Deleting test data might affect debugging
   **Mitigation**: Keep a copy of the test data for debugging purposes

2. **Risk**: Deleting test data might slow down tests
   **Mitigation**: Only delete test data when necessary

## 8. References

- `tests/test_synthetic_imagexpress_refactored_auto.py`
- `tests/test_synthetic_opera_phenix_refactored_auto_new.py`

## 9. Completion Summary

We successfully implemented test data cleanup in both test files. The key changes were:

1. Modified the `base_test_dir` fixture in both test files to delete the test data directory if it exists
2. Ensured that each test creates a copy of the original data with an `_original` suffix
3. Verified that the tests work correctly with the new cleanup logic

All tests are now passing, confirming that our changes work correctly. The test data is now properly organized in the `/tests/tests_data/` directory, with each test file having its own directory and each test method having its own subdirectory. The original data is preserved with an `_original` suffix, allowing us to see how the data may have changed throughout the test.

Date: 2023-07-11
