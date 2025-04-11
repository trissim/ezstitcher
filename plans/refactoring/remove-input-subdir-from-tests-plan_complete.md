# Plan: Remove Manual Directory Structure Handling in Microscope Tests

Status: Complete
Progress: 100%
Last Updated: 2023-07-15
Dependencies: None

## 1. Problem Analysis

In the current implementation of `test_synthetic_microscopes.py`, we're manually handling the directory structure differences between ImageXpress and Opera Phenix by:

1. Including an `input_subdir` parameter in the microscope configurations
2. Manually appending this subdirectory to the plate directory in the fixture return value

```python
# In MICROSCOPE_CONFIGS:
"OperaPhenix": {
    "format": "OperaPhenix",
    "test_dir_name": "opera_phenix_refactored",
    "input_subdir": "Images",  # Opera Phenix uses Images subdirectory
    "microscope_type": "OperaPhenix",
    "auto_image_size": True,
}

# In the fixture:
# Return the appropriate directory based on microscope type
if microscope_config["input_subdir"]:
    return plate_dir / microscope_config["input_subdir"]
return plate_dir
```

This is problematic because:

1. **Duplication of Logic**: The core library already has the capability to handle different directory structures based on the microscope type through the `ImageLocator` class.

2. **Unnecessary Coupling**: The tests are coupled to implementation details about directory structures that should be encapsulated within the core library.

3. **Violation of Responsibility**: The microscope type should fully determine how to handle the data structure, not the test code.

4. **Reduced Maintainability**: If the directory structure handling in the core library changes, the tests would need to be updated separately.

## 2. High-Level Solution

The solution is to let the core library handle the directory structure differences by:

1. Removing the `input_subdir` parameter from the microscope configurations
2. Always returning the plate directory from the fixtures
3. Letting `process_plate_auto` handle the directory structure based on the microscope type

This will make the tests more robust and better aligned with how the library is intended to be used.

## 3. Implementation Details

### 3.1 Update Microscope Configurations

Remove the `input_subdir` parameter from `MICROSCOPE_CONFIGS`:

```python
# Before:
MICROSCOPE_CONFIGS = {
    "ImageXpress": {
        "format": "ImageXpress",
        "test_dir_name": "imagexpress_refactored",
        "input_subdir": None,  # No subdirectory needed
        "microscope_type": "auto",
        "auto_image_size": True,
    },
    "OperaPhenix": {
        "format": "OperaPhenix",
        "test_dir_name": "opera_phenix_refactored",
        "input_subdir": "Images",  # Opera Phenix uses Images subdirectory
        "microscope_type": "OperaPhenix",
        "auto_image_size": True,
    }
}

# After:
MICROSCOPE_CONFIGS = {
    "ImageXpress": {
        "format": "ImageXpress",
        "test_dir_name": "imagexpress_refactored",
        "microscope_type": "auto",
        "auto_image_size": True,
    },
    "OperaPhenix": {
        "format": "OperaPhenix",
        "test_dir_name": "opera_phenix_refactored",
        "microscope_type": "OperaPhenix",
        "auto_image_size": True,
    }
}
```

### 3.2 Update Fixture Return Values

Modify the fixture return values to always return the plate directory:

```python
# Before:
@pytest.fixture
def flat_plate_dir(test_dir, microscope_config, test_params):
    # ... (code to generate data) ...

    # Return the appropriate directory based on microscope type
    if microscope_config["input_subdir"]:
        return plate_dir / microscope_config["input_subdir"]
    return plate_dir

# After:
@pytest.fixture
def flat_plate_dir(test_dir, microscope_config, test_params):
    # ... (code to generate data) ...

    # Always return the plate directory
    return plate_dir
```

### 3.3 Verify Core Library Handling

Ensure that the core library correctly handles the directory structure differences. This involves checking that `process_plate_auto` correctly:

1. Detects the microscope type
2. Finds the appropriate directory structure
3. Processes the images correctly

## 4. Testing Plan

1. Run the tests after making the changes to ensure they still pass
2. Verify that the tests work correctly for both ImageXpress and Opera Phenix
3. Check the logs to ensure that the core library is correctly handling the directory structure

## 5. Implementation Steps

1. Update the `MICROSCOPE_CONFIGS` dictionary to remove the `input_subdir` parameter
2. Modify the `flat_plate_dir` and `zstack_plate_dir` fixtures to always return the plate directory
3. Run the tests to verify that they still pass
4. Check the logs to ensure that the core library is correctly handling the directory structure

## 6. Potential Risks and Mitigations

**Risk**: The core library might not correctly handle the directory structure differences.
**Mitigation**: Add more detailed logging to verify the directory structure detection and handling.

**Risk**: The tests might fail if the core library's behavior changes.
**Mitigation**: Ensure that the tests are robust against changes in the library's implementation by focusing on the expected outcomes rather than implementation details.

## 7. Future Enhancements

1. Add more detailed logging to the core library to make it easier to debug directory structure issues
2. Consider adding a test that explicitly verifies the directory structure handling
3. Document the expected directory structures for different microscope types

## 8. Completion Summary

Date: 2023-07-15

The implementation has been completed successfully. The following changes were made:

1. Removed the `input_subdir` parameter from the `MICROSCOPE_CONFIGS` dictionary
2. Modified the `flat_plate_dir` and `zstack_plate_dir` fixtures to always return the plate directory
3. Verified that the tests still pass for both ImageXpress and Opera Phenix microscope types

These changes have improved the test code by:

1. Removing duplication of logic that should be handled by the core library
2. Eliminating unnecessary coupling between tests and implementation details
3. Respecting the principle that the microscope type should fully determine how to handle the data structure
4. Making the tests more robust against changes in the library's implementation

The core library now correctly handles the directory structure differences based on the microscope type, as evidenced by the successful test runs. This makes the tests more maintainable and better aligned with the library's design.
