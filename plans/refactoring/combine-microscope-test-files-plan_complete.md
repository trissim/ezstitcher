# Plan: Combine ImageXpress and Opera Phenix Test Files

Status: Complete
Progress: 100%
Last Updated: 2023-07-15
Dependencies: None

## 1. Problem Analysis

Currently, we have two nearly identical test files for testing the ImageXpress and Opera Phenix microscope formats:
- `tests/integration/test_synthetic_imagexpress_auto.py`
- `tests/integration/test_synthetic_opera_phenix_auto.py`

These files have significant duplication with only minor differences:

**Key differences:**
1. The microscope format parameter in the SyntheticMicroscopyGenerator ("ImageXpress" vs "OperaPhenix")
2. The base test directory path ("imagexpress_refactored_auto" vs "opera_phenix_refactored_auto")
3. For Opera Phenix, tests use the "Images" subdirectory as the input directory
4. The microscope_type parameter in process_plate_auto ("auto" vs "OperaPhenix")
5. The ImageXpress tests use auto_image_size=True, but Opera Phenix tests don't yet

**Requirements:**
- Combine both test files into a single, parameterized test file
- Maintain all existing test functionality
- Make it easy to add new microscope formats in the future
- Allow for microscope-specific customizations when needed
- Keep the code DRY (Don't Repeat Yourself)
- Make it easy to modify test parameters for both microscope types simultaneously

## 2. High-Level Solution

Create a single parameterized test file that:
1. Uses pytest's parameterization to run tests for both microscope types
2. Defines microscope-specific configurations in a central location
3. Uses fixtures that adapt to the microscope type
4. Maintains all existing test functionality

## 3. Implementation Details

### 3.1 Create a New Test File

Create a new file `tests/integration/test_synthetic_microscopes.py` that will replace both existing test files.

### 3.2 Define Microscope Configurations

```python
# Define microscope configurations
MICROSCOPE_CONFIGS = {
    "ImageXpress": {
        "format": "ImageXpress",
        "test_dir_name": "imagexpress_refactored",
        "input_subdir": None,  # No subdirectory needed
        "microscope_type": "auto",  # Use auto-detection
        "auto_image_size": True,
    },
    "OperaPhenix": {
        "format": "OperaPhenix",
        "test_dir_name": "opera_phenix_refactored",
        "input_subdir": "Images",  # Opera Phenix uses Images subdirectory
        "microscope_type": "OperaPhenix",  # Explicitly specify type
        "auto_image_size": True,  # Update to use auto_image_size
    }
}
```

### 3.3 Create Parameterized Fixtures

```python
@pytest.fixture(scope="module", params=list(MICROSCOPE_CONFIGS.keys()))
def microscope_config(request):
    """Provide microscope configuration based on the parameter."""
    return MICROSCOPE_CONFIGS[request.param]

@pytest.fixture(scope="module")
def base_test_dir(microscope_config):
    """Create base test directory for the specific microscope type."""
    base_dir = Path(__file__).parent / "tests_data" / microscope_config["test_dir_name"]

    # Delete the directory if it exists
    if base_dir.exists():
        shutil.rmtree(base_dir)

    # Create the directory
    base_dir.mkdir(parents=True, exist_ok=True)

    yield base_dir

    # Uncomment to clean up after tests
    # shutil.rmtree(base_dir)
```

### 3.4 Update Data Generation Fixtures

```python
@pytest.fixture
def flat_plate_dir(test_dir, microscope_config):
    """Create synthetic flat plate data for the specified microscope type."""
    plate_dir = test_dir / "flat_plate"
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=(3, 3),
        tile_size=(128, 128),
        overlap_percent=10,
        wavelengths=2,
        z_stack_levels=1,
        cell_size_range=(5, 10),
        format=microscope_config["format"],
        auto_image_size=microscope_config["auto_image_size"]
    )
    generator.generate_dataset()

    # Create a copy of the original data for inspection
    original_dir = test_dir / "flat_plate_original"
    if not original_dir.exists():
        shutil.copytree(plate_dir, original_dir)

    # Return the appropriate directory based on microscope type
    if microscope_config["input_subdir"]:
        return plate_dir / microscope_config["input_subdir"]
    return plate_dir
```

### 3.5 Update Test Functions

```python
def test_flat_plate_minimal(flat_plate_dir, microscope_config):
    """Test processing a flat plate with minimal configuration."""
    success = process_plate_auto(
        flat_plate_dir,
        microscope_type=microscope_config["microscope_type"]
    )
    assert success, "Flat plate processing failed"
```

### 3.6 Add Additional Test Parameters

Add the ability to customize test parameters for specific tests:

```python
# Test-specific parameters that can be customized per microscope
TEST_PARAMS = {
    "ImageXpress": {
        "test_flat_plate_minimal": {
            "grid_size": (3, 3),
            "tile_size": (128, 128),
            "overlap_percent": 10,
        },
        # Add more test-specific parameters as needed
    },
    "OperaPhenix": {
        "test_flat_plate_minimal": {
            "grid_size": (3, 3),
            "tile_size": (128, 128),
            "overlap_percent": 10,
        },
        # Add more test-specific parameters as needed
    }
}

@pytest.fixture
def test_params(request, microscope_config):
    """Get test-specific parameters for the current microscope type."""
    test_name = request.node.name
    return TEST_PARAMS.get(microscope_config["format"], {}).get(test_name, {})
```

## 4. Testing Plan

1. Create the new combined test file
2. Run the tests to ensure they work correctly for both microscope types
3. Verify that all tests pass and produce the same results as the original separate test files
4. Remove the original test files once the combined file is working correctly

## 5. Implementation Steps

1. Create the new test file with the parameterized fixtures and tests
2. Update the microscope configurations to include all necessary parameters
3. Implement the test-specific parameter customization
4. Run the tests to verify functionality
5. Remove the original test files

## 6. Future Enhancements

1. Add support for additional microscope types
2. Implement more customizable test parameters
3. Add the ability to run tests for specific microscope types only
4. Create a test configuration file for easier customization

## 7. Completion Summary

Date: 2023-07-15

The implementation has been completed successfully. A new combined test file `tests/integration/test_synthetic_microscopes.py` has been created that:

1. Uses pytest's parameterization to run tests for both ImageXpress and Opera Phenix microscope types
2. Defines microscope-specific configurations in a central location (MICROSCOPE_CONFIGS)
3. Provides a flexible test parameter system that allows customizing test parameters per microscope type
4. Maintains all existing test functionality from the original separate test files
5. Reduces code duplication and makes it easier to maintain tests for multiple microscope types
6. Makes it easy to add support for additional microscope types in the future

The tests have been run and verified to work correctly for both microscope types. The original separate test files can now be removed once this implementation is fully tested and approved.
