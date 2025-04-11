# Plan: Fix Wells Parameter in Synthetic Microscope Tests

Status: Complete
Progress: 100%
Last Updated: 2023-07-15
Dependencies: None

## 1. Problem Analysis

In the current implementation of `test_synthetic_microscopes.py`, the `wells` parameter is included in the `syn_data_params` dictionary but is not being passed to the `SyntheticMicroscopyGenerator` constructor in the fixture functions. This means that any changes to the `wells` parameter in `syn_data_params` don't affect the generated data.

**Current Code:**

```python
syn_data_params = {
    "grid_size": (3, 3),
    "tile_size": (128, 128),
    "overlap_percent": 10,
    "wavelengths": 2,
    "cell_size_range": (5, 10),
    "wells": ['A01', 'B02'],  # This parameter is not being used
}
```

In the fixture functions, the parameters are extracted individually:

```python
# Get parameters from test_params with defaults if not specified
grid_size = test_params.get("grid_size", (3, 3))
tile_size = test_params.get("tile_size", (128, 128))
overlap_percent = test_params.get("overlap_percent", 10)
wavelengths = test_params.get("wavelengths", 2)
z_stack_levels = test_params.get("z_stack_levels", 1)
cell_size_range = test_params.get("cell_size_range", (5, 10))
# wells parameter is missing here
```

And then passed to the generator:

```python
generator = SyntheticMicroscopyGenerator(
    output_dir=str(plate_dir),
    grid_size=grid_size,
    tile_size=tile_size,
    overlap_percent=overlap_percent,
    wavelengths=wavelengths,
    z_stack_levels=z_stack_levels,
    cell_size_range=cell_size_range,
    format=microscope_config["format"],
    auto_image_size=microscope_config["auto_image_size"]
    # wells parameter is missing here
)
```

## 2. High-Level Solution

The solution is to extract the `wells` parameter from `test_params` and pass it to the `SyntheticMicroscopyGenerator` constructor in both fixture functions.

## 3. Implementation Details

### 3.1 Update the `flat_plate_dir` Fixture

```python
@pytest.fixture
def flat_plate_dir(test_dir, microscope_config, test_params):
    """Create synthetic flat plate data for the specified microscope type."""
    plate_dir = test_dir / "flat_plate"

    # Get parameters from test_params with defaults if not specified
    grid_size = test_params.get("grid_size", (3, 3))
    tile_size = test_params.get("tile_size", (128, 128))
    overlap_percent = test_params.get("overlap_percent", 10)
    wavelengths = test_params.get("wavelengths", 2)
    z_stack_levels = test_params.get("z_stack_levels", 1)
    cell_size_range = test_params.get("cell_size_range", (5, 10))
    wells = test_params.get("wells", ['A01'])  # Add this line

    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=grid_size,
        tile_size=tile_size,
        overlap_percent=overlap_percent,
        wavelengths=wavelengths,
        z_stack_levels=z_stack_levels,
        cell_size_range=cell_size_range,
        wells=wells,  # Add this line
        format=microscope_config["format"],
        auto_image_size=microscope_config["auto_image_size"]
    )
    generator.generate_dataset()

    # ... rest of the function
```

### 3.2 Update the `zstack_plate_dir` Fixture

```python
@pytest.fixture
def zstack_plate_dir(test_dir, microscope_config, test_params):
    """Create synthetic Z-stack plate data for the specified microscope type."""
    plate_dir = test_dir / "zstack_plate"

    # Get parameters from test_params with defaults if not specified
    grid_size = test_params.get("grid_size", (3, 3))
    tile_size = test_params.get("tile_size", (128, 128))
    overlap_percent = test_params.get("overlap_percent", 10)
    wavelengths = test_params.get("wavelengths", 2)
    cell_size_range = test_params.get("cell_size_range", (5, 10))
    wells = test_params.get("wells", ['A01'])  # Add this line

    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=grid_size,
        tile_size=tile_size,
        overlap_percent=overlap_percent,
        wavelengths=wavelengths,
        z_stack_levels=5,  # Always use 5 z-stack levels for this fixture
        cell_size_range=cell_size_range,
        wells=wells,  # Add this line
        format=microscope_config["format"],
        auto_image_size=microscope_config["auto_image_size"]
    )
    generator.generate_dataset()

    # ... rest of the function
```

## 4. Testing Plan

1. Update the fixtures to include the `wells` parameter
2. Run the tests to verify that the changes work correctly
3. Verify that the generated data includes the specified wells

## 5. Implementation Steps

1. Update the `flat_plate_dir` fixture to extract and pass the `wells` parameter
2. Update the `zstack_plate_dir` fixture to extract and pass the `wells` parameter
3. Run the tests to verify that the changes work correctly

## 6. Potential Risks and Mitigations

**Risk**: The changes might break existing tests that rely on the current behavior.
**Mitigation**: Run all tests after making the changes to ensure they still pass.

**Risk**: The `wells` parameter might not be properly handled by the `SyntheticMicroscopyGenerator` class.
**Mitigation**: Verify that the `SyntheticMicroscopyGenerator` class correctly handles the `wells` parameter.

## 7. Completion Summary

Date: 2023-07-15

The implementation has been completed successfully. The following changes were made:

1. Added the `wells` parameter extraction in the `flat_plate_dir` fixture
2. Added the `wells` parameter extraction in the `zstack_plate_dir` fixture
3. Updated both fixtures to pass the `wells` parameter to the `SyntheticMicroscopyGenerator` constructor
4. Verified that the changes work correctly by running the tests

The tests now correctly generate data for all wells specified in the `syn_data_params` dictionary. The output of the test run shows that both wells (A01 and B02) are being generated for both microscope types:

```
Wells: A01, B02
...
Generating data for well A01...
...
Generating data for well B02...
```

This fix ensures that all parameters in the `syn_data_params` dictionary are correctly passed to the `SyntheticMicroscopyGenerator` constructor, making the tests more flexible and easier to customize.
