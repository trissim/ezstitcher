# Test Output Standardization Plan

Status: In Progress  
Progress: 0%  
Last Updated: 2023-07-11  
Dependencies: None

## 1. Problem Analysis

The current test files (`test_synthetic_imagexpress_refactored_auto.py` and `test_synthetic_opera_phenix_refactored_auto.py`) have different approaches to managing test data:

1. `test_synthetic_imagexpress_refactored_auto.py` uses pytest's `tmp_path_factory` to create temporary directories
2. `test_synthetic_opera_phenix_refactored_auto.py` uses a fixed path at `tests/test_data/opera_phenix_synthetic_refactored`

We need to standardize both files to:
- Output test data to `/tests/tests_data/`
- Create a folder for each test file
- Create a subfolder for each test method
- Maintain the same test functionality

## 2. High-Level Solution

1. **Create a common directory structure**:
   - `/tests/tests_data/{test_file_name}/{test_method_name}/`

2. **Modify the ImageXpress test file**:
   - Replace `tmp_path_factory` with a fixed path
   - Create a folder for each test method
   - Update fixtures to use the new directory structure

3. **Modify the Opera Phenix test file**:
   - Update the base directory to use `/tests/tests_data/`
   - Ensure each test method has its own subfolder
   - Maintain the same test functionality

## 3. Implementation Details

### 3.1 Create Common Directory Structure

```python
# Common function to create test directories
def create_test_dir(test_file_name, test_method_name):
    """Create a standardized test directory structure."""
    base_dir = Path(__file__).parent / "tests_data" / test_file_name
    test_dir = base_dir / test_method_name
    
    # Create the directory if it doesn't exist
    test_dir.mkdir(parents=True, exist_ok=True)
    
    return test_dir
```

### 3.2 Modify ImageXpress Test File

```python
# In test_synthetic_imagexpress_refactored_auto.py
import os
import shutil
import pytest
from pathlib import Path
from ezstitcher.core.main import process_plate_auto
from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator

@pytest.fixture(scope="module")
def base_test_dir():
    """Create base test directory for ImageXpress tests."""
    base_dir = Path(__file__).parent / "tests_data" / "imagexpress_refactored_auto"
    
    # Create the directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    
    yield base_dir
    
    # Uncomment to clean up after tests
    # shutil.rmtree(base_dir)

@pytest.fixture
def test_dir(base_test_dir, request):
    """Create test-specific directory."""
    test_name = request.node.name
    test_dir = base_test_dir / test_name
    
    # Create the directory if it doesn't exist
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSetting up test directory for {test_name}: {test_dir}")
    
    return test_dir

@pytest.fixture
def flat_plate_dir(test_dir):
    """Create synthetic flat plate data."""
    plate_dir = test_dir / "flat_plate"
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=(2, 2),
        image_size=(256, 256),
        tile_size=(128, 128),
        overlap_percent=10,
        wavelengths=2,
        z_stack_levels=1,
        format="ImageXpress"
    )
    generator.generate_dataset()
    
    # Create a copy of the original data for inspection
    original_dir = test_dir / "flat_plate_original"
    if not original_dir.exists():
        shutil.copytree(plate_dir, original_dir)
    
    return plate_dir

@pytest.fixture
def zstack_plate_dir(test_dir):
    """Create synthetic Z-stack plate data."""
    plate_dir = test_dir / "zstack_plate"
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(plate_dir),
        grid_size=(2, 2),
        image_size=(256, 256),
        tile_size=(128, 128),
        overlap_percent=10,
        wavelengths=2,
        z_stack_levels=5,
        format="ImageXpress"
    )
    generator.generate_dataset()
    
    # Create a copy of the original data for inspection
    original_dir = test_dir / "zstack_plate_original"
    if not original_dir.exists():
        shutil.copytree(plate_dir, original_dir)
    
    return plate_dir

def test_flat_plate_minimal(flat_plate_dir):
    """Test processing a flat plate with minimal configuration."""
    success = process_plate_auto(
        flat_plate_dir,
        microscope_type="ImageXpress"
    )
    assert success, "Flat plate processing failed"

def test_zstack_projection_minimal(zstack_plate_dir):
    """Test processing a Z-stack plate with projection."""
    success = process_plate_auto(
        zstack_plate_dir,
        microscope_type="ImageXpress",
        **{"z_stack_processor.create_projections": True}
    )
    assert success, "Z-stack projection processing failed"

def test_zstack_per_plane_minimal(zstack_plate_dir):
    """Test processing a Z-stack plate with per-plane stitching."""
    success = process_plate_auto(
        zstack_plate_dir,
        microscope_type="ImageXpress",
        **{"z_stack_processor.stitch_all_z_planes": True}
    )
    assert success, "Z-stack per-plane processing failed"

def test_multi_channel_minimal(flat_plate_dir):
    """Test processing a flat plate with multiple reference channels."""
    success = process_plate_auto(
        flat_plate_dir,
        microscope_type="ImageXpress",
        **{"reference_channels": ["1", "2"]}
    )
    assert success, "Multi-channel reference processing failed"
```

### 3.3 Modify Opera Phenix Test File

For the Opera Phenix test file, we need to update the `base_test_dir` fixture:

```python
@pytest.fixture(scope="class")
def base_test_dir(self):
    """Set up base test directory."""
    # Create base test data directory
    base_dir = Path(__file__).parent / "tests_data" / "opera_phenix_refactored_auto"

    # Create the base test data directory
    base_dir.mkdir(parents=True, exist_ok=True)

    yield base_dir

    # Clean up after all tests
    # Uncomment the following line to clean up after tests
    # shutil.rmtree(base_dir)
```

## 4. Validation

### 4.1 Unit Tests

1. Run the modified ImageXpress tests to verify they work with the new directory structure
2. Run the modified Opera Phenix tests to verify they work with the new directory structure
3. Verify that the test data is correctly organized in the `/tests/tests_data/` directory

### 4.2 Integration Tests

1. Run all tests to verify that the changes don't break existing functionality
2. Verify that the test data is correctly organized in the `/tests/tests_data/` directory

## 5. Implementation Order

1. Create the `/tests/tests_data/` directory if it doesn't exist
2. Modify the ImageXpress test file
3. Modify the Opera Phenix test file
4. Run the tests to verify the changes

## 6. Benefits

1. **Standardized directory structure**: All test data is organized in a consistent way
2. **Improved test isolation**: Each test method has its own directory
3. **Better test data management**: Test data is stored in a fixed location for easier inspection
4. **Consistent approach**: Both test files use the same approach to managing test data

## 7. Risks and Mitigations

1. **Risk**: Changes might break existing tests
   **Mitigation**: Run tests after each change to verify functionality

2. **Risk**: Disk space usage might increase
   **Mitigation**: Add cleanup code to remove test data after tests complete

3. **Risk**: Test data might be accidentally committed to version control
   **Mitigation**: Add `/tests/tests_data/` to `.gitignore`

## 8. References

- `tests/test_synthetic_imagexpress_refactored_auto.py`
- `tests/test_synthetic_opera_phenix_refactored_auto.py`
