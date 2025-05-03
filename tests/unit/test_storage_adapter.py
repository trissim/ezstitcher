import pytest
import numpy as np
from pathlib import Path
import shutil
import zarr # Ensure zarr is imported if used directly in tests, though factory is preferred

# Import the classes and factory function to be tested
from ezstitcher.io.storage_adapter import (
    StorageAdapter,
    MemoryStorageAdapter,
    ZarrStorageAdapter,
    select_storage,
    resolve_persist_path,
    generate_storage_key,
)

# --- Fixtures ---

@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for storage adapters that need disk space."""
    storage_dir = tmp_path / "storage_test"
    storage_dir.mkdir()
    yield storage_dir
    # Teardown: remove the directory after test completes
    # Use shutil.rmtree for robustness, especially for non-empty dirs like Zarr stores
    # shutil.rmtree(storage_dir, ignore_errors=True) # tmp_path fixture handles cleanup

@pytest.fixture
def memory_adapter() -> MemoryStorageAdapter:
    """Fixture for a MemoryStorageAdapter instance."""
    return MemoryStorageAdapter()

@pytest.fixture
def zarr_adapter(temp_storage_dir: Path) -> ZarrStorageAdapter:
    """Fixture for a ZarrStorageAdapter instance using a temporary directory."""
    return ZarrStorageAdapter(storage_root=temp_storage_dir)

@pytest.fixture(params=["memory", "zarr"])
def adapter(request, memory_adapter, zarr_adapter, temp_storage_dir) -> StorageAdapter:
    """Parametrized fixture providing both memory and zarr adapters."""
    if request.param == "memory":
        return memory_adapter
    elif request.param == "zarr":
        # Need to return the instance created with temp_storage_dir
        return zarr_adapter
    else:
        raise ValueError(f"Unknown adapter type requested: {request.param}")

# --- Test Data ---

@pytest.fixture
def sample_data() -> dict[str, np.ndarray]:
    """Sample data for testing."""
    return {
        "array1": np.arange(10),
        "array2": np.random.rand(3, 4),
        "array3": np.array([1, 2, 3], dtype=np.int16),
    }

# --- Test Cases ---

def test_write_read_exists(adapter: StorageAdapter, sample_data: dict[str, np.ndarray]):
    """Test writing, checking existence, and reading back data."""
    key, data = list(sample_data.items())[0]

    assert not adapter.exists(key), "Key should not exist initially"

    adapter.write(key, data)
    assert adapter.exists(key), "Key should exist after writing"

    read_data = adapter.read(key)
    np.testing.assert_array_equal(read_data, data)
    assert read_data.dtype == data.dtype

    # Test reading non-existent key raises KeyError
    with pytest.raises(KeyError):
        adapter.read("non_existent_key")

def test_list_keys(adapter: StorageAdapter, sample_data: dict[str, np.ndarray]):
    """Test listing keys."""
    assert adapter.list_keys() == [], "Should have no keys initially"

    for key, data in sample_data.items():
        adapter.write(key, data)

    keys = adapter.list_keys()
    assert sorted(keys) == sorted(sample_data.keys())

def test_delete(adapter: StorageAdapter, sample_data: dict[str, np.ndarray]):
    """Test deleting keys."""
    key_to_delete = "array1"
    other_key = "array2"
    data_to_delete = sample_data[key_to_delete]
    other_data = sample_data[other_key]

    adapter.write(key_to_delete, data_to_delete)
    adapter.write(other_key, other_data)

    assert adapter.exists(key_to_delete)
    assert adapter.exists(other_key)
    assert sorted(adapter.list_keys()) == sorted([key_to_delete, other_key])

    adapter.delete(key_to_delete)

    assert not adapter.exists(key_to_delete), "Key should not exist after deletion"
    assert adapter.exists(other_key), "Other key should still exist"
    assert adapter.list_keys() == [other_key]

    # Test reading deleted key raises KeyError
    with pytest.raises(KeyError):
        adapter.read(key_to_delete)

    # Test deleting non-existent key (should not raise error)
    try:
        adapter.delete("non_existent_key_again")
    except Exception as e:
        pytest.fail(f"Deleting non-existent key raised an exception: {e}")

def test_overwrite(adapter: StorageAdapter):
    """Test overwriting an existing key."""
    key = "test_key"
    initial_data = np.array([1, 2, 3])
    overwrite_data = np.array([4, 5, 6, 7])

    adapter.write(key, initial_data)
    read_initial = adapter.read(key)
    np.testing.assert_array_equal(read_initial, initial_data)

    adapter.write(key, overwrite_data)
    read_overwritten = adapter.read(key)
    np.testing.assert_array_equal(read_overwritten, overwrite_data)
    assert read_overwritten.dtype == overwrite_data.dtype

# --- Adapter-Specific Tests ---

def test_memory_persist(memory_adapter: MemoryStorageAdapter, sample_data: dict[str, np.ndarray], temp_storage_dir: Path):
    """Test the persist method of MemoryStorageAdapter."""
    persist_dir = temp_storage_dir / "persist_output"

    # Write data
    for key, data in sample_data.items():
        memory_adapter.write(key, data)

    # Persist
    memory_adapter.persist(persist_dir)

    # Verify files
    assert persist_dir.is_dir()
    for key, original_data in sample_data.items():
        expected_file = persist_dir / f"{key}.npy"
        assert expected_file.is_file(), f"File {expected_file} not found after persist"
        loaded_data = np.load(expected_file)
        np.testing.assert_array_equal(loaded_data, original_data)
        assert loaded_data.dtype == original_data.dtype

def test_zarr_persist(zarr_adapter: ZarrStorageAdapter, sample_data: dict[str, np.ndarray], temp_storage_dir: Path):
    """Test the persist method of ZarrStorageAdapter (should be a no-op)."""
    # Write data (which persists immediately for Zarr)
    key, data = list(sample_data.items())[0]
    zarr_adapter.write(key, data)

    # Call persist with a different directory - it should be ignored
    other_dir = temp_storage_dir / "other_persist_output"
    try:
        zarr_adapter.persist(other_dir)
    except Exception as e:
        pytest.fail(f"Zarr persist raised an exception: {e}")

    # Check that the original Zarr store still exists and the other dir wasn't created by persist
    assert zarr_adapter.zarr_path.exists()
    assert not other_dir.exists() # persist shouldn't create this

# --- Factory Tests ---

def test_select_storage_memory(temp_storage_dir: Path):
    """Test factory function for 'memory' mode."""
    from ezstitcher.io.storage_config import StorageConfig
    storage_config = StorageConfig()
    adapter = select_storage(mode="memory", storage_config=storage_config, storage_root=temp_storage_dir)
    assert isinstance(adapter, MemoryStorageAdapter)

def test_select_storage_zarr(temp_storage_dir: Path):
    """Test factory function for 'zarr' mode."""
    from ezstitcher.io.storage_config import StorageConfig
    storage_config = StorageConfig()
    adapter = select_storage(mode="zarr", storage_config=storage_config, storage_root=temp_storage_dir)
    assert isinstance(adapter, ZarrStorageAdapter)
    # Check if the path was correctly passed and the store initialized
    assert adapter.storage_root == temp_storage_dir
    assert adapter.zarr_path.exists()

def test_select_storage_invalid(temp_storage_dir: Path):
    """Test factory function with an invalid mode."""
    from ezstitcher.io.storage_config import StorageConfig
    storage_config = StorageConfig()
    with pytest.raises(ValueError, match="Invalid storage mode"):
        select_storage(mode="invalid_mode", storage_config=storage_config, storage_root=temp_storage_dir)

# --- Key Generation Tests ---

def test_generate_storage_key():
    """Test that generate_storage_key properly normalizes step names."""
    # Test that "Test Step" becomes "test_step" in the key
    key = generate_storage_key("Test Step", "A01", "channel_1")
    assert "test_step" in key, f"Expected 'test_step' in key, got '{key}'"
    assert key == "test_step_A01_channel_1", f"Expected 'test_step_A01_channel_1', got '{key}'"

    # Test with extra whitespace
    key = generate_storage_key("  Test Step  ", "A01", "channel_1")
    assert "test_step" in key, f"Expected 'test_step' in key, got '{key}'"
    assert key == "test_step_A01_channel_1", f"Expected 'test_step_A01_channel_1', got '{key}'"

    # Test with different case
    key = generate_storage_key("TEST STEP", "A01", "channel_1")
    assert "test_step" in key, f"Expected 'test_step' in key, got '{key}'"
    assert key == "test_step_A01_channel_1", f"Expected 'test_step_A01_channel_1', got '{key}'"

    # Test with no well or component
    key = generate_storage_key("Test Step")
    assert key == "test_step", f"Expected 'test_step', got '{key}'"

# --- Persist Path Resolution Tests ---

def test_resolve_persist_path():
    """Test the resolve_persist_path function with different modes and paths."""
    workspace = Path("/workspace")
    storage_root = Path("/storage_root")

    # Test memory mode with both paths
    path = resolve_persist_path("memory", workspace, storage_root)
    assert path == storage_root

    # Test memory mode with only workspace
    path = resolve_persist_path("memory", workspace)
    assert path == workspace / "adapter_output" / "memory"

    # Test memory mode with no paths
    with pytest.raises(ValueError, match="Cannot resolve persist path for memory mode"):
        resolve_persist_path("memory")

    # Test zarr mode (should return None)
    path = resolve_persist_path("zarr", workspace, storage_root)
    assert path is None

    # Test legacy mode (should return None)
    path = resolve_persist_path("legacy", workspace, storage_root)
    assert path is None

    # Test invalid mode
    with pytest.raises(ValueError, match="Unknown storage mode"):
        resolve_persist_path("invalid_mode", workspace, storage_root)
