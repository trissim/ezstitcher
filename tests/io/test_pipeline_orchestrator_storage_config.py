@pytest.mark.skip(reason="Part of IO module cleanup - awaiting approval")
def test_initialize_storage_adapter_normalizes_mode():
    """Test that _initialize_storage_adapter normalizes storage mode to lowercase."""
    # Create mocks
    mock_file_manager = MagicMock()
    mock_select_storage = MagicMock()
    mock_storage_adapter = MagicMock()
    mock_select_storage.return_value = mock_storage_adapter
    
    # Create a PipelineOrchestrator with mixed case storage mode
    orchestrator = PipelineOrchestrator(
        plate_path="dummy/plate",
        storage_mode="MeMoRy",  # Mixed case
        overlay_mode="auto"
    )
    
    # Set file_manager and mock select_storage
    orchestrator.file_manager = mock_file_manager
    orchestrator._ensure_file_manager = MagicMock(return_value=mock_file_manager)
    
    # Patch select_storage
    with patch('ezstitcher.core.pipeline_orchestrator.select_storage', mock_select_storage):
        # Call _initialize_storage_adapter
        orchestrator._initialize_storage_adapter()
    
    # Check that select_storage was called with normalized mode
    mock_select_storage.assert_called_once()
    args, kwargs = mock_select_storage.call_args
    
    # Check that mode was normalized to lowercase
    assert kwargs['mode'] == "memory"
    
    # Check that storage_config has normalized mode
    storage_config = kwargs['storage_config']
    assert storage_config.storage_mode == "memory"

@pytest.mark.skip(reason="Part of IO module cleanup - awaiting approval")
def test_initialize_storage_adapter_handles_enum():
    """Test that _initialize_storage_adapter handles Enum values for storage mode."""
    # Create mocks
    mock_file_manager = MagicMock()
    mock_select_storage = MagicMock()
    mock_storage_adapter = MagicMock()
    mock_select_storage.return_value = mock_storage_adapter
    
    # Create a mock Enum for storage mode
    class StorageModeEnum:
        def __str__(self):
            return "ZARR"
    
    # Create a PipelineOrchestrator with Enum storage mode
    orchestrator = PipelineOrchestrator(
        plate_path="dummy/plate",
        storage_mode=StorageModeEnum(),  # Enum that stringifies to "ZARR"
        overlay_mode="auto"
    )
    
    # Set file_manager and mock select_storage
    orchestrator.file_manager = mock_file_manager
    orchestrator._ensure_file_manager = MagicMock(return_value=mock_file_manager)
    
    # Patch select_storage
    with patch('ezstitcher.core.pipeline_orchestrator.select_storage', mock_select_storage):
        # Call _initialize_storage_adapter
        orchestrator._initialize_storage_adapter()
    
    # Check that select_storage was called with normalized mode
    mock_select_storage.assert_called_once()
    args, kwargs = mock_select_storage.call_args
    
    # Check that mode was normalized to lowercase
    assert kwargs['mode'] == "zarr"
    
    # Check that storage_config has normalized mode
    storage_config = kwargs['storage_config']
    assert storage_config.storage_mode == "zarr"

@pytest.mark.skip(reason="Part of IO module cleanup - awaiting approval")
def test_initialize_storage_adapter_preserves_original_mode():
    """Test that _initialize_storage_adapter preserves the original storage_mode attribute."""
    # Create mocks
    mock_file_manager = MagicMock()
    mock_select_storage = MagicMock()
    mock_storage_adapter = MagicMock()
    mock_select_storage.return_value = mock_storage_adapter
    
    # Create a PipelineOrchestrator with mixed case storage mode
    original_mode = "MeMoRy"  # Mixed case
    orchestrator = PipelineOrchestrator(
        plate_path="dummy/plate",
        storage_mode=original_mode,
        overlay_mode="auto"
    )
    
    # Set file_manager and mock select_storage
    orchestrator.file_manager = mock_file_manager
    orchestrator._ensure_file_manager = MagicMock(return_value=mock_file_manager)
    
    # Patch select_storage
    with patch('ezstitcher.core.pipeline_orchestrator.select_storage', mock_select_storage):
        # Call _initialize_storage_adapter
        orchestrator._initialize_storage_adapter()
    
    # Check that the original storage_mode attribute is preserved
    assert orchestrator.storage_mode == original_mode
    
    # Check that select_storage was called with normalized mode
    mock_select_storage.assert_called_once()
    args, kwargs = mock_select_storage.call_args
    assert kwargs['mode'] == "memory"
