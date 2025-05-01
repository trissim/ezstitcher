import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from ezstitcher.core.steps import ImageStitchingStep, PositionGenerationStep
from ezstitcher.core.pipeline import Pipeline, Step, ProcessingContext
from ezstitcher.io.storage_adapter import StorageAdapter # Added
from pathlib import Path

# Unit tests for the pipeline module

def test_step_path_resolution():
    """Test that paths are correctly resolved through the context."""

    # Create a pipeline with multiple steps
    pipeline = Pipeline(
        steps=[
            Step(name="Step 1", func=lambda x: x),
            PositionGenerationStep(),
            ImageStitchingStep()
        ]
    )

    # Create a mock context with path overrides
    step1_id = id(pipeline.steps[0])
    step2_id = id(pipeline.steps[1])
    step3_id = id(pipeline.steps[2])

    # Create a context with path overrides
    context = ProcessingContext()

    # Add step plans to the context
    from ezstitcher.core.pipeline import StepExecutionPlan

    # Step 1 plan
    context.add_step_plan(
        pipeline.steps[0],
        StepExecutionPlan(
            step_id=step1_id,
            step_name="Step 1",
            step_type="Step",
            input_dir=Path("input_dir"),
            output_dir=Path("step1_output")
        )
    )

    # Step 2 plan (PositionGenerationStep)
    context.add_step_plan(
        pipeline.steps[1],
        StepExecutionPlan(
            step_id=step2_id,
            step_name="Position Generation",
            step_type="PositionGenerationStep",
            input_dir=Path("step1_output"),
            output_dir=Path("positions_dir")
        )
    )

    # Step 3 plan (ImageStitchingStep)
    context.add_step_plan(
        pipeline.steps[2],
        StepExecutionPlan(
            step_id=step3_id,
            step_name="Image Stitching",
            step_type="ImageStitchingStep",
            input_dir=Path("positions_dir"),
            output_dir=Path("output_dir")
        )
    )

    # Verify that paths are correctly resolved through the context
    assert context.get_step_input_dir(pipeline.steps[0]) == Path("input_dir")
    assert context.get_step_output_dir(pipeline.steps[0]) == Path("step1_output")
    assert context.get_step_input_dir(pipeline.steps[1]) == Path("step1_output")
    assert context.get_step_output_dir(pipeline.steps[1]) == Path("positions_dir")
    assert context.get_step_input_dir(pipeline.steps[2]) == Path("positions_dir")
    assert context.get_step_output_dir(pipeline.steps[2]) == Path("output_dir")


# --- Fixtures for Adapter Test ---

@pytest.fixture
def mock_storage_adapter() -> MagicMock:
    """Provides a mock StorageAdapter."""
    mock = MagicMock(spec=StorageAdapter)
    mock.write = MagicMock() # Ensure write method is mockable
    return mock

@pytest.fixture
def mock_orchestrator_with_adapter(mock_storage_adapter) -> MagicMock:
    """Provides a mock PipelineOrchestrator holding a mock StorageAdapter."""
    mock = MagicMock()
    mock.storage_adapter = mock_storage_adapter
    return mock

@pytest.fixture
def mock_context_with_adapter(mock_orchestrator_with_adapter) -> MagicMock:
    """Provides a mock ProcessingContext linked to an orchestrator with an adapter."""
    # Use a real context instance but mock the orchestrator
    context = ProcessingContext(orchestrator=mock_orchestrator_with_adapter)
    # Add dummy step plans if needed by steps, though not strictly necessary for this test
    return context

# --- Test for Adapter Writing ---

def test_pipeline_run_writes_numpy_to_adapter(mock_context_with_adapter, mock_storage_adapter):
    """Test that Pipeline.run writes numpy array outputs to the storage adapter."""
    # Create a mock step that returns numpy array and other data types
    mock_step = MagicMock(spec=Step)
    mock_step.name = "TestStepWithOutput"
    step_output_data = np.array([1, 2, 3])
    step_outputs = {
        "numpy_result": step_output_data,
        "string_result": "hello",
        "list_result": [4, 5, 6]
    }
    mock_step.process.return_value = step_outputs

    # Create pipeline
    pipeline = Pipeline(steps=[mock_step], name="AdapterTestPipeline")

    # Run the pipeline
    pipeline.run(context=mock_context_with_adapter)

    # Assertions
    mock_step.process.assert_called_once_with(mock_context_with_adapter)

    # Check that adapter.write was called ONLY for the numpy array
    expected_key = "AdapterTestPipeline_teststepwithoutput_0_numpy_result" # Based on key generation logic
    # Use assert_called_once_with, checking the key and that the data is the numpy array
    # Comparing numpy arrays directly in mock calls can be tricky, check args manually if needed
    assert mock_storage_adapter.write.call_count == 1
    call_args, call_kwargs = mock_storage_adapter.write.call_args
    assert call_args[0] == expected_key
    np.testing.assert_array_equal(call_args[1], step_output_data)


def test_pipeline_run_does_not_write_if_no_adapter(mock_context_with_adapter, mock_storage_adapter):
    """Test that Pipeline.run does not attempt to write if no adapter is present."""
    # Remove the adapter from the mock orchestrator in the context
    mock_context_with_adapter.orchestrator.storage_adapter = None

    # Create a mock step
    mock_step = MagicMock(spec=Step)
    mock_step.name = "TestStepNoAdapter"
    step_output_data = np.array([1, 2, 3])
    step_outputs = {"numpy_result": step_output_data}
    mock_step.process.return_value = step_outputs

    # Create pipeline
    pipeline = Pipeline(steps=[mock_step], name="NoAdapterTestPipeline")

    # Run the pipeline
    pipeline.run(context=mock_context_with_adapter)

    # Assertions
    mock_step.process.assert_called_once_with(mock_context_with_adapter)
    # Check that adapter.write was NOT called
    mock_storage_adapter.write.assert_not_called()


def test_pipeline_run_handles_non_dict_step_output(mock_context_with_adapter, mock_storage_adapter):
    """Test Pipeline.run handles steps not returning a dict gracefully."""
    # Create a mock step that returns something other than a dict
    mock_step = MagicMock(spec=Step)
    mock_step.name = "TestStepBadOutput"
    mock_step.process.return_value = "not a dictionary" # Bad return type

    # Create pipeline
    pipeline = Pipeline(steps=[mock_step], name="BadOutputPipeline")

    # Run the pipeline - should log warning but not crash
    try:
        pipeline.run(context=mock_context_with_adapter)
    except Exception as e:
        pytest.fail(f"Pipeline.run raised an unexpected exception for non-dict output: {e}")

    # Assertions
    mock_step.process.assert_called_once_with(mock_context_with_adapter)
    # Check that adapter.write was NOT called
    mock_storage_adapter.write.assert_not_called()