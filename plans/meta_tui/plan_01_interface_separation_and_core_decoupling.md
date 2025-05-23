# Plan 01: TUI-Core Interface Separation and Decoupling

**Version**: 1.0
**Date**: 2025-05-23
**Author**: MasterMind Architect

## 1. Introduction & Goal

**Problem**: The `openhcs.tui` package currently exhibits high coupling with the `openhcs.core` modules. UI components and command logic in modules like `commands.py`, `menu_bar.py`, and `tui_architecture.py` directly import and interact with core classes such as `PipelineOrchestrator`, `AbstractStep`, `FunctionStep`, and `FUNC_REGISTRY`. This tight coupling violates fundamental architectural principles, leading to a brittle system where changes in the core can necessitate widespread changes in the TUI, and vice-versa. It also makes testing UI components in isolation difficult.

**Goal**: To establish a clear, well-defined boundary between the TUI (presentation and interaction mechanism) and the Core (business logic and data processing policy). This will be achieved by:
    1. Defining formal API contracts (Python `Protocol`s) for all TUI-to-Core interactions.
    2. Implementing an Adapter layer within the TUI (`openhcs.tui.core_adapters`) that conforms to these interfaces and mediates all communication with the Core.
    3. Refactoring existing TUI modules to use these adapters and interfaces, removing direct dependencies on Core implementation details.

**Architectural Principles**:
*   **Information Hiding**: Core implementation details will be hidden from the TUI.
*   **Law of Demeter (Principle of Least Knowledge)**: TUI components will only talk to their "immediate friends" (the adapter interfaces).
*   **Separation of Concerns**: UI logic will be distinctly separated from core processing logic.
*   **Dependency Inversion Principle**: TUI modules will depend on abstractions (interfaces/protocols), not on concrete core implementations.

## 2. Proposed Interfaces (to be created in `openhcs/tui/interfaces.py`)

This new file will house all protocols defining the contract between the TUI and its abstraction of the core.

```python
# openhcs/tui/interfaces.py
from typing import Protocol, Any, List, Optional, Dict, Coroutine, Union, Tuple, TYPE_CHECKING
from pathlib import Path

# Forward declare for type hints if needed, or use actual types if importable without circularity
if TYPE_CHECKING:
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.io.filemanager import FileManager
    # Add other core types that might appear in interface signatures if they don't create import cycles.
    # For data structures, prefer generic Dicts or Pydantic models if complex.


class CoreStepData(Protocol):
    """Data representation of a pipeline step for TUI consumption. Should be serializable (e.g. via pickling)."""
    uid: str
    name: str
    step_type: str # e.g., "FunctionStep", "CompositeStep"
    func_display_name: Optional[str] # Name of the function/pattern
    params: Dict[str, Any]
    status: str # e.g., "new", "configured", "error_config", "error_validation"
    is_enabled: bool
    # Potentially other TUI-relevant metadata like description, tags

    def to_dict(self) -> Dict[str, Any]: ... # For potential serialization if not directly pickleable


class CorePlateData(Protocol):
    """Data representation of a plate/orchestrator for TUI consumption."""
    id: str # Unique identifier for the plate/orchestrator
    name: str # Display name, often derived from path
    path: str # Filesystem path to the plate data
    status: str # TUI-facing status string, e.g., "!" (new), "?" (initialized), "✔︎" (compiled_ok), "✗c" (compile_error), "✗r" (run_error). Adapter maps core status to these.
    backend_name: Optional[str]
    pipeline_definition_summary: Optional[str] # e.g., "5 steps" or list of step names


class CoreOrchestratorAdapterInterface(Protocol):
    """
    Interface for TUI interactions related to a single plate/pipeline orchestrator.
    Methods should be asynchronous if they involve I/O or potentially long-running core operations.
    Return types often include a Tuple with success status and an optional error message.
    """

    async def get_plate_data(self) -> CorePlateData: ...
    async def get_orchestrator_config_dict(self) -> Dict[str, Any]: ... # Returns a dict representation of the orchestrator's current config.
    async def update_orchestrator_config(self, config_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]: ... # Takes full config data, returns success/error.

    async def initialize(self) -> Tuple[bool, Optional[str]]: ... # Returns success status and optional error message.
    async def get_pipeline_steps(self) -> List[CoreStepData]: ... # Gets the current pipeline steps as TUI-consumable data.

    async def add_new_step_to_pipeline(self, step_configuration: Dict[str, Any]) -> Tuple[Optional[CoreStepData], Optional[str]]: ...
    # Takes a dictionary with step configuration (e.g., name, type, func_identifier for FunctionStep).
    # Returns the created CoreStepData or an error message.

    async def add_default_step_to_pipeline(self, default_step_type: str = "FunctionStep") -> Tuple[Optional[CoreStepData], Optional[str]]: ...
    # Adds a step with default parameters of the specified type.

    async def update_pipeline_step(self, step_uid: str, updated_step_data: Dict[str, Any]) -> Tuple[Optional[CoreStepData], Optional[str]]: ...
    # Takes the UID of the step to update and a dictionary with the new data for the step.
    # Returns the updated CoreStepData or an error message.

    async def remove_step(self, step_uid: str) -> bool: ... # Returns True if successful.
    async def move_step(self, step_uid: str, direction: str) -> bool: ... # direction: "up" or "down". Returns True if successful.

    async def save_pipeline_to_file(self, file_path: Path, pipeline_steps: List[CoreStepData]) -> Tuple[bool, Optional[str]]: ...
    # Adapter handles conversion of CoreStepData list to a savable format (e.g., list of AbstractStep, then pickling).

    async def load_pipeline_from_file(self, file_path: Path) -> Tuple[Optional[List[CoreStepData]], Optional[str]]: ...
    # Adapter handles unpickling and conversion of loaded core steps back to CoreStepData list.

    async def compile_orchestrator_pipeline(self, pipeline_steps: List[CoreStepData]) -> Tuple[bool, Optional[str]]: ...
    # Takes the current list of TUI step data, adapter converts to core steps and compiles.

    async def execute_orchestrator_pipeline(self) -> Tuple[bool, Optional[str]]: ...
    # Executes the last successfully compiled pipeline for this orchestrator.

    async def get_step_definition_details(self, step_type_name: str, existing_params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]: ...
    # Returns constructor parameters (name, type, default value, annotation) for the given AbstractStep subclass (e.g., "FunctionStep").
    # Optionally populates with existing_params if editing a step. Uses Python's `inspect` module.


class CoreApplicationAdapterInterface(Protocol):
    """
    Interface for TUI interactions related to global application state and general core functionalities.
    """

    async def get_global_config(self) -> 'GlobalPipelineConfig': ... # Use actual type if safe, returns the actual Pydantic model instance.
    async def update_global_config_dict(self, config_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]: ...
    # Updates the global configuration from a dictionary, returns success/error.

    async def get_available_plates(self) -> List[CorePlateData]: ...

    async def add_new_plate_orchestrator(self, folder_path: str) -> Tuple[Optional[str], Optional[str]]: ...
    # Creates a new plate/orchestrator for the given folder_path. Returns (plate_id, error_message).
    # The new orchestrator uses default global config settings.

    async def add_new_plate_orchestrators(self, folder_paths: List[str]) -> List[Tuple[Optional[str], Optional[str]]]: ...
    # Batch version of add_new_plate_orchestrator.

    async def remove_plate(self, plate_id: str) -> bool: ... # Returns True if successful.
    async def get_orchestrator_adapter(self, plate_id: str) -> Optional[CoreOrchestratorAdapterInterface]: ...

    async def get_file_manager(self) -> 'FileManager': ... # Use actual type if safe

    async def list_directory_contents(self, path: str, backend_name: str = "disk") -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]: ...
    # Returns list of {'name': str, 'is_dir': bool, 'path': str} or error. Uses FileManager.

    async def get_available_backends_for_func_registry(self) -> List[str]: ... # Returns list of backend names (e.g. "core", "custom").

    async def get_functions_for_backend(self, backend_name: str) -> List[Dict[str, Any]]: ...
    # Returns a list of dictionaries, each representing a function: {'name': str, 'doc': Optional[str]}.

    async def get_function_signature(self, backend_name: str, func_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]: ...
    # Returns parameters with defaults/types for a registered function using `inspect.signature`.
    # Format: {'param_name': {'default': ..., 'annotation': ...}, ...} or error.

    async def validate_function_pattern(self, pattern: Union[List, Dict]) -> bool: ... # Remains for quick validation if needed by TUI.

    async def save_step_to_file(self, step_data: CoreStepData, file_path: Path) -> Tuple[bool, Optional[str]]: ...
    # Adapter converts CoreStepData to a serializable AbstractStep form (if not already) and pickles it.

    async def load_step_from_file(self, file_path: Path) -> Tuple[Optional[CoreStepData], Optional[str]]: ...
    # Adapter unpickles AbstractStep and converts to CoreStepData.

    async def save_func_pattern_to_file(self, func_pattern: Any, file_path: Path) -> Tuple[bool, Optional[str]]: ...
    # Pickles a function pattern (typically a list or dict).

    async def load_func_pattern_from_file(self, file_path: Path) -> Tuple[Optional[Any], Optional[str]]: ...
    # Unpickles a function pattern.

    async def shutdown_core_services(self) -> None: ...
```

## 3. Adapter Implementation (New file: `openhcs/tui/core_adapters.py`)

This new module will contain the concrete adapter class(es).

```python
# openhcs/tui/core_adapters.py
import asyncio
from typing import Any, List, Optional, Dict, Coroutine, Union, Type
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.io.filemanager import FileManager
from openhcs.processing.func_registry import FUNC_REGISTRY
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.pipeline.funcstep_contract_validator import validate_pattern_structure # Example
from openhcs.constants.constants import Backend # Example

from .interfaces import (
    CoreApplicationAdapterInterface,
    CoreOrchestratorAdapterInterface,
    CoreStepData,
    CorePlateData
)

# Shared ThreadPoolExecutor for running synchronous core methods
# This can be defined here or passed in if a global executor is preferred.
SHARED_EXECUTOR = ThreadPoolExecutor(max_workers=5, thread_name_prefix="tui-core-adapter")

# Helper to convert core Step object to CoreStepData
def _core_step_to_tui_data(step: AbstractStep) -> CoreStepData:
    # This is a simplified conversion. Actual implementation will need more detail.
    # Step status needs to be determined/updated based on core step state.
    func_display_name = None
    if isinstance(step, FunctionStep):
        if isinstance(step.func_pattern, list) and step.func_pattern:
            first_call = step.func_pattern[0]
            if isinstance(first_call, dict) and 'func' in first_call:
                func_display_name = str(first_call['func'])
        elif isinstance(step.func_pattern, dict) and 'func' in step.func_pattern:
            func_display_name = str(step.func_pattern['func'])

    return { # type: ignore # Assuming CoreStepData is a TypedDict or similar
        "uid": step.uid,
        "name": step.name,
        "step_type": step.__class__.__name__,
        "func_display_name": func_display_name,
        "params": step.params.copy() if step.params else {},
        "status": getattr(step, 'tui_status', 'new'), # Core step should have a 'tui_status' or adapter maps it
        "is_enabled": getattr(step, 'is_enabled', True),
    }

# Helper to convert core Orchestrator state to CorePlateData
def _core_orchestrator_to_tui_plate_data(orchestrator: PipelineOrchestrator) -> CorePlateData:
    # The orchestrator's internal status (e.g., new, initialized, compiled_ok, error_compile, error_run)
    # needs to be mapped to the TUI-specific symbols (e.g., "!", "?", "✔︎", "✗c", "✗r").
    # This mapping logic will reside in the adapter or this helper.
    # Example mapping:
    core_status = getattr(orchestrator, 'status', 'unknown') # Orchestrator needs a robust status attribute
    tui_status_map = {
        'new': '!',
        'initialized': '?',
        'compiled_ok': '✔︎',
        'error_init': '✗i',
        'error_compile': '✗c',
        'error_run': '✗r',
        'run_completed': '✓', # Different check from compiled_ok
        'saved': '💾',
        'loaded': '📤',
        'unknown': '⁇'
    }
    tui_status = tui_status_map.get(core_status, '⁇')

    pipeline_def = orchestrator.pipeline_definition # This is List[AbstractStep]
    summary = f"{len(pipeline_def)} steps" if pipeline_def else "No pipeline"

    return { # type: ignore
        "id": orchestrator.plate_id,
        "name": orchestrator.plate_path.name, # Or a configured display name
        "path": str(orchestrator.plate_path),
        "status": tui_status,
        "backend_name": orchestrator.config.default_backend.value if orchestrator.config and orchestrator.config.default_backend else None,
        "pipeline_definition_summary": summary,
    }


class SingleOrchestratorAdapter(CoreOrchestratorAdapterInterface):
    # ... (initializer and _run_sync as before) ...
    # The adapter is responsible for managing the PipelineOrchestrator's status attribute
    # (e.g., self._orchestrator.status = "initialized") after operations.
    # It also manages the self._orchestrator.pipeline_definition (List[AbstractStep]),
    # converting to/from List[CoreStepData] for TUI interaction.

    # Example method stubs reflecting changes:
    async def initialize(self) -> Tuple[bool, Optional[str]]:
        # ... call self._orchestrator.initialize() ...
        # ... update self._orchestrator.status ...
        # ... return True, None or False, "Error message" ...
        pass

    async def get_orchestrator_config_dict(self) -> Dict[str, Any]:
        # ... return self._orchestrator.config.model_dump() ...
        pass

    async def update_orchestrator_config(self, config_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        # ... update self._orchestrator.config based on config_data ...
        # ... handle persistence if orchestrator config supports it ...
        # ... return success/error ...
        pass

    async def add_new_step_to_pipeline(self, step_configuration: Dict[str, Any]) -> Tuple[Optional[CoreStepData], Optional[str]]:
        # 1. Convert step_configuration into parameters for a new core AbstractStep (e.g., FunctionStep).
        #    This might involve looking up FUNC_REGISTRY for function patterns if type is FunctionStep.
        # 2. Create the core step instance.
        # 3. Add to self._orchestrator.pipeline_definition (the List[AbstractStep]).
        # 4. Convert the new core step to CoreStepData.
        # 5. Return (CoreStepData, None) or (None, "Error message").
        pass

    async def add_default_step_to_pipeline(self, default_step_type: str = "FunctionStep") -> Tuple[Optional[CoreStepData], Optional[str]]:
        # Similar to add_new_step_to_pipeline but with default configuration for the step_type.
        # E.g., create a FunctionStep with a placeholder name and no function selected.
        pass

    async def update_pipeline_step(self, step_uid: str, updated_step_data: Dict[str, Any]) -> Tuple[Optional[CoreStepData], Optional[str]]:
        # 1. Find the core AbstractStep in self._orchestrator.pipeline_definition by step_uid.
        # 2. Update its attributes based on updated_step_data. This is complex as `updated_step_data` is TUI-facing;
        #    careful mapping to core step attributes (e.g. `func_pattern` for FunctionStep) is needed.
        # 3. Convert the updated core step back to CoreStepData.
        # 4. Return (CoreStepData, None) or (None, "Error message").
        pass

    async def save_pipeline_to_file(self, file_path: Path, pipeline_steps: List[CoreStepData]) -> Tuple[bool, Optional[str]]:
        # 1. The adapter needs to convert `pipeline_steps: List[CoreStepData]` back into `List[AbstractStep]`.
        #    This is a critical step: it might involve re-creating AbstractStep instances based on CoreStepData.
        #    Alternatively, the adapter could maintain the List[AbstractStep] internally and `pipeline_steps` is just for TUI display sync.
        #    Assuming the adapter holds the canonical List[AbstractStep] (self._orchestrator.pipeline_definition).
        # 2. Use a utility (perhaps on FileManager or a dedicated serialization service) to pickle
        #    `self._orchestrator.pipeline_definition` to `file_path`.
        # 3. Return (True, None) or (False, "Error message").
        pass

    async def load_pipeline_from_file(self, file_path: Path) -> Tuple[Optional[List[CoreStepData]], Optional[str]]:
        # 1. Use a utility to unpickle `List[AbstractStep]` from `file_path`.
        # 2. Set `self._orchestrator.pipeline_definition` with the loaded steps.
        # 3. Convert each loaded AbstractStep into CoreStepData.
        # 4. Return (List[CoreStepData], None) or (None, "Error message").
        pass

    async def compile_orchestrator_pipeline(self, pipeline_steps: List[CoreStepData]) -> Tuple[bool, Optional[str]]:
        # 1. Similar to save_pipeline_to_file, ensure `self._orchestrator.pipeline_definition` (List[AbstractStep])
        #    is consistent with `pipeline_steps` from TUI. This might involve updating the core list.
        # 2. Call `self._orchestrator.compile_pipelines(self._orchestrator.pipeline_definition)`.
        # 3. Store compiled contexts and update orchestrator status.
        # 4. Return (True, None) or (False, "Error message with compilation error").
        pass

    async def execute_orchestrator_pipeline(self) -> Tuple[bool, Optional[str]]:
        # ... call self._orchestrator.execute_compiled_plate(...) ...
        # ... update orchestrator status ...
        # ... return success/error with execution error details ...
        pass

    async def get_step_definition_details(self, step_type_name: str, existing_params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        # 1. Dynamically import or get a reference to the actual AbstractStep subclass (e.g., FunctionStep, CompositeStep).
        # 2. Use `inspect.signature(StepClass.__init__)` to get parameters.
        # 3. Format these into a dictionary: {'param_name': {'default': ..., 'annotation': ..., 'type_str': ...}}.
        # 4. If `existing_params` are provided, merge them into the 'default' values for editing.
        # 5. Return (details_dict, None) or (None, "Error message").
        pass

    # ... other methods like get_plate_data, get_pipeline_steps, remove_step, move_step need to be fully implemented ...


class TUICoreAdapter(CoreApplicationAdapterInterface):
    def __init__(self, initial_context: ProcessingContext, global_config: GlobalPipelineConfig):
        # ... (initialization as before) ...
        # The TUICoreAdapter will manage multiple SingleOrchestratorAdapter instances or directly
        # manage multiple PipelineOrchestrator core instances.
        # For "compile all" / "run all", it would iterate over its managed orchestrator adapters
        # and call their compile/execute methods sequentially, aggregating results.
        self._initial_context = initial_context
        self._global_config = global_config
        self._file_manager = initial_context.filemanager
        self._loop = asyncio.get_event_loop()
        self._orchestrators: Dict[str, PipelineOrchestrator] = {}
        self._orchestrator_adapters: Dict[str, SingleOrchestratorAdapter] = {}


    async def _run_sync(self, func, *args, **kwargs):
        return await self._loop.run_in_executor(SHARED_EXECUTOR, lambda: func(*args, **kwargs))

    async def get_global_config(self) -> GlobalPipelineConfig:
        return self._global_config

    async def update_global_config_dict(self, config_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        # ... similar to previous update_global_config, but returns Tuple ...
        # Ensure Pydantic model update and persistence logic is robust.
        pass

    async def add_new_plate_orchestrator(self, folder_path: str) -> Tuple[Optional[str], Optional[str]]:
        # 1. Create Path object from folder_path.
        # 2. plate_id = str(folder_path.resolve()).
        # 3. If plate_id already in self._orchestrators, return (plate_id, "Plate already exists").
        # 4. Create a new OrchestratorConfig for this plate_path. It should inherit defaults from
        #    self._global_config (e.g., default_backend). If an orchestrator-specific config file
        #    exists in folder_path, load it. Otherwise, create a new one.
        # 5. Create PipelineOrchestrator instance.
        #    `core_orchestrator = PipelineOrchestrator(context=self._initial_context_for_new_plate, config_override=new_orch_config, plate_id_override=plate_id)`
        #    The context might need to be tailored or a new one created.
        #    The new orchestrator should be initialized with a default GlobalPipelineConfig.
        # 6. Store core_orchestrator in self._orchestrators.
        # 7. Create SingleOrchestratorAdapter for it and store in self._orchestrator_adapters.
        # 8. Return (plate_id, None) or (None, "Error message").
        pass

    async def add_new_plate_orchestrators(self, folder_paths: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
        results = []
        for path in folder_paths:
            results.append(await self.add_new_plate_orchestrator(path))
        return results

    async def list_directory_contents(self, path: str, backend_name: str = "disk") -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        # Use self._file_manager (which might need a backend concept) to list contents.
        # Example: `items = await self._run_sync(self._file_manager.list_directory, path, backend_name)`
        # Convert items to the required format: `[{'name': str, 'is_dir': bool, 'path': str}, ...]`.
        pass

    async def get_available_backends_for_func_registry(self) -> List[str]:
        # return [backend.value for backend in Backend] # Assuming Backend is an Enum used in FUNC_REGISTRY keys
        # Or directly: `return list(FUNC_REGISTRY.keys())` if keys are strings.
        pass

    async def get_functions_for_backend(self, backend_name: str) -> List[Dict[str, Any]]:
        # Access FUNC_REGISTRY[backend_name] and format into list of {'name': str, 'doc': Optional[str]}.
        pass

    async def get_function_signature(self, backend_name: str, func_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        # 1. Get the actual function object from FUNC_REGISTRY[backend_name][func_name]['function_ref'].
        # 2. Use `inspect.signature(func_obj)` to get signature.
        # 3. Iterate `sig.parameters.values()`:
        #    For each param, extract name, default (if not inspect.Parameter.empty), annotation (if not inspect.Parameter.empty).
        #    Store as {'param_name': {'default': ..., 'annotation': ..., 'kind': str(param.kind)}}.
        # 4. Return (params_dict, None) or (None, "Error message").
        pass

    async def save_step_to_file(self, step_data: CoreStepData, file_path: Path) -> Tuple[bool, Optional[str]]:
        # 1. Convert CoreStepData to an AbstractStep instance. This is complex.
        #    Requires knowing the step_type from CoreStepData and instantiating the correct AbstractStep subclass.
        #    Example: `core_step = FunctionStep(name=step_data.name, func_pattern=..., params=...)`.
        #    This conversion logic is crucial and might need a factory.
        # 2. Use `pickle.dump(core_step, file_handle)` to save.
        # 3. Return (True, None) or (False, "Error message").
        pass

    async def load_step_from_file(self, file_path: Path) -> Tuple[Optional[CoreStepData], Optional[str]]:
        # 1. Use `pickle.load(file_handle)` to load an AbstractStep instance.
        # 2. Convert the loaded AbstractStep to CoreStepData using `_core_step_to_tui_data`.
        # 3. Return (CoreStepData, None) or (None, "Error message").
        pass

    async def save_func_pattern_to_file(self, func_pattern: Any, file_path: Path) -> Tuple[bool, Optional[str]]:
        # Use `pickle.dump(func_pattern, file_handle)`.
        pass

    async def load_func_pattern_from_file(self, file_path: Path) -> Tuple[Optional[Any], Optional[str]]:
        # Use `pickle.load(file_handle)`.
        pass

    # ... other methods like get_available_plates, remove_plate, get_orchestrator_adapter ...
    # ... shutdown_core_services needs to properly shutdown all orchestrators and the SHARED_EXECUTOR ...

```

## 4. Refactoring TUI Modules

High-level strategy for refactoring key TUI modules:

*   **`openhcs.tui.TUIState`**:
    *   Will no longer hold direct instances of `PipelineOrchestrator`. Instead, it will store `plate_id` strings.
    *   `active_orchestrator` attribute will be removed or changed to `active_plate_id: Optional[str]`.
    *   `current_pipeline_definition` will store `List[CoreStepData]` (or `List[Dict]`) instead of `List[AbstractStep]`.
    *   `step_to_edit_config` will store `CoreStepData` (or `Dict`) instead of `FunctionStep`.

*   **`openhcs.tui.OpenHCSTUI`**:
    *   Will instantiate and hold a single `TUICoreAdapter` instance.
    *   All interactions that previously accessed `self.state.active_orchestrator` will now:
        1.  Get `active_plate_id` from `self.state`.
        2.  Call `self.core_adapter.get_orchestrator_adapter(active_plate_id)`.
        3.  Use the returned `CoreOrchestratorAdapterInterface` to perform operations.
    *   Example: `_handle_show_edit_plate_config_request` will pass the `CoreOrchestratorAdapterInterface` for the selected plate to the dialog/editor, or the dialog/editor will fetch it using `plate_id`.

*   **`openhcs.tui.commands.py`**:
    *   The `Command.execute` signature will change:
        `async def execute(self, app_adapter: CoreApplicationAdapterInterface, plate_adapter: Optional[CoreOrchestratorAdapterInterface], state: "TUIState", **kwargs: Any) -> None:`
        (Or pass `TUICoreAdapter` and let commands get specific adapters).
    *   Commands like `InitializePlatesCommand`, `CompilePlatesCommand`, `RunPlatesCommand` will use methods from `plate_adapter` (e.g., `plate_adapter.initialize()`).
    *   `AddStepCommand` will use `plate_adapter.add_step(...)`. It will no longer directly instantiate `FunctionStep` or access `FUNC_REGISTRY`.
    *   `LoadPipelineCommand` / `SavePipelineCommand` will use `plate_adapter.load_pipeline_definition_from_storage(...)` and `save_pipeline_definition_to_storage(...)`.
    *   Direct imports of core types like `PipelineOrchestrator`, `AbstractStep`, `FunctionStep`, `FUNC_REGISTRY`, `GlobalPipelineConfig` (for direct use, type hints for interfaces are okay) will be removed.
    *   The `SHARED_EXECUTOR` currently in `commands.py` will be moved to `core_adapters.py` or be a shared utility.

*   **`openhcs.tui.menu_bar.py` (and other UI components like `PlateManagerPane`, `PipelineEditorPane`, `DualStepFuncEditorPane`)**:
    *   Event handlers (e.g., `_on_compile`, `_on_run`) will primarily construct and dispatch `Command` objects.
    *   The `CommandRegistry` will execute these commands, passing the necessary adapter interfaces.
    *   Conditional enabling/disabling of menu items (`Condition` objects) will rely on data fetched from `TUIState`. `TUIState` itself will be updated via notifications originating from commands (after adapter calls) or directly from adapter methods if appropriate (e.g., after a polling update).
    *   Direct core imports will be removed.

## 5. Specific Import Changes (Illustrative)

*   **In `openhcs.tui.commands.py`**:
    *   **REMOVE**:
        ```python
        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
        from openhcs.core.steps.abstract import AbstractStep
        from openhcs.core.steps.function_step import FunctionStep
        from openhcs.processing.func_registry import FUNC_REGISTRY
        from openhcs.core.config import GlobalPipelineConfig
        from openhcs.core.context.processing_context import ProcessingContext # If only used for type hints for core objects
        ```
    *   **ADD**:
        ```python
        from .interfaces import CoreApplicationAdapterInterface, CoreOrchestratorAdapterInterface, CoreStepData
        # TYPE_CHECKING imports for TUIState, GlobalPipelineConfig (if used in signatures)
        ```

*   **In `openhcs.tui.tui_architecture.py` (`OpenHCSTUI` class)**:
    *   **REMOVE**: Direct instantiation of `PipelineOrchestrator`.
    *   **ADD**:
        ```python
        from .core_adapters import TUICoreAdapter
        from .interfaces import CoreStepData # For TUIState.current_pipeline_definition
        # TYPE_CHECKING for GlobalPipelineConfig, ProcessingContext for constructor
        ```

## 6. Addressing Dependency/Call Graph Findings

*   **Breaking Direct Dependencies**: This plan directly addresses and aims to eliminate the following critical dependencies identified in `reports/code_analysis/tui_comprehensive.md/module_dependency_graph_tui.md`:
    *   `openhcs.tui.commands` -> `openhcs.core.orchestrator.orchestrator`, `openhcs.core.steps.*`, `openhcs.processing.func_registry`, `openhcs.core.config`.
    *   `openhcs.tui.menu_bar` -> `openhcs.core.orchestrator.orchestrator`, `openhcs.core.steps.abstract`, `openhcs.core.config`.
    *   `openhcs.tui.tui_architecture` -> `openhcs.core.orchestrator.orchestrator`, `openhcs.core.steps.*`, `openhcs.processing.func_registry`.
    *   `openhcs.tui.plate_manager_core` -> `openhcs.core.orchestrator.orchestrator`.
    *   `openhcs.tui.pipeline_editor` -> `openhcs.core.orchestrator.orchestrator`, `openhcs.core.steps.*`.
*   **Resolution via Adapter**: The `TUICoreAdapter` and the defined interfaces act as an intermediary. TUI components will only know about these TUI-local abstractions, not the concrete core classes.

## 7. Verification Steps

1.  **Static Analysis**: After implementing changes, re-run `python tools/code_analysis/code_analyzer_cli.py dependencies openhcs/tui -o updated_tui_dependencies.md`. Verify that direct dependencies to `openhcs.core` from the refactored TUI modules are significantly reduced or eliminated, replaced by dependencies on `openhcs.tui.interfaces` and `openhcs.tui.core_adapters`.
2.  **Unit Tests**:
    *   Write unit tests for `TUICoreAdapter` and `SingleOrchestratorAdapter`. Mock the actual core objects (`PipelineOrchestrator`, `GlobalPipelineConfig`, etc.) and verify that adapter methods correctly delegate calls and handle data transformations.
    *   Update unit tests for `Command` subclasses. Mock the adapter interfaces and verify that commands call the correct adapter methods with appropriate arguments.
3.  **Integration Tests (TUI-Adapter)**: Test the interaction between TUI components (like `PlateManagerPane`, `PipelineEditorPane` through their commands) and the `TUICoreAdapter`, ensuring that UI actions correctly translate to adapter calls.
4.  **End-to-End Tests**: Existing TUI end-to-end tests (if any) should continue to pass, demonstrating that the refactoring has not broken overall functionality. New tests might be needed to cover specific interaction flows through the adapter.
5.  **Code Review**: Focus on ensuring that no direct core imports remain in the presentation/command layers of the TUI, and that all core interactions pass through the defined interfaces and adapter.

This plan provides a clear path to decouple the TUI from core logic, improving modularity, testability, and maintainability.