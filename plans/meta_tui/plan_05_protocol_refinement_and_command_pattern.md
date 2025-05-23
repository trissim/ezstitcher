# Plan 05: TUI Protocol Refinement and Command Pattern Solidification

**Version**: 1.0
**Date**: 2025-05-23
**Author**: MasterMind Architect

## 1. Introduction & Goal

**Problem**: The `openhcs.tui` package uses Python `Protocol`s for defining interfaces, notably for commands and callbacks. However, the `reports/code_analysis/tui_comprehensive.md/interface_analysis.md` report indicates that some classes implement a large number of these protocols (e.g., command classes implementing `Command`, `PlateEventHandler`, `DialogResultCallback`, `ErrorCallback`, `ValidationResultCallback`). This suggests that the current protocols might be too broad or that classes are taking on too many disparate roles. Furthermore, the `Command` protocol itself is very generic. The `semantic_role_analysis.md` report also shows a mix of roles within UI components, some of which might be better encapsulated by more specific command objects or delegated through clearer interfaces.

**Goal**: To refine the existing TUI protocols and solidify the Command pattern by:
    1.  Reviewing and potentially specializing the `Command` protocol and other key protocols (like `DialogResultCallback`, `ErrorCallback`, `PlateEventHandler`) to better reflect specific interaction types.
    2.  Ensuring that classes implement only the protocols relevant to their primary responsibility, potentially by delegating secondary responsibilities to collaborator objects.
    3.  Clarifying the data contracts (arguments and return types) for protocol methods, using `CorePlateData`, `CoreStepData` (from Plan 01), or other TUI-specific data transfer objects (DTOs) instead of raw `Dict[str, Any]` where appropriate.
    4.  Ensuring command objects are pure encapsulations of an action, taking necessary context (like adapters and `TUIState`) during execution rather than holding excessive state themselves.
    5.  Recognizing that the TUI redesign (Plans 01, 02, 03) introduces a large number of new user interactions which must be encapsulated via a robust and comprehensive command pattern.

**Architectural Principles**:
*   **Interface Segregation Principle (ISP)**: Clients should not be forced to depend on interfaces they do not use. Protocols should be fine-grained.
*   **Single Responsibility Principle (SRP)**: Classes (including commands) should have one primary responsibility.
*   **Command Pattern**: Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.
*   **Clear Contracts**: Interfaces should clearly define the expected inputs and outputs.

## 2. Proposed Protocol and Command Refinements

### 2.1. Review and Specialize `Command` Protocol

*   **Current**: `Command(Protocol)` with `async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any)` and `def can_execute(self, state: "TUIState")`.
*   **Observation**: The `context: "ProcessingContext"` and `**kwargs` are very general. Plan 01 proposes changing `execute` to receive adapter interfaces.
*   **Proposed Action**:
    1.  **Base `TUICommand(Protocol)`**:
        ```python
        # In openhcs.tui.interfaces.py (or a new openhcs.tui.commands.protocols.py)
        from typing import Protocol, Any, List, Optional, Dict, Coroutine, Tuple, TYPE_CHECKING
        if TYPE_CHECKING:
            from openhcs.tui.state import TUIState # Assuming TUIState is defined here
            from openhcs.tui.adapters.core_adapter_protocols import CoreApplicationAdapterInterface, CoreOrchestratorAdapterInterface

        class TUICommand(Protocol):
            async def execute(self, app_adapter: "CoreApplicationAdapterInterface",
                              plate_adapter: Optional["CoreOrchestratorAdapterInterface"], # If relevant to current context
                              ui_state: "TUIState", # Renamed for clarity
                              **kwargs: Any) -> Coroutine[Any, Any, Tuple[bool, Optional[str]]]: ... # kwargs for command-specific params

            def can_execute(self, ui_state: "TUIState",
                            app_adapter: Optional["CoreApplicationAdapterInterface"] = None, # Optional for simple checks
                            plate_adapter: Optional["CoreOrchestratorAdapterInterface"] = None
                           ) -> bool:
                return True # Default
        ```
    2.  **Specialized Command Protocols (Examples, if beneficial)**:
        *   While specific marker protocols like `DialogCommand` could be created, the primary focus will be on implementing concrete command classes. The base `TUICommand` should be robust enough. Specialization might occur naturally if common setup/execution patterns emerge for groups of commands.
    *   **Rationale**: The `execute` method now returns a success status and an optional message, allowing UI components (like the status bar or dialogs) to react to the outcome of operations. This aligns with the adapter methods from Plan 01.

    3.  **Examples of Specific `TUICommand` Classes (Non-Exhaustive List)**:
        *   **Top Bar:**
            *   `ShowGlobalSettingsDialogCommand(TUICommand)`: Params: None.
            *   `ShowHelpDialogCommand(TUICommand)`: Params: None.
            *   `ExitApplicationCommand(TUICommand)`: Params: None.
        *   **Plate Manager Toolbar & List Interactions:**
            *   `AddPlatesCommand(TUICommand)`: Params: `folder_paths: List[str]`. (Handles multi-folder selection from a dialog).
            *   `DeleteSelectedPlatesCommand(TUICommand)`: Params: `plate_ids: List[str]` (from `ui_state.selected_plate_ids`).
            *   `ShowEditPlateConfigDialogCommand(TUICommand)`: Params: `plate_id: str`.
            *   `InitializeSelectedPlatesCommand(TUICommand)`: Params: `plate_ids: List[str]`.
            *   `CompileSelectedPlatesCommand(TUICommand)`: Params: `plate_ids: List[str]`.
            *   `CompileAllInitializedPlatesCommand(TUICommand)`: Params: None. (Alternative to selected)
            *   `RunSelectedPlatesCommand(TUICommand)`: Params: `plate_ids: List[str]`.
            *   `RunAllCompiledPlatesCommand(TUICommand)`: Params: None. (Alternative to selected)
            *   `ReorderPlateCommand(TUICommand)`: Params: `plate_id: str`, `direction: Literal["up", "down"]`.
        *   **Pipeline Editor Toolbar & List Interactions:**
            *   `AddDefaultStepCommand(TUICommand)`: Params: `plate_id: str`, `default_step_type: str`.
            *   `DeleteSelectedStepsCommand(TUICommand)`: Params: `plate_id: str`, `step_ids: List[str]`.
            *   `ShowStepFuncEditorCommand(TUICommand)`: Params: `plate_id: str`, `step_id: str` (or `is_new_step: bool`). Updates `ui_state.active_left_pane_view` and `ui_state.current_step_for_editing`.
            *   `LoadPipelineCommand(TUICommand)`: Params: `plate_id: str`, `file_path: Path`.
            *   `SavePipelineCommand(TUICommand)`: Params: `plate_id: str`, `file_path: Optional[Path]` (optional for save-as).
            *   `ReorderStepCommand(TUICommand)`: Params: `plate_id: str`, `step_id: str`, `direction: Literal["up", "down"]`.
        *   **STEP/FUNC Editor (Main Toolbar):**
            *   `ToggleStepFuncViewCommand(TUICommand)`: Params: `target_view: Literal["step_settings", "func_menu"]`. Updates `ui_state.step_func_editor_active_tab`.
            *   `SaveStepFromEditorCommand(TUICommand)`: Params: `plate_id: str`, `step_data: CoreStepData` (from `ui_state.current_step_for_editing` potentially modified by `current_step_unsaved_changes`).
            *   `CloseStepFuncEditorCommand(TUICommand)`: Params: None. Updates `ui_state.active_left_pane_view` to "plate_manager".
        *   **STEP/FUNC Editor (Step Settings View):**
            *   `LoadStepDefinitionCommand(TUICommand)`: Params: `plate_id: str`, `file_path: Path`. (Loads into current editor state).
            *   `SaveStepDefinitionAsCommand(TUICommand)`: Params: `step_data: CoreStepData`, `file_path: Path`.
            *   `ResetStepParameterCommand(TUICommand)`: Params: `step_uid: str`, `parameter_key: str`. (Resets to default or initial value).
        *   **STEP/FUNC Editor (Func Menu View):**
            *   `AddFunctionToPatternCommand(TUICommand)`: Params: `step_uid: str`.
            *   `LoadFuncPatternCommand(TUICommand)`: Params: `step_uid: str`, `file_path: Path`.
            *   `SaveFuncPatternAsCommand(TUICommand)`: Params: `func_pattern_data: Any`, `file_path: Path`.
            *   `ManageFuncPatternDictKeyCommand(TUICommand)`: Params: `step_uid: str`, `action: Literal["add", "remove", "edit"]`, `key_name: Optional[str]`, `new_key_name: Optional[str]`.
            *   `SelectFunctionForPatternItemCommand(TUICommand)`: Params: `step_uid: str`, `pattern_item_index: int`, `backend_name: str`, `func_name: str`.
            *   `ResetFuncKwargCommand(TUICommand)`: Params: `step_uid: str`, `pattern_item_index: int`, `kwarg_name: str`.
        *   **Status Bar:**
            *   `ToggleLogHistoryDrawerCommand(TUICommand)`: Params: None. Updates `ui_state.is_status_log_expanded`.
        *   **Dialog Actions:**
            *   `SubmitGlobalSettingsCommand(TUICommand)`: Params: `updated_config_data: Dict[str, Any]`.
            *   `SubmitPlateConfigCommand(TUICommand)`: Params: `plate_id: str`, `updated_config_data: Dict[str, Any]`.
            *   `FileDialogConfirmCommand(TUICommand)`: Params: `dialog_context: Dict[str, Any]`, `selected_path: Path`. Invokes the callback command from `dialog_context`.
            *   `FileDialogCancelCommand(TUICommand)`: Params: `dialog_context: Dict[str, Any]`.

### 2.2. Refine Callback Protocols

*   **Current**: `DialogResultCallback`, `ErrorCallback`, `ValidationResultCallback`, `PlateEventHandler` are defined as `Protocol`s.
*   **Observation**: Wide implementation suggests these might be too generic or components act as catch-alls.
*   **Proposed Action**:
    1.  **Prioritize `TUIState` as Event Bus**: Many previous callback protocols might be deprecated. Instead of direct callbacks, commands will modify `TUIState`. UI components (views and controllers) will observe `TUIState` and react to relevant changes. This decouples components significantly. For example, instead of a `PlateEventHandler.on_plate_added` callback, an `AddPlatesCommand` would update `ui_state.plates_data`, and any view displaying the plate list would re-render based on this new state.
    2.  **Minimal Essential Callbacks**: If any callbacks remain essential for highly dynamic UI updates not easily achieved by observing `TUIState` (e.g., a very specific interaction within a complex widget not fully managed by a controller), their `data` parameters must use specific DTOs/TypedDicts, not generic `Dict[str, Any]`.
        *   `DialogResultCallback(data: SpecificDialogOutputData) -> None` (if specific dialogs need direct, non-command callbacks).
        *   `ErrorCallback` might still be useful for components that can receive errors outside the main command flow, but its usage should be reviewed.
    3.  **Remove `PlateEventHandler` as a broad protocol**: Its specific events are better handled by commands updating `TUIState` (e.g., `plate_added` results in updated `plates_data` list; `plate_selected` updates `active_plate_id`).

### 2.3. Data Contracts for Protocol Methods

*   **Action**: Review all protocol methods. This includes the `CoreApplicationAdapterInterface` and `CoreOrchestratorAdapterInterface` from Plan 01, the refined `TUICommand`, and any remaining callback protocols.
*   Replace generic `Dict[str, Any]`, `Any`, `List[Dict]` with specific types:
    *   `CorePlateData`, `CoreStepData` (from Plan 01 interfaces).
    *   New `TypedDict` or Pydantic models for other structured data (e.g., `PlateConfigUpdateData`, `GlobalConfigUpdateData`, `FileDialogContext`).
    *   Command `__init__` parameters or `execute`'s `**kwargs` should also use these specific DTOs or `TypedDict`s for clarity when commands are instantiated or dispatched with parameters.
*   **Example**:
    *   `ShowEditPlateConfigDialogCommand(plate_id: str)` in `__init__`.
    *   `AddPlatesCommand.execute(**kwargs)` might expect `kwargs['folder_paths']: List[str]`.

### 2.4. Command Object State and Responsibilities

*   **Action**:
    1.  **Statelessness**: Commands should be stateless or hold minimal state directly related to their specific invocation (e.g., parameters passed via `__init__` or `**kwargs` to `execute`). They derive necessary information from `ui_state` and adapters during execution.
    2.  **Single Action**: Each command class encapsulates a single, well-defined user action.
    3.  **Constructor**: Command `__init__` methods should be lightweight, primarily for setting up fixed parameters necessary for the command's execution.
    4.  **`can_execute` Logic**: This method is crucial for dynamically enabling/disabling UI elements in the new TUI design. It will rely heavily on `ui_state` (e.g., `ui_state.selected_plate_id`, `ui_state.active_left_pane_view`, status of items) and potentially query adapters. It must be efficient.

## 3. Refactoring Steps

1.  **Create/Update Interface Files**:
    *   Modify `openhcs/tui/interfaces.py` (from Plan 01) to include the refined `TUICommand` protocol.
    *   Review and significantly reduce or eliminate `openhcs.tui.callbacks.py` in favor of `TUIState` observation.
    *   Define necessary DTOs/`TypedDict`s in `openhcs.tui.types.py`.
2.  **Refactor and Create Command Classes (`openhcs.tui.commands/` directory with submodules)**:
    *   Update existing command classes to inherit from the new `TUICommand` and modify their signatures and logic.
    *   **Create all the new command classes** identified in section 2.1.3, ensuring they correctly interact with adapters and `TUIState`.
    *   Organize commands into submodules within `openhcs.tui.commands/` (e.g., `plate_commands.py`, `step_commands.py`, `dialog_commands.py`).
3.  **Refactor Components Implementing Callbacks**:
    *   Remove implementations of deprecated callback protocols.
    *   Modify components (now primarily controllers and views as per Plan 02) to observe `TUIState` for changes and update themselves accordingly.
4.  **Update Command Instantiation and Execution**:
    *   Wire up the new commands in the new UI components (toolbars, menus, dialogs, list item interactions as defined in Plan 02). This involves:
        *   Ensuring UI elements (buttons, etc.) trigger the instantiation and execution of the correct command.
        *   Passing necessary parameters (often from `TUIState` or user input) to the command's `__init__` or `execute`'s `**kwargs`.
        *   Using the result (`Tuple[bool, Optional[str]]`) of command execution to provide feedback to the user (e.g., via status bar updates in `TUIState`).
    *   Ensure `can_execute` is called by UI components to dynamically set the enabled/disabled state of buttons.

## 4. Verification

1.  **Static Type Checking**: Run `mypy` on the `openhcs.tui` package. Verify that type hints for protocols, commands, and callbacks are consistent and that type errors are resolved.
2.  **Interface Analysis**: Re-run `python tools/code_analysis/interface_classifier.py openhcs/tui -o updated_interface_analysis.md`.
    *   Check that the new/refined protocols are correctly identified.
    *   Analyze the "Implementations" section to see if classes now implement a more focused set of protocols.
3.  **Semantic Role Analysis**: Re-run `python tools/code_analysis/semantic_role_analyzer.py openhcs/tui -o updated_semantic_role_analysis.md`.
    *   Assess if command classes now primarily contain `ACTION_DISPATCH` or `LOGIC_EXECUTION` roles related to their specific command, and less of other roles.
4.  **Code Review**:
    *   Ensure protocol methods have clear, typed signatures.
    *   Verify that commands are largely stateless and encapsulate single actions.
    *   Check that callback implementations are focused.
5.  **Functional Tests**: Ensure TUI interactions, dialogs, and command-driven actions still function correctly.

This plan will lead to a more robust and maintainable command and event handling system within the TUI, with clearer contracts and better separation of responsibilities.