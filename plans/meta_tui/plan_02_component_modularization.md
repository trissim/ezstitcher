# Plan 02: TUI Component Modularization and Responsibility Refinement

**Version**: 1.0
**Date**: 2025-05-23
**Author**: MasterMind Architect

## 1. Introduction & Goal

**Problem**: Several TUI classes, notably `OpenHCSTUI` (in `tui_architecture.py`), `PlateManagerPane` (in `plate_manager_core.py`), `PipelineEditorPane` (in `pipeline_editor.py`), and `DualStepFuncEditorPane` (in `dual_step_func_editor.py`), have grown large and accumulate multiple responsibilities. `OpenHCSTUI` handles main layout construction, state observation, and acts as a central hub. The pane classes manage their own complex UI, state, event handling, and sometimes direct core interactions (which Plan 01 aims to decouple via adapters). This violates the Single Responsibility Principle and makes the components difficult to understand, test, and maintain.

**Goal**: To decompose these large UI classes into smaller, more focused, and cohesive sub-components. Each new component will have a single, well-defined responsibility (e.g., displaying a list, handling a specific set of user interactions, managing a sub-section of the UI). This will improve modularity, testability, and adherence to the Law of Demeter by ensuring components primarily interact through a central `TUIState` object or a dedicated event bus, rather than direct, complex inter-dependencies.

**Architectural Principles**:
*   **Single Responsibility Principle (SRP)**: Each class/module should have one reason to change.
*   **Composition over Inheritance**: Favor composing UIs from smaller, independent components.
*   **Law of Demeter**: Minimize direct knowledge between components; interactions mediated by `TUIState` or an event bus.
*   **Information Hiding**: Internal state and implementation details of sub-components should be encapsulated.

## 2. Proposed Component Decomposition

The TUI will be refactored into a more granular structure of views and controllers. Views are responsible for rendering UI elements and capturing user input, while controllers handle application logic, state management, and interactions with the core adapters. `TUIState` will serve as the central observable state.

### 2.1. Main Application Structure

*   **`openhcs.tui.AppController` class**:
    *   **Responsibility**: Top-level controller. Initializes and manages the lifecycle of other controllers and core services (like `TUICoreAdapter`). Handles global application events (startup, shutdown), global key bindings, and orchestrates major UI mode changes based on `TUIState`.
    *   **Interaction**: Instantiates `LayoutManager`, `TUIState`, and other primary controllers. Observes `TUIState` for high-level changes.
*   **`openhcs.tui.LayoutManager` class**:
    *   **Responsibility**: Constructing and managing the main application layout using `prompt_toolkit` containers (e.g., `HSplit`, `VSplit`, `Window`, `DynamicContainer`). This includes the top bar, a main content area split into a dynamic left pane and a right pane (Pipeline Editor), and the status bar.
    *   **Interaction**: Takes view components or their containers as input and arranges them. The left pane's content will be a `DynamicContainer` that switches between `PlateManagerView` and `StepFuncEditorOverlayView` based on `TUIState.active_left_pane_view`.
*   **`openhcs.tui.TUIState` class**:
    *   **Responsibility**: Central observable state holder. Contains attributes that define the current state of the UI (e.g., selected items, active views, data for dialogs). Implements an observer pattern (`register`, `notify`) to allow components to react to state changes.
    *   **Interaction**: Accessed by controllers to get current state and by views (often via controllers) to get data for rendering. Controllers update `TUIState`, triggering notifications to subscribed components.

### 2.2. Top Bar Components

*   **`openhcs.tui.components.top_bar.TopBarView` class**:
    *   **Responsibility**: Container for all top bar elements.
    *   **Interaction**: Arranges `GlobalSettingsButton`, `HelpButton`, `ExitButton`, and `TitleDisplay`.
*   **`openhcs.tui.components.top_bar.GlobalSettingsButton` class**:
    *   **Responsibility**: Button to open the global settings dialog.
    *   **Interaction**: On click, updates `TUIState` to signal the `AppController` (or a dedicated dialog controller) to show the global settings dialog, populating it with data from `TUICoreAdapter.get_global_config()`.
*   **`openhcs.tui.components.top_bar.HelpButton` class**:
    *   **Responsibility**: Button to display help information (e.g., key bindings, quick start).
    *   **Interaction**: On click, updates `TUIState` to trigger a help dialog/overlay.
*   **`openhcs.tui.components.top_bar.ExitButton` class**:
    *   **Responsibility**: Button to exit the application.
    *   **Interaction**: On click, triggers a shutdown sequence via `AppController`.
*   **`openhcs.tui.components.top_bar.TitleDisplay` class**:
    *   **Responsibility**: Displays the application title and current version.
    *   **Interaction**: Static display, potentially reads version from a config file or `TUIState`.

### 2.3. Plate Manager Components (Dynamic Left Pane Content)

*   **`openhcs.tui.views.plate_manager.PlateManagerView` class**:
    *   **Responsibility**: Main container for the plate manager interface.
    *   **Interaction**: Composes `PlateManagerTitleBarView`, `PlateManagerToolbarView`, and `PlateListView`.
*   **`openhcs.tui.components.plate_manager.PlateManagerTitleBarView` class**:
    *   **Responsibility**: Displays the title "Plate Manager".
*   **`openhcs.tui.components.plate_manager.PlateManagerToolbarView` class**:
    *   **Responsibility**: Horizontal bar with action buttons: Add (folder icon), Del (trash icon), Edit (pencil icon for plate config), Init (play icon), Compile (gear icon), Run (double play icon).
    *   **Interaction**: Buttons dispatch commands via `PlateManagerController`. Enablement state of buttons driven by `TUIState` (e.g., selected plate, plate status).
*   **`openhcs.tui.components.plate_manager.PlateListView` class**:
    *   **Responsibility**: Scrollable list of available plates. Each item displays status symbol (from `CorePlateData.status`), plate name, and plate path. Includes clickable "^" / "v" buttons next to each item for reordering.
    *   **Interaction**: Renders `List[CorePlateData]` from `TUIState`. Selection changes update `TUIState.selected_plate_id`. Reorder button clicks dispatch commands. Uses `InteractiveListItemWidget`.
*   **`openhcs.tui.controllers.PlateManagerController` class**:
    *   **Responsibility**: Manages the logic for the Plate Manager. Fetches plate data via `CoreApplicationAdapterInterface`. Handles actions from `PlateManagerToolbarView` and `PlateListView` (e.g., adding, removing, editing plates, initiating compilation/execution). Updates `TUIState` with changes to plate list or statuses. Manages display of plate configuration dialog.
    *   **Interaction**: Observes `TUIState` for relevant changes. Uses `CoreApplicationAdapterInterface` for data and operations. Updates `TUIState` which in turn refreshes views.

### 2.4. Pipeline Editor Components (Right Pane)

*   **`openhcs.tui.views.pipeline_editor.PipelineEditorView` class**:
    *   **Responsibility**: Main container for the pipeline editor interface, displaying steps of the currently selected plate.
    *   **Interaction**: Composes `PipelineEditorTitleBarView`, `PipelineEditorToolbarView`, and `StepListView`.
*   **`openhcs.tui.components.pipeline_editor.PipelineEditorTitleBarView` class**:
    *   **Responsibility**: Displays the title "Pipeline Editor" and the name/path of the active plate.
    *   **Interaction**: Reads active plate information from `TUIState`.
*   **`openhcs.tui.components.pipeline_editor.PipelineEditorToolbarView` class**:
    *   **Responsibility**: Horizontal bar with action buttons: Add (+ icon), Del (trash icon), Edit (pencil icon, triggers switch to STEP/FUNC Editor view), Load (open folder icon), Save (disk icon).
    *   **Interaction**: Buttons dispatch commands via `PipelineEditorController`. Edit button changes `TUIState.active_left_pane_view` to "step_func_editor" and sets `TUIState.current_step_for_editing`.
*   **`openhcs.tui.components.pipeline_editor.StepListView` class**:
    *   **Responsibility**: Scrollable list of steps for the active pipeline. Each item displays step name, status, and other `CoreStepData` details. Includes clickable "^" / "v" buttons for reordering.
    *   **Interaction**: Renders `List[CoreStepData]` from `TUIState` (based on `TUIState.selected_plate_id`). Selection changes update `TUIState.selected_step_id`. Reorder/edit actions dispatch commands or update `TUIState`. Uses `InteractiveListItemWidget`.
*   **`openhcs.tui.controllers.PipelineEditorController` class**:
    *   **Responsibility**: Manages logic for the Pipeline Editor. Fetches pipeline steps for the selected plate using `CoreOrchestratorAdapterInterface`. Handles actions from `PipelineEditorToolbarView` and `StepListView`. Updates `TUIState` with changes to steps.
    *   **Interaction**: Observes `TUIState` (e.g., `selected_plate_id`). Uses `CoreOrchestratorAdapterInterface` (obtained via `CoreApplicationAdapterInterface.get_orchestrator_adapter(selected_plate_id)`).

### 2.5. STEP/FUNC Editor Components (Dynamic Left Pane Content)

This view replaces the Plate Manager when a step is edited or a new step is created.

*   **`openhcs.tui.views.step_func_editor.StepFuncEditorOverlayView` class**:
    *   **Responsibility**: Main container for the step/function editor, managing the overall layout of this complex editing interface.
    *   **Interaction**: Composes `StepFuncToggleBarView`, and a dynamic area showing either `StepSettingsEditorView` or `FuncMenuView` based on `TUIState.step_func_editor_active_tab`.
*   **`openhcs.tui.components.step_func_editor.StepFuncToggleBarView` class**:
    *   **Responsibility**: Displays "Step Settings" and "Func Menu" toggle buttons. Also includes "Save" (saves current step) and "Close" (closes editor, switches back to Plate Manager view) buttons.
    *   **Interaction**: Toggle buttons update `TUIState.step_func_editor_active_tab`. Save/Close buttons dispatch commands via `StepFuncEditorController`.
*   **`openhcs.tui.views.step_func_editor.StepSettingsEditorView` class**:
    *   **Responsibility**: Displays and allows editing of general step parameters.
        *   **Header**: Contains "Load .step" and "Save As .step" buttons.
        *   **Form**: Dynamically generated form for `AbstractStep` parameters using `DynamicFormGenerator`. Fields include: `name` (text input), `input_dir` (text input with browse button), `output_dir` (text input with browse button), `force_disk_output` (checkbox), `variable_components` (complex editor/list), `group_by` (list editor). Each field has a "reset to default" button.
    *   **Interaction**: Data loaded from `TUIState.current_step_for_editing`. Changes update a temporary state within `StepFuncEditorController`, committed on "Save". Browse buttons trigger file dialogs. Load/Save As buttons dispatch commands.
*   **`openhcs.tui.views.step_func_editor.FuncMenuView` class (for `FunctionStep` types)**:
    *   **Responsibility**: Manages the `func_pattern` (list of function calls) for a `FunctionStep`.
        *   **Header**: "Add Function Call" button, "Load .func" (loads a pattern), "Save As .func" (saves current pattern).
        *   **Keys Management**: Area to manage `dict_keys` for the function pattern.
        *   **Function Call List**: Scrollable list of "Func X" items. Each item represents a function call in the pattern and includes:
            *   A dropdown to select the function (from `FUNC_REGISTRY` via adapter).
            *   Dynamically generated fields for function keyword arguments (`kwargs`), each with a reset button.
            *   Move Up/Down buttons for reordering the function call.
            *   Delete button for removing the function call.
    *   **Interaction**: Data loaded from `TUIState.current_step_for_editing.params['func_pattern']`. Changes update temporary state in `StepFuncEditorController`. Add/Load/Save As buttons dispatch commands. Dropdown selection fetches signature for dynamic kwargs.
*   **`openhcs.tui.controllers.StepFuncEditorController` class**:
    *   **Responsibility**: Manages the state and logic for the entire STEP/FUNC editor. Loads `CoreStepData` to be edited from `TUIState.current_step_for_editing`. Handles data conversion between `CoreStepData` and UI representation. Interfaces with `CoreApplicationAdapterInterface` and `CoreOrchestratorAdapterInterface` for loading/saving step/pattern files, fetching function signatures, etc. Manages temporary state for edits before saving.
    *   **Interaction**: Observes `TUIState`. Drives `StepSettingsEditorView` and `FuncMenuView`. Handles save/close logic.

### 2.6. Status Bar Components

*   **`openhcs.tui.views.status_bar.StatusBarView` class**:
    *   **Responsibility**: Displays the live log messages. Clickable to expand/collapse `LogHistoryDrawerView`.
    *   **Interaction**: Shows latest log message from `TUIState.log_messages`. Click updates `TUIState.is_status_log_expanded`.
*   **`openhcs.tui.components.status_bar.LogHistoryDrawerView` class**:
    *   **Responsibility**: A drawer/overlay that shows a scrollable history of log messages.
    *   **Interaction**: Visibility controlled by `TUIState.is_status_log_expanded`. Displays `TUIState.log_messages_history`.
*   **`openhcs.tui.controllers.StatusBarController` class**:
    *   **Responsibility**: Manages log message updates and the expanded state of the log history. Appends new log messages to `TUIState` attributes.
    *   **Interaction**: Subscribes to logging events/notifications from other parts of the application (e.g., core adapter after operations). Updates `TUIState`.

### 2.7. Shared/Common Components (Optional Section for Clarity)

*   **`openhcs.tui.widgets.InteractiveListItemWidget`**: Reusable widget for creating selectable, focusable list items with complex layouts (e.g., text, buttons). Used by `PlateListView` and `StepListView`.
*   **`openhcs.tui.widgets.DynamicFormGenerator`**: Utility/component to create form UI elements based on a schema or parameter list (e.g., for `StepSettingsEditorView` and `FuncMenuView` kwargs).
*   **`openhcs.tui.dialogs.FileDialogView`**: A generic file/directory dialog component, invoked by various buttons (e.g., Load/Save .step, Load/Save .func, browse for input/output dirs). Its behavior (file/dir mode, callback) configured via `TUIState.file_dialog_context`.

## 3. State Management and Communication (`TUIState` and Event Bus)

*   **`TUIState` (`openhcs.tui.TUIState`)**:
    *   Will continue to be the central observable state holder.
    *   Key attributes will include:
        *   `active_plate_id: Optional[str]`
        *   `selected_step_id: Optional[str]`
        *   `plates_data: List[CorePlateData]` (list of all available plates)
        *   `current_pipeline_steps: List[CoreStepData]` (steps for the `active_plate_id`)
        *   `active_left_pane_view: Literal["plate_manager", "step_func_editor"]` (controls dynamic left pane)
        *   `step_func_editor_active_tab: Literal["step_settings", "func_menu"]` (controls tab in STEP/FUNC editor)
        *   `current_step_for_editing: Optional[CoreStepData]` (data for the step being edited in STEP/FUNC editor)
        *   `current_step_unsaved_changes: Dict[str, Any]` (temporary storage for unsaved step edits)
        *   `data_for_global_settings_dialog: Optional[Dict[str, Any]]` (derived from `GlobalPipelineConfig` for dialogs)
        *   `data_for_plate_config_dialog: Optional[Dict[str, Any]]` (orchestrator config for dialogs)
        *   `is_status_log_expanded: bool` (controls visibility of `LogHistoryDrawerView`)
        *   `log_messages_history: List[str]`
        *   `latest_log_message: Optional[str]`
        *   `file_dialog_context: Optional[Dict[str, Any]]` (e.g., `{ "is_active": True, "mode": "file_open", "current_path": Path, "allowed_extensions": [".json"], "callback_command_name": "LoadPipelineCommand", "callback_args": {"plate_id": ...} }`)
        *   `command_to_execute: Optional[Tuple[Type[Command], Dict[str, Any]]]` (for deferred execution or cross-component calls)
        *   Other flags for dialog visibility (e.g., `show_help_dialog: bool`, `show_confirmation_dialog: Optional[Dict[str,Any]]`).
*   **Event Bus (implemented via `TUIState.notify` and specific event handler methods in controllers)**:
    *   UI view components (e.g., `PlateListView`, `StepListView`, various buttons) will primarily dispatch `Command` objects or call methods on their respective controllers.
    *   Controllers will handle these actions, interact with core adapters, and then update `TUIState`.
    *   `TUIState.notify()` will alert all registered components (views, other controllers) about general state changes.
    *   For more specific, less state-driven communication (e.g., a dialog needs to signal completion of a specific task not directly tied to a persistent state field), a command-based approach or dedicated callback mechanisms within `TUIState` can be used. Reiterate the role of the event bus for communication, especially for less direct state changes or actions, like triggering a file dialog with specific parameters and a callback command.
    *   Controller components (`PlateManagerController`, `PipelineEditorController`, `StepFuncEditorController`, `AppController`, `StatusBarController`) will subscribe to relevant granular notifications from `TUIState` if needed, or simply re-read relevant parts of `TUIState` when a general notification is received.
    *   Commands will be dispatched by controllers or action toolbars, and their execution (via adapters) will result in `TUIState` updates, triggering view refreshes. The `command_to_execute` attribute in `TUIState` can be used for a more decoupled command invocation pattern.

## 4. Refactoring Steps (High-Level)

1.  **Create New Component Files**: Create the new Python files for the decomposed views and controllers as outlined above (e.g., `plate_list_view.py`, `plate_manager_controller.py`, etc., within `openhcs.tui.components` and `openhcs.tui.controllers` sub-packages).
2.  **Migrate UI Rendering Logic**:
    *   Move UI element creation (e.g., `_build_plate_items_container` from `PlateManagerPane` to `PlateListView`) and associated display logic (e.g., `_get_plate_display_text`) into the new view components.
    *   Views will become primarily responsible for rendering data they receive and emitting user interaction events.
3.  **Migrate Control Logic**:
    *   Move event handling, state management logic, and command dispatching logic from the old large pane classes into the new controller classes.
    *   Controllers will observe `TUIState`, fetch data via core adapters (as per Plan 01), prepare data for their views, and handle actions initiated by views or toolbars.
4.  **Refactor `OpenHCSTUI`**:
    *   Delegate layout construction to `LayoutManager`.
    *   Delegate component lifecycle and high-level state orchestration to `AppController`.
5.  **Update `TUIState`**:
    *   Ensure `TUIState` holds data in the form of `CorePlateData` and `CoreStepData` where appropriate.
    *   Refine event types for more granular communication if needed.
6.  **Testing**:
    *   Write unit tests for new view components (testing rendering based on input data, and event emission).
    *   Write unit tests for new controller components (testing state handling, command dispatch, and interaction with mocked adapters and views).
    *   Update/create integration tests to ensure composed components work together correctly.

## 5. Verification

*   **Code Structure**: Verify that the new file structure reflects the decomposition.
*   **Class Size and Responsibility**: Check that the new classes are smaller and have more focused responsibilities (e.g., using `wily` or manual inspection).
*   **Reduced Coupling**: Analyze dependencies (e.g., using `tools/code_analysis/code_analyzer_cli.py dependencies`) to ensure new view components primarily depend on `TUIState` or their controller, and controllers depend on `TUIState` and core adapter interfaces. Direct dependencies between view components should be minimized.
*   **Test Coverage**: Ensure adequate test coverage for the new and refactored components.
*   **Functional Equivalence**: The TUI should retain its existing functionality after refactoring.

This plan aims to create a more maintainable, testable, and understandable TUI codebase by adhering to established software design principles.