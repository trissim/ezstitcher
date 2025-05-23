# Plan 04: TUI Import Hygiene and Structure Refinement

**Version**: 1.0
**Date**: 2025-05-23
**Author**: MasterMind Architect

## 1. Introduction & Goal

**Problem**: The `openhcs.tui` package, as revealed by the `reports/code_analysis/tui_comprehensive.md/import_analysis.md` report, suffers from numerous import-related issues:
    *   **Missing Imports**: Many symbols are used without being explicitly imported (e.g., in `components.py`, `pipeline_editor.py`). This often relies on symbols being implicitly available from wildcard imports or being injected into a module's namespace, leading to code that is hard to understand and prone to breakage.
    *   **Unused Imports**: Several modules import symbols that are never used (e.g., `Dialog`, `HTML`, `ast` in `components.py`; `FunctionPatternEditor` in `openhcs/tui/__init__.py`). This clutters the namespace and can mislead developers about module dependencies.
    *   **Module Structure Issues**: Some imports might point to internal structures of other modules or use non-standard import paths (e.g., `prompt_toolkit.layout.Container` being noted as an issue, though this might be a linter false positive if it's the canonical path).
    *   **Disorganized Imports**: Imports are not consistently grouped or ordered according to PEP 8, making it harder to quickly assess module dependencies.

**Goal**: To improve the clarity, maintainability, and robustness of the `openhcs.tui` package by:
    1.  Systematically addressing all missing and unused imports identified in the analysis report and ensuring new modules adhere to strict import hygiene from the outset.
    2.  Organizing imports according to PEP 8 guidelines (standard library, third-party, local application) across all existing and new TUI modules.
    3.  Consolidating common TUI utility functions into dedicated utility modules to reduce scattered helper code and improve import paths, especially for new functionalities introduced in the redesign.
    4.  Enforcing clearer module boundaries by ensuring modules only import what they directly need from public interfaces of other modules.
    *   The TUI redesign (Plans 01, 02, 03) will introduce many new modules (components, controllers, views, adapters, interfaces, utilities). It's critical that these new parts of the codebase establish and maintain excellent import hygiene from their inception to avoid repeating past issues.

**Architectural Principles**:
*   **Explicitness**: Imports should clearly declare all external symbols a module uses.
*   **Minimality (Least Privilege for Imports)**: Modules should only import what they need.
*   **Standardization**: Adherence to PEP 8 for import formatting improves readability.
*   **Modularity**: Well-defined utility modules promote reuse and reduce code duplication.

## 2. Refactoring Steps

### 2.1. Address Missing Imports

*   **Action**: For *all existing and newly created files* within the `openhcs.tui` package as part of the redesign:
    1.  Identify the source module for any missing symbols. This requires diligent development practices for new modules and careful review of existing ones.
    2.  Add an explicit `from module import symbol` or `import module` statement.
    3.  Prioritize importing from public APIs of modules rather than internal submodules if possible.
*   **Note**: While the original report highlighted specific files, the TUI redesign means this check must be comprehensive across the entire `openhcs.tui` package, including all new view, controller, component, adapter, interface, and utility modules.
*   **Tooling**: Use an IDE with import resolution capabilities or manually trace symbol origins during development and code reviews. Linters can help identify missing imports in new code.

### 2.2. Remove Unused Imports

*   **Action**: For *all existing and newly created files* within the `openhcs.tui` package:
    1.  Verify that any imported symbol is genuinely used within the module.
    2.  Remove the corresponding import statement if unused.
*   **Note**: Similar to missing imports, this applies to the entire refactored `openhcs.tui` package. The original "Key Files from Report" serve as examples of past issues to avoid.
*   **Tooling**: Linters like `flake8` (with `flake8-import-order` and `flake8-unused-imports`) or `pylint` can automate detection. Autofixers like `autoflake` can remove them. Regular use of these tools during development of new modules is crucial.

### 2.3. Organize Imports (PEP 8)

*   **Action**: For every Python file in `openhcs.tui`, *including all new files created during the redesign*:
    1.  Group imports into three sections:
        *   Standard library imports (e.g., `os`, `typing`, `asyncio`).
        *   Third-party library imports (e.g., `prompt_toolkit`).
        *   Local application/library imports (e.g., `from openhcs.core.config import ...`, `from .components import ...`).
    2.  Within each section, sort imports alphabetically.
    3.  Separate sections with a blank line.
*   **Tooling**: `isort` is the standard tool for automatically sorting and formatting imports. Configure and use it project-wide, especially as new files are added.

### 2.4. Consolidate TUI Utilities

*   **Problem**: The TUI redesign will likely generate new shared functionalities. The existing `openhcs/tui/utils.py` and its submodules (`dialog_helpers.py`, `error_handling.py`) provide a foundation, but new needs will arise.
*   **Action**:
    1.  **Review existing utilities**: Ensure `openhcs/tui/utils/__init__.py` correctly exports symbols from its submodules for use by other TUI components. Consolidate or keep them specific as appropriate.
    2.  **Proactively create new utilities**: As new shared functionalities are identified during the implementation of components from Plan 02, they should be actively considered for placement in `openhcs.tui.utils` or new, appropriately named utility submodules (e.g., `openhcs.tui.utils.form_helpers`, `openhcs.tui.utils.ui_converters`).
        *   **Examples of potential new utilities arising from the redesign**:
            *   `DynamicFormGenerator`: For creating UI elements from configuration objects or function signatures (used in Global Settings, Plate Config, Step Settings Editor, Func Menu kwargs).
            *   `FileDialogLogic`: Helper functions for managing TUI file/folder dialog navigation, potentially using `FileManager` from Plan 01 for backend operations if the TUI library's built-in dialogs are insufficient.
            *   `TUIEnumHelpers`: Functions to convert enums (e.g., TUI-side representations of `VariableComponents` or `GroupBy` from `CoreStepData`) into choices suitable for dropdown menus or other UI widgets.
            *   `InspectUtils (TUI-specific)`: Any TUI-specific helpers for dealing with function/constructor signatures obtained from the adapter layer, if further processing is needed for UI display (e.g., formatting parameter types, defaults).
            *   `WidgetFactory`: Functions that create standardized versions of common `prompt_toolkit` widgets with consistent styling or behavior.
    3.  **Refactor imports**: Ensure all TUI components import utilities from these centralized and well-defined locations.
    4.  **Import Hygiene for Utilities**: Stress that these utility modules themselves must adhere to strict import hygiene, importing only necessary dependencies and using clear, absolute, or relative imports.
*   **Example**: If a new function `create_styled_button(text: str, handler: Callable)` is developed and used in multiple toolbars, it should be placed in a utility module like `openhcs.tui.utils.widget_helpers` and imported from there.

### 2.5. Address Module Structure Issues (from report and for new modules)

*   **Original Issue Example**: The report mentioned `prompt_toolkit.layout.Container` in `openhcs/tui/components.py`. The analysis remains: verify canonical import paths. If `from prompt_toolkit.layout import Container` is standard, it's fine. If a higher-level API like `from prompt_toolkit.widgets import SomeContainerWidget` is preferred, use that.
*   **General Action for All Modules (Existing and New)**:
    *   Ensure imports use the most direct and public paths to symbols from external libraries (like `prompt_toolkit`). Avoid importing from `_internal` or deeply nested modules if a more public API is available.
    *   Within `openhcs.tui`, new modules (views, controllers, components, utilities) should be structured to allow clear and direct imports of the symbols they intend to expose. For example, if `PlateListView` is in `openhcs.tui.views.plate_manager`, it should be importable as `from openhcs.tui.views.plate_manager import PlateListView`. Avoid overly complex internal structures that necessitate deep imports into other modules' internals.
    *   Use `__init__.py` files strategically to define the public API of a sub-package (e.g., `openhcs.tui.utils/__init__.py` could expose key utility functions by importing them: `from .form_helpers import DynamicFormGenerator`).

## 3. Verification

1.  **Static Analysis (Primary)**:
    *   After refactoring and implementing new TUI modules, re-run `python tools/code_analysis/meta_analyzer.py comprehensive openhcs/tui -o reports/code_analysis/tui_comprehensive_updated.md` (or specifically the import analysis part: `python tools/code_analysis/../import_analysis/import_validator.py openhcs/tui -o updated_import_analysis.md`). This analysis must cover the *entire refactored* `openhcs.tui` package.
    *   Verify that "Missing Imports" and "Unused Imports" are zero or acceptably minimal across all modules.
    *   Manually inspect a selection of new and existing files to confirm PEP 8 import ordering.
2.  **Linter Pass**: Run `flake8` and `pylint` (if configured) over the *entire* `openhcs.tui` package. Address all new import-related warnings or errors. This should be part of the standard development workflow for new modules.
3.  **Functionality Tests**: Run unit and integration tests for the TUI. Correct import hygiene is crucial for tests to run correctly and for the application to function as expected.
4.  **Code Review**:
    *   During code reviews for new TUI components and utilities, pay specific attention to import clarity and correctness.
    *   Ensure utility functions are appropriately placed in shared modules and imported cleanly.
    *   Challenge any imports that seem to violate module boundaries or rely on implicit symbol availability.

This plan will lead to a cleaner, more explicit, and more maintainable import structure for the `openhcs.tui` package, especially as it grows with the new redesigned components and utilities.