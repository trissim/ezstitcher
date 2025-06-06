"""
Clean pipeline editor using unified list management architecture.
"""
import asyncio
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy

from prompt_toolkit.application import get_app
from prompt_toolkit.layout import Container, VSplit, Window
from prompt_toolkit.widgets import Label, Dialog

from openhcs.tui.components import ListManagerPane, ListConfig, ButtonConfig
from openhcs.tui.utils.dialog_helpers import show_error_dialog, prompt_for_file_dialog
from openhcs.tui.interfaces.swappable_pane import SwappablePaneInterface
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.function_step import FunctionStep
from openhcs.constants.constants import Backend
# from openhcs.core.pipeline import Pipeline # May not be needed directly as much
from openhcs.tui.services.visual_programming_dialog_service import VisualProgrammingDialogService
from openhcs.business_logic.pipeline_logic_service import PipelineLogicService

logger = logging.getLogger(__name__)


class PipelineEditorPane:
    """Clean pipeline editor using unified list management architecture."""

    def __init__(self, state, context: ProcessingContext):
        """Initialize with clean architecture."""
        self.state = state
        self.context = context
        self.steps_lock = asyncio.Lock()

        # Initialize PipelineLogicService
        self.pipeline_logic_service = PipelineLogicService(file_manager=context.filemanager)
        # self.plate_pipelines is now managed by pipeline_logic_service
        self.current_selected_plates: List[str] = []    # Currently selected plate paths
        self.pipeline_differs_across_plates: bool = False

        # EXACT: Visual programming dialog service with dependency injection
        from openhcs.tui.editors.dual_editor_pane import DualEditorPane
        self.visual_programming_service = VisualProgrammingDialogService(
            state=self.state,
            context=self.context,
            dual_editor_pane_class=DualEditorPane
        )

        # Create unified list manager with declarative config (without enabled_func initially)
        config = ListConfig(
            title="Pipeline Editor",
            frame_title="Pipeline Editor",
            button_configs=[
                ButtonConfig("Add", self._handle_add_step, width=len("Add") + 2),
                ButtonConfig("Del", self._handle_delete_step, width=len("Del") + 2),
                ButtonConfig("Edit", self._handle_edit_step, width=len("Edit") + 2),
                ButtonConfig("Load", self._handle_load_pipeline, width=len("Load") + 2),
                ButtonConfig("Save", self._handle_save_pipeline, width=len("Save") + 2),
            ],
            display_func=self._get_display_text,
            can_move_up_func=self._can_move_up,
            can_move_down_func=self._can_move_down,
            empty_message="No steps available.\n\nSelect a plate first."
        )

        self.list_manager = ListManagerPane(config, context)

        # Now set enabled functions after list_manager exists
        config.button_configs[1].enabled_func = lambda: self._has_items()  # Del
        config.button_configs[2].enabled_func = lambda: self._has_selection()  # Edit
        config.button_configs[4].enabled_func = lambda: self._has_items()  # Save
        self.list_manager._on_model_changed = self._on_selection_changed

        logger.info("PipelineEditorPane: Initialized with clean architecture")

    @classmethod
    async def create(cls, state, context: ProcessingContext):
        """Factory for backward compatibility - simplified."""
        instance = cls(state, context)
        await instance._register_observers()
        
        # Load initial display based on current state
        await instance._update_pipeline_display_for_cursor_plate()
        
        return instance

    @property
    def container(self) -> Container:
        """Get the UI container."""
        return self.list_manager.container

    def get_buttons_container(self) -> Container:
        """Return the button bar from list manager."""
        return self.list_manager.view._create_button_bar()

    # Helper methods for button state
    def _has_items(self) -> bool:
        """Check if list has items."""
        return len(self.list_manager.model.items) > 0

    def _has_selection(self) -> bool:
        """Check if there's a valid selection."""
        return self.list_manager.get_selected_item() is not None

    # Display and validation functions
    def _get_display_text(self, step_data: Dict[str, Any], is_selected: bool) -> str:
        """Generate display text for a step."""
        status_icon = self._get_status_icon(step_data.get('status', 'unknown'))
        name = step_data.get('name', 'Unknown Step')
        func_name = self._get_function_name(step_data)
        output_type = step_data.get('output_memory_type', '[N/A]')
        return f"{status_icon} {name} | {func_name} → {output_type}"

    def _can_move_up(self, index: int, step_data: Dict[str, Any]) -> bool:
        """Check if step can be moved up (within same pipeline)."""
        if index <= 0:
            return False
        items = self.list_manager.model.items
        current_pipeline = step_data.get('pipeline_id')
        prev_pipeline = items[index - 1].get('pipeline_id')
        return current_pipeline == prev_pipeline

    def _can_move_down(self, index: int, step_data: Dict[str, Any]) -> bool:
        """Check if step can be moved down (within same pipeline)."""
        items = self.list_manager.model.items
        if index >= len(items) - 1:
            return False
        current_pipeline = step_data.get('pipeline_id')
        next_pipeline = items[index + 1].get('pipeline_id')
        return current_pipeline == next_pipeline

    def _get_status_icon(self, status: str) -> str:
        """Get status icon for a step."""
        icons = {
            'pending': "?", 'validated': "o", 'error': "!",
            'not_initialized': "?", 'initialized': "!", 'ready': "!",
            'compiled_ok': "o", 'compiled': "o", 'running': "o"
        }
        return icons.get(status, "o")

    def _get_function_name(self, step: Dict[str, Any]) -> str:
        """Get display name for function pattern."""
        if 'func' not in step:
            return '[MISSING FUNC]'
        
        func = step['func']
        if func is None:
            return '[NULL FUNC]'
        
        if callable(func):
            return func.__name__
        elif isinstance(func, tuple) and len(func) == 2 and callable(func[0]):
            return f"{func[0].__name__}(...)"
        elif isinstance(func, list):
            return f"[{len(func)} functions]"
        elif isinstance(func, dict):
            return f"{{{len(func)} components}}"
        else:
            return str(func)

    # Event handling
    async def _register_observers(self):
        """Register observers for state integration."""
        try:
            from openhcs.tui.utils.unified_task_manager import get_task_manager

            self.state.add_observer('plate_selected',
                lambda plate: get_task_manager().fire_and_forget(self._on_plate_selected(plate), "pipeline_plate_selected"))
            self.state.add_observer('steps_updated',
                lambda _: get_task_manager().fire_and_forget(self._refresh_steps(), "pipeline_steps_updated"))
            self.state.add_observer('step_pattern_saved',
                lambda data: get_task_manager().fire_and_forget(self._handle_step_pattern_saved(data), "pipeline_step_pattern_saved"))
            logger.info("PipelineEditorPane: Observers registered")
        except Exception as e:
            logger.error(f"Error registering observers: {e}", exc_info=True)

    def _on_selection_changed(self):
        """Handle selection changes."""
        # CRITICAL: Call the original method to trigger UI invalidation
        from prompt_toolkit.application import get_app
        get_app().invalidate()

        # Then handle our business logic
        selected_item = self.list_manager.get_selected_item()
        if selected_item:
            self.state.set_selected_step(selected_item)

    async def _on_plate_selected(self, plate):
        """Handle plate selection event."""
        if plate:
            await self._update_pipeline_display_for_cursor_plate()

    async def _refresh_steps(self, _=None):
        """Refresh the step list."""
        await self._update_pipeline_display_for_cursor_plate()

    async def _update_pipeline_display_for_cursor_plate(self):
        """Update pipeline display based on cursor position and orchestrator state."""
        # Check if we have an active orchestrator (set by PlateManager)
        if not hasattr(self.state, 'active_orchestrator') or not self.state.active_orchestrator:
            # No orchestrator selected - clear everything
            self.current_selected_plates = []
            self.list_manager.load_items([])
            return

        orchestrator = self.state.active_orchestrator

        # Update current_selected_plates for button handlers
        if hasattr(self.state, 'selected_plate') and self.state.selected_plate:
            plate_path = self.state.selected_plate['path']
            self.current_selected_plates = [plate_path]
        else:
            self.current_selected_plates = []

        # Check if orchestrator is initialized
        if not orchestrator.is_initialized():
            # Orchestrator exists but not initialized - show message
            self.list_manager.load_items([{
                'id': 'not_initialized',
                'name': 'Plate not initialized',
                'func': 'Click "Init" button to initialize this plate before editing pipeline',
                'status': 'not_initialized',
                'pipeline_id': None,
                'output_memory_type': '[N/A]'
            }])
            return

        # Orchestrator is initialized - check if we have a pipeline for this plate
        # Pipeline definitions are stored separately, not in the orchestrator
        if hasattr(self.state, 'selected_plate') and self.state.selected_plate:
            plate_path = self.state.selected_plate['path']
            pipeline = self.pipeline_logic_service.get_pipeline(plate_path)

            if pipeline and len(pipeline) > 0:
                # Convert Pipeline (list of steps) to display format
                steps = [self._transform_step_to_dict(step) for step in pipeline
                        if isinstance(step, FunctionStep)]
                self.list_manager.load_items(steps)
            else:
                # Initialized but no pipeline steps yet
                self.list_manager.load_items([{
                    'id': 'no_steps',
                    'name': 'No pipeline steps',
                    'func': 'Click "Add" button to add steps to this pipeline',
                    'status': 'ready',
                    'pipeline_id': None,
                    'output_memory_type': '[N/A]'
                }])
        else:
            # No plate selected
            self.list_manager.load_items([])

    # Step loading
    async def _load_steps_for_plate(self, plate_id: str):
        """Load steps for the specified plate."""
        async with self.steps_lock:
            # Try orchestrator first, fallback to context
            raw_steps = self._get_orchestrator_steps()
            if raw_steps:
                steps = [self._transform_step_to_dict(step) for step in raw_steps 
                        if isinstance(step, FunctionStep)]
            else:
                steps = self.context.list_steps_for_plate(plate_id)
            
            self.list_manager.load_items(steps)

    def _get_orchestrator_steps(self) -> List[Any]:
        """Get step objects from stored pipeline (orchestrator doesn't store pipeline_definition)."""
        if hasattr(self.state, 'selected_plate') and self.state.selected_plate:
            plate_path = self.state.selected_plate['path']
            pipeline = self.pipeline_logic_service.get_pipeline(plate_path)
            return list(pipeline) if pipeline else []
        return []

    def _transform_step_to_dict(self, step_obj: FunctionStep) -> Dict[str, Any]:
        """Transform FunctionStep to display dictionary."""
        return {
            'id': step_obj.step_id,
            'name': step_obj.name,
            'func': step_obj.func,
            'status': 'pending',
            'pipeline_id': getattr(step_obj, 'pipeline_id', None),
            'output_memory_type': getattr(step_obj, 'output_memory_type', '[N/A]')
        }

    # Action handlers
    async def _handle_add_step(self):
        """EXACT: Add step handler with visual programming integration."""
        if not self.current_selected_plates:
            await show_error_dialog(
                "No Plate Selected",
                "Please select a plate to add steps to its pipeline.",
                self.state
            )
            return

        if self.pipeline_differs_across_plates:
            await show_error_dialog(
                "Multiple Different Pipelines",
                "Cannot add steps when selected plates have different pipelines. Please select plates with identical pipelines or edit them individually.",
                self.state
            )
            return

        # The visual_programming_service.show_add_step_dialog might need adjustment
        # if it relies on direct Pipeline objects for its `target_pipelines` argument.
        # For now, we assume it can work without direct pipeline objects or can be adapted.
        # It's more likely it needs some context or identifiers.
        # Let's assume it returns a FunctionStep instance or None.

        # For the purpose of this refactor, we'll assume `show_add_step_dialog`
        # might not need `target_pipelines` or this part of its usage will change.
        # The core task is to get a `created_step`.
        # If `show_add_step_dialog` *must* have the pipeline objects, we'd fetch them:
        # target_pipelines_objects = [self.pipeline_logic_service.get_or_create_pipeline(pp) for pp in self.current_selected_plates]
        # created_step = await self.visual_programming_service.show_add_step_dialog(target_pipelines_objects)
        # This detail depends on visual_programming_service's interface.
        # For now, simplifying to focus on pipeline_logic_service integration:

        created_step = await self.visual_programming_service.show_add_step_dialog(
            target_pipelines=self.current_selected_plates # Or adjust based on service needs
        )

        if created_step:
            if not isinstance(created_step, FunctionStep):
                await show_error_dialog("Add Step Error", "Failed to create a valid step.", self.state)
                return

            for plate_path in self.current_selected_plates:
                try:
                    # The service handles pipeline creation if it doesn't exist.
                    self.pipeline_logic_service.add_step(plate_path, created_step)
                except ValueError as ve: # Catch if step is not FunctionStep (already checked but good practice)
                    await show_error_dialog("Add Step Error", str(ve), self.state)
                    return # Stop if one fails
                except Exception as e:
                    await show_error_dialog("Add Step Error", f"Error adding step to {plate_path}: {e}", self.state)
                    return # Stop if one fails

            logger.info(f"Added step {created_step.name} to pipelines for plates: {self.current_selected_plates}")
            await self._update_pipeline_display_for_selection(self.current_selected_plates)

    async def _handle_delete_step(self):
        """EXACT: Delete step handler with multi-plate support."""
        if not self.current_selected_plates:
            await show_error_dialog(
                "No Plate Selected",
                "Please select a plate to delete steps from its pipeline.",
                self.state
            )
            return

        if self.pipeline_differs_across_plates:
            await show_error_dialog(
                "Multiple Different Pipelines",
                "Cannot delete steps when selected plates have different pipelines. Please select plates with identical pipelines or edit them individually.",
                self.state
            )
            return

        selected_step = self.list_manager.get_selected_item()
        if not selected_step:
            await show_error_dialog("No Selection", "Please select a step to delete.", self.state)
            return

        step_id = selected_step.get('id')
        if not step_id:
            await show_error_dialog("Delete Error", "Selected step has no ID.", self.state)
            return

        # Remove step from all target pipelines using the service
        deleted_in_at_least_one_pipeline = False
        for plate_path in self.current_selected_plates:
            try:
                if self.pipeline_logic_service.delete_step(plate_path, step_id):
                    deleted_in_at_least_one_pipeline = True
            except Exception as e:
                # Log error for this specific plate but continue with others
                logger.error(f"Error deleting step {step_id} from pipeline for plate {plate_path}: {e}")
                await show_error_dialog("Delete Error", f"Error deleting step from {plate_path}: {e}", self.state)
                # Depending on desired behavior, might want to stop or collect errors

        if deleted_in_at_least_one_pipeline:
            logger.info(f"Attempted deletion of step {step_id} from pipelines for plates: {self.current_selected_plates}")
            await self._update_pipeline_display_for_selection(self.current_selected_plates)
        else:
            # This message might be misleading if some deletions failed due to exceptions
            # but others succeeded, or if the step just wasn't found.
            # The service logs a warning if step not found.
            await show_error_dialog("Delete Error", f"Step {step_id} not found or not deleted in any selected pipeline.", self.state)

    async def _handle_edit_step(self):
        """EXACT: Edit step handler with visual programming integration."""
        if not self.current_selected_plates:
            await show_error_dialog(
                "No Plate Selected",
                "Please select a plate to edit steps in its pipeline.",
                self.state
            )
            return

        if self.pipeline_differs_across_plates:
            await show_error_dialog(
                "Multiple Different Pipelines",
                "Cannot edit steps when selected plates have different pipelines. Please select plates with identical pipelines or edit them individually.",
                self.state
            )
            return

        selected_step = self.list_manager.get_selected_item()
        if not selected_step:
            await show_error_dialog("No Selection", "Please select a step to edit.", self.state)
            return

        step_id = selected_step.get('id')
        if not step_id:
            await show_error_dialog("Edit Error", "Selected step has no ID.", self.state)
            return

        # Find the original FunctionStep object to pass to the dialog
        # This requires getting the pipeline from the service first.
        first_plate_path = self.current_selected_plates[0]
        pipeline = self.pipeline_logic_service.get_pipeline(first_plate_path)

        if not pipeline:
            await show_error_dialog("Edit Error", "No pipeline found for selected plate.", self.state)
            return

        original_step_object = None
        for step_in_pipeline in pipeline:
            # Assuming step_id from list_manager matches step_in_pipeline.step_id
            if step_in_pipeline.step_id == step_id:
                original_step_object = step_in_pipeline
                break

        if not original_step_object:
            await show_error_dialog("Edit Error", f"Step {step_id} not found in the actual pipeline.", self.state)
            return

        # Launch visual programming dialog for editing using service, passing the actual step object
        edited_step = await self.visual_programming_service.show_edit_step_dialog(original_step_object)

        if edited_step:
            if not isinstance(edited_step, FunctionStep):
                await show_error_dialog("Edit Error", "Editing did not return a valid step.", self.state)
                return

            updated_in_at_least_one_pipeline = False
            for plate_path in self.current_selected_plates:
                try:
                    # Use original_step_object.step_id for identifying the step to update.
                    # The service will replace the step found by this ID with edited_step.
                    if self.pipeline_logic_service.update_step(plate_path, original_step_object.step_id, edited_step):
                        updated_in_at_least_one_pipeline = True
                except ValueError as ve: # Catch if edited_step is not FunctionStep
                    await show_error_dialog("Edit Error", str(ve), self.state)
                    # Decide if to stop or continue for other plates
                except Exception as e:
                    logger.error(f"Error updating step {original_step_object.step_id} in pipeline for plate {plate_path}: {e}")
                    await show_error_dialog("Edit Error", f"Error updating step in {plate_path}: {e}", self.state)

            if updated_in_at_least_one_pipeline:
                logger.info(f"Attempted update of step {original_step_object.step_id} in pipelines for plates: {self.current_selected_plates}")
                await self._update_pipeline_display_for_selection(self.current_selected_plates)
            else:
                # Service logs warning if step not found for update.
                await show_error_dialog("Edit Error", f"Step {original_step_object.step_id} not updated in any selected pipeline.", self.state)

    async def _handle_load_pipeline(self):
        """EXACT: Load pipeline handler with multi-plate support."""
        if not self.current_selected_plates:
            await show_error_dialog(
                "No Plate Selected",
                "Please select plates to load pipeline into.",
                self.state
            )
            return

        file_path = await prompt_for_file_dialog(
            title="Load Pipeline",
            prompt_message="Select pipeline file:",
            app_state=self.state,
            filemanager=self.context.filemanager,
            selection_mode="files",
            filter_extensions=[".pipeline"]
        )

        if not file_path:
            return

        try:
            # Load and apply pipeline for each selected plate using the service
            for plate_path in self.current_selected_plates:
                # The service method handles loading, associating, and storing the pipeline.
                # It will raise an exception on failure, which is caught below.
                self.pipeline_logic_service.load_pipeline_from_file(plate_path, file_path)

            # EXACT: Refresh display (assuming _update_pipeline_display_for_selection is still valid
            # or can be adapted to use the service for fetching pipelines if needed for refresh)
            await self._update_pipeline_display_for_selection(self.current_selected_plates)
            logger.info(f"Loaded pipeline from {file_path} into {len(self.current_selected_plates)} plate(s)")

        except Exception as e:
            await show_error_dialog("Load Error", f"Failed to load pipeline: {str(e)}", self.state)

    async def _handle_save_pipeline(self):
        """EXACT: Save pipeline handler with multi-plate support."""
        if not self.current_selected_plates:
            await show_error_dialog(
                "No Plate Selected",
                "Please select a plate to save its pipeline.",
                self.state
            )
            return

        if self.pipeline_differs_across_plates:
            await show_error_dialog(
                "Multiple Different Pipelines",
                "Cannot save when selected plates have different pipelines. Please select plates with identical pipelines or save them individually.",
                self.state
            )
            return

        # Get pipeline to save (from first selected plate)
        first_plate_path = self.current_selected_plates[0]
        # Pipeline itself is fetched by the service

        file_path = await prompt_for_file_dialog(
            title="Save Pipeline",
            prompt_message="Select save location:",
            app_state=self.state,
            filemanager=self.context.filemanager, # Still needed for dialog
            selection_mode="files",
            filter_extensions=[".pipeline"]
        )

        if not file_path:
            return

        try:
            # The service method handles fetching the pipeline and saving it.
            # It will raise FileNotFoundError if no pipeline, or other errors on save failure.
            self.pipeline_logic_service.save_pipeline_to_file(first_plate_path, file_path)
            # Logger info is now inside the service method.
            # We might want a success message here if desired.
            # For example:
            # await show_message_dialog("Success", f"Pipeline saved to {file_path}", self.state)
            logger.info(f"Call to save pipeline for {first_plate_path} to {file_path} completed.")

        except FileNotFoundError:
            await show_error_dialog("Save Error", "No pipeline steps to save for the selected plate.", self.state)
        except Exception as e:
            await show_error_dialog("Save Error", f"Failed to save pipeline: {str(e)}", self.state)







    async def shutdown(self):
        """Cleanup observers."""
        logger.info("PipelineEditorPane: Shutting down")
        # No observers to remove - they're handled automatically by the state system
        logger.info("PipelineEditorPane: Shutdown complete")
