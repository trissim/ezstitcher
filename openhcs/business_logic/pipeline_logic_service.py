import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from openhcs.core.pipeline import Pipeline
from openhcs.core.steps.function_step import FunctionStep
from openhcs.io.filemanager import FileManager # Assuming FileManager is the correct class
# from openhcs.constants.constants import Backend # If needed for filemanager operations

logger = logging.getLogger(__name__)

class PipelineLogicService:
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
        self.plate_pipelines: Dict[str, Pipeline] = {}

    def get_pipeline(self, plate_path: str) -> Optional[Pipeline]:
        """Retrieves the pipeline for a given plate path."""
        return self.plate_pipelines.get(plate_path)

    def get_or_create_pipeline(self, plate_path: str, pipeline_name: Optional[str] = None) -> Pipeline:
        """Gets an existing pipeline or creates a new one if it doesn't exist."""
        pipeline = self.plate_pipelines.get(plate_path)
        if not pipeline:
            name = pipeline_name or f"Pipeline for {Path(plate_path).name}"
            pipeline = Pipeline(name=name)
            self.plate_pipelines[plate_path] = pipeline
            logger.info(f"Created new pipeline for plate: {plate_path}")
        return pipeline

    def add_step(self, plate_path: str, step: FunctionStep, pipeline_name: Optional[str] = None) -> None:
        """Adds a step to the pipeline of the given plate path.
        Creates the pipeline if it doesn't exist.
        """
        # Ensure step is a FunctionStep instance, not just data
        if not isinstance(step, FunctionStep):
            # This check might be too strict if step creation is complex and
            # the service is meant to receive data and construct the step.
            # For now, assume TUI layer provides the FunctionStep instance.
            raise ValueError("Step must be an instance of FunctionStep.")

        pipeline = self.get_or_create_pipeline(plate_path, pipeline_name)
        pipeline.append(copy.deepcopy(step)) # Deepcopy to ensure pipeline owns its steps
        logger.info(f"Added step {step.name} to pipeline for plate: {plate_path}")

    def delete_step(self, plate_path: str, step_id: str) -> bool:
        """Deletes a step from the pipeline of the given plate path by step_id.
        Returns True if a step was deleted, False otherwise.
        """
        pipeline = self.get_pipeline(plate_path)
        if not pipeline:
            logger.warning(f"Pipeline not found for plate {plate_path} when trying to delete step {step_id}")
            return False

        initial_len = len(pipeline)
        # FunctionStep.step_id is a property that uses id(self) by default.
        # The step_id received here would be from the TUI representation.
        # We need to ensure FunctionStep instances in the pipeline have a persistent ID
        # or can be matched. The current PipelineEditorPane uses str(id(step_obj)).
        # Let's assume step_id is FunctionStep.step_id for now.
        new_steps = [s for s in pipeline if s.step_id != step_id]

        if len(new_steps) < initial_len:
            pipeline.steps = new_steps # Assuming Pipeline.steps can be directly set or use a method
            logger.info(f"Deleted step {step_id} from pipeline for plate: {plate_path}")
            return True
        else:
            logger.warning(f"Step {step_id} not found in pipeline for plate {plate_path}")
            return False

    def update_step(self, plate_path: str, step_id: str, updated_step: FunctionStep) -> bool:
        """Updates an existing step in the pipeline of the given plate path.
        Returns True if a step was updated, False otherwise.
        """
        # Ensure updated_step is a FunctionStep instance
        if not isinstance(updated_step, FunctionStep):
            raise ValueError("Updated step must be an instance of FunctionStep.")

        pipeline = self.get_pipeline(plate_path)
        if not pipeline:
            logger.warning(f"Pipeline not found for plate {plate_path} when trying to update step {step_id}")
            return False

        for i, existing_step in enumerate(pipeline):
            if existing_step.step_id == step_id:
                # Ensure the new step carries the same ID if it's intrinsic
                if updated_step.step_id != step_id:
                     # This might indicate an issue or a need to re-assign ID.
                     # For now, we assume the TUI ensures the updated_step has the correct ID.
                     logger.warning(f"Updating step {step_id} with a FunctionStep object that has a different internal step_id ({updated_step.step_id}). This might be problematic.")

                pipeline[i] = copy.deepcopy(updated_step) # Deepcopy new step
                logger.info(f"Updated step {step_id} in pipeline for plate: {plate_path}")
                return True

        logger.warning(f"Step {step_id} not found for update in pipeline for plate {plate_path}")
        return False

    def load_pipeline_from_file(self, plate_path: str, file_path: str) -> Pipeline:
        """Loads a pipeline from a file and associates it with the given plate path.
        The loaded data is expected to be a list of FunctionStep objects or data
        that can be reconstituted into FunctionStep objects.
        """
        try:
            # Assuming filemanager.load returns data that can be used to build a Pipeline
            # The exact structure of loaded_data needs to match what Pipeline expects
            # or be processed here.
            # PipelineEditorPane uses: loaded_data = self.context.filemanager.load(file_path, Backend.DISK.value)
            #                           loaded_pipeline.extend(loaded_data)
            # This implies loaded_data is a list of FunctionStep compatible items.
            from openhcs.constants.constants import Backend # Ensure Backend is imported
            loaded_step_data_list = self.file_manager.load(file_path, Backend.DISK.value)

            if not isinstance(loaded_step_data_list, list):
                raise ValueError(f"Invalid pipeline format in {file_path} - must be a list of steps.")

            pipeline_name = f"Loaded from {Path(file_path).name}"
            new_pipeline = Pipeline(name=pipeline_name)

            # Reconstitute FunctionStep objects if necessary
            # For now, assume items in loaded_step_data_list are already FunctionStep instances
            # or can be directly used by Pipeline.extend()
            # This part is critical and depends on the serialization format.
            # If they are dicts, they need to be converted to FunctionStep objects.
            # Let's assume for now they are FunctionStep objects as per current PipelineEditorPane logic.
            # This might require FunctionStep to be serializable/deserializable or created from dicts.

            # TODO: Clarify if FunctionStep objects are directly stored or if they need reconstitution
            # For now, mirroring PipelineEditorPane:
            # new_pipeline.extend(loaded_step_data_list)

            # A safer approach if FunctionStep needs args:
            reconstituted_steps = []
            for step_data_or_obj in loaded_step_data_list:
                if isinstance(step_data_or_obj, FunctionStep):
                    reconstituted_steps.append(step_data_or_obj)
                elif isinstance(step_data_or_obj, dict):
                    # This assumes FunctionStep can be created from a dict.
                    # This is a placeholder for actual reconstitution logic.
                    # E.g., func_ref = step_data_or_obj.get('func') # Needs to resolve to callable
                    # name = step_data_or_obj.get('name')
                    # kwargs = step_data_or_obj.get('kwargs', {})
                    # reconstituted_steps.append(FunctionStep(func=func_ref, name=name, **kwargs))
                    # This part is complex due to resolving function references.
                    # For now, we stick to the assumption that loaded_data contains FunctionStep instances
                    # or something Pipeline.extend can handle directly.
                    pass # Add actual reconstitution if needed
                else:
                    logger.warning(f"Skipping unrecognized item in loaded pipeline data: {type(step_data_or_obj)}")

            if not reconstituted_steps and loaded_step_data_list: # if list wasn't empty but we couldn't process it
                 logger.warning(f"Pipeline data from {file_path} could not be fully processed into FunctionSteps. Assuming direct extendability.")
                 new_pipeline.extend(loaded_step_data_list) # Fallback to current behavior
            else:
                new_pipeline.extend(reconstituted_steps)


            self.plate_pipelines[plate_path] = new_pipeline
            logger.info(f"Loaded pipeline from {file_path} into plate: {plate_path}")
            return new_pipeline
        except Exception as e:
            logger.error(f"Failed to load pipeline for plate {plate_path} from {file_path}: {e}")
            raise # Re-raise after logging

    def save_pipeline_to_file(self, plate_path: str, file_path: str) -> None:
        """Saves the pipeline for the given plate path to a file."""
        pipeline = self.get_pipeline(plate_path)
        if not pipeline:
            raise FileNotFoundError(f"No pipeline found for plate {plate_path} to save.")

        if len(pipeline) == 0:
            logger.warning(f"Pipeline for plate {plate_path} is empty. Saving an empty list.")

        # Pipeline object is iterable (yields steps). Convert to list for saving.
        # This assumes FunctionStep objects are serializable by the filemanager
        # or that filemanager handles FunctionStep objects correctly.
        # TODO: Verify serialization of FunctionStep (especially the 'func' callable).
        # Pickle is used in some places; if so, this should work.
        pipeline_data = list(pipeline)

        try:
            from openhcs.constants.constants import Backend # Ensure Backend is imported
            self.file_manager.save(pipeline_data, file_path, Backend.DISK.value)
            logger.info(f"Saved pipeline for plate {plate_path} to {file_path} ({len(pipeline_data)} steps)")
        except Exception as e:
            logger.error(f"Failed to save pipeline for plate {plate_path} to {file_path}: {e}")
            raise # Re-raise after logging

    def clear_pipeline(self, plate_path: str) -> None:
        """Clears the pipeline for a given plate path."""
        if plate_path in self.plate_pipelines:
            del self.plate_pipelines[plate_path]
            logger.info(f"Cleared pipeline for plate: {plate_path}")

    # Additional methods might be needed for reordering steps, etc.
