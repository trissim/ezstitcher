# Diff 4: Refactor PipelineOrchestrator

Status: Not Started
Progress: 0%
Last Updated: 2023-05-15
Dependencies: [plans/refactor/05-diff-03-strategies.md]

## Overview

This diff refactors the `PipelineOrchestrator` class to use the new pipeline and strategy classes. The `process_reference_images` and `process_final_images` methods are updated to use the new approach, and the `process_well` method is updated to handle the return values from these methods.

## Files to Modify

### 1. `ezstitcher/core/processing_pipeline.py`

```diff
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable, Any

from ezstitcher.core.microscope_interfaces import create_microscope_handler
from ezstitcher.core.image_locator import ImageLocator
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.file_system_manager import FileSystemManager
from ezstitcher.core.image_preprocessor import ImagePreprocessor
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.config import PipelineConfig, FocusAnalyzerConfig
+ from ezstitcher.core.processing.context import ProcessingContext
+ from ezstitcher.core.processing.strategies import ReferenceProcessingStrategy, FinalProcessingStrategy

logger = logging.getLogger(__name__)

DEFAULT_PADDING = 3

class PipelineOrchestrator:
    """Orchestrates the complete image processing and stitching pipeline."""

    def __init__(self, config=None, fs_manager=None, image_preprocessor=None, focus_analyzer=None):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Pipeline configuration
            fs_manager: File system manager
            image_preprocessor: Image preprocessor
            focus_analyzer: Focus analyzer
        """
        self.config = config or PipelineConfig()
        self.fs_manager = fs_manager or FileSystemManager()
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()

        # Initialize focus analyzer
        focus_config = self.config.focus_config or FocusAnalyzerConfig(method=self.config.focus_method)
        self.focus_analyzer = focus_analyzer or FocusAnalyzer(focus_config)

        self.microscope_handler = None
        self.stitcher = None

    # ... (other methods remain unchanged)

    def process_well(self, well, dirs):
        """
        Process a single well through the pipeline.

        Args:
            well: Well identifier
            dirs: Dictionary of directories
        """
        logger.info("Processing well %s", well)

        # 1. Process reference images (for position generation)
-       self.process_reference_images(well, dirs)
+       reference_context = self.process_reference_images(well, dirs)

        # 2. Generate stitching positions
        positions_file, stitch_pattern = self.generate_positions(well, dirs)

        # 3. Process final images (for stitching)
-       self.process_final_images(well, dirs)
+       final_context = self.process_final_images(well, dirs)

        # 4. Stitch final images
        self.stitch_images(well, dirs, positions_file)

-   def process_reference_images(self, well, dirs):
+   def process_reference_images(self, well, dirs) -> ProcessingContext:
        """
        Process images for position generation.

        Args:
            well: Well identifier
            dirs: Dictionary of directories
+
+       Returns:
+           ProcessingContext: Processing context with results
        """
        logger.info("Processing reference images for well %s", well)

-       # Determine which channels to use as reference
-       reference_channels = self.config.reference_channels
-
-       # Get reference processing functions from config
-       processing_funcs = {}
-       for channel in reference_channels:
-           channel_funcs = self._get_processing_functions(
-               getattr(self.config, 'reference_processing', None),
-               channel
-           )
-           if channel_funcs:
-               processing_funcs[channel] = channel_funcs
-
-
-       # Flatten Z-stacks if needed - use create_projection directly
-       self.process_patterns_with_variable_components(
-           input_dir=dirs['input'],
-           output_dir=dirs['processed'],
-           well_filter=[well],
-           variable_components=['z_index'],
-           processing_funcs=self.image_preprocessor.create_projection,
-           processing_args={
-               'method': self.config.reference_flatten,
-               'focus_analyzer': self.focus_analyzer
-           }
-       ).get(well, [])
-
-       # Process reference images
-       self.process_patterns_with_variable_components(
-           input_dir=dirs['processed'],
-           output_dir=dirs['processed'],
-           well_filter=[well],
-           variable_components=['site'],
-           group_by='channel',
-           processing_funcs=processing_funcs
-       ).get(well, [])
-
-       # Create composites in one step
-       self.process_patterns_with_variable_components(
-           input_dir=dirs['processed'],
-           output_dir=dirs['processed'],
-           well_filter=[well],
-           variable_components=['channel'],
-           group_by='site',
-           processing_funcs=self.image_preprocessor.create_composite,
-           processing_args={'weights': self.config.reference_composite_weights}
-       ).get(well, [])
+       # Create context
+       context = ProcessingContext(
+           pipeline_orchestrator=self,
+           well=well,
+           dirs=dirs,
+           channels=self.config.reference_channels
+       )
+
+       # Create strategy
+       strategy = ReferenceProcessingStrategy(
+           config=self.config,
+           image_preprocessor=self.image_preprocessor,
+           focus_analyzer=self.focus_analyzer
+       )
+
+       # Create and run pipeline
+       pipeline = strategy.create_pipeline(context)
+       return pipeline.process(context)

-   def process_final_images(self, well, dirs):
+   def process_final_images(self, well, dirs) -> ProcessingContext:
        """
        Process images for final stitching.

        Args:
            well: Well identifier
            dirs: Dictionary of directories
+
+       Returns:
+           ProcessingContext: Processing context with results
        """
        logger.info("Processing final images for well %s", well)

        # Get all available channels
        channels = self._get_available_channels(dirs['input'], well)
        logger.info("Processing all %d available channels for well %s", len(channels), well)

-       # Get final processing functions from config
-       processing_funcs = {}
-       for channel in channels:
-           channel_funcs = self._get_processing_functions(
-               getattr(self.config, 'final_processing', None),
-               channel
-           )
-           if channel_funcs:
-               processing_funcs[channel] = channel_funcs
-           else:
-               processing_funcs[channel] = []
-
-       # Process final images
-       self.process_patterns_with_variable_components(
-           input_dir=dirs['input'],
-           output_dir=dirs['post_processed'],
-           well_filter=[well],
-           variable_components=['site'],
-           processing_funcs=processing_funcs
-       ).get(well, [])
-
-       # Flatten Z-stacks if needed - use create_projection directly
-       self.process_patterns_with_variable_components(
-           input_dir=dirs['post_processed'],
-           output_dir=dirs['post_processed'],
-           well_filter=[well],
-           variable_components=['z_index'],
-           processing_funcs=self.image_preprocessor.create_projection,
-           processing_args={
-               'method': self.config.stitch_flatten,
-               'focus_analyzer': self.focus_analyzer
-           }
-       ).get(well, [])
+       # Create context
+       context = ProcessingContext(
+           pipeline_orchestrator=self,
+           well=well,
+           dirs=dirs,
+           channels=channels
+       )
+
+       # Create strategy
+       strategy = FinalProcessingStrategy(
+           config=self.config,
+           image_preprocessor=self.image_preprocessor,
+           focus_analyzer=self.focus_analyzer
+       )
+
+       # Create and run pipeline
+       pipeline = strategy.create_pipeline(context)
+       return pipeline.process(context)

    # ... (other methods remain unchanged)
```

## Implementation Notes

1. The `process_reference_images` method is refactored to use the `ReferenceProcessingStrategy` and `ProcessingContext` classes. It now returns a `ProcessingContext` object with the results.

2. The `process_final_images` method is refactored to use the `FinalProcessingStrategy` and `ProcessingContext` classes. It now returns a `ProcessingContext` object with the results.

3. The `process_well` method is updated to handle the return values from the `process_reference_images` and `process_final_images` methods.

4. The imports are updated to include the new classes.

5. The method signatures are updated to include return type annotations.

## Testing Plan

1. Update the existing tests for `PipelineOrchestrator` to work with the refactored code:
   - Test that `process_reference_images` returns a `ProcessingContext` object with the expected results
   - Test that `process_final_images` returns a `ProcessingContext` object with the expected results
   - Test that `process_well` correctly handles the return values from these methods

2. Run the existing integration tests to ensure that the refactored code works correctly with the rest of the system.

## Validation Criteria

1. All unit tests pass
2. All integration tests pass
3. The refactored code produces the same results as the original code
4. The code is more modular, flexible, and testable
