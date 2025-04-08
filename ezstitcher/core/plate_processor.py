import logging
from pathlib import Path

from ezstitcher.core.config import PlateProcessorConfig
from ezstitcher.core.zstack_processor import ZStackProcessor
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)

class PlateProcessor:
    """
    High-level orchestrator for processing a microscopy plate.
    Coordinates Z-stack handling, stitching, and output management.
    """
    def __init__(self, config: PlateProcessorConfig):
        self.config = config
        self.zstack_processor = ZStackProcessor(config.z_stack_processor)
        self.focus_analyzer = FocusAnalyzer(config.focus_analyzer)
        self.image_preprocessor = ImagePreprocessor(config.image_preprocessor)
        self.stitcher = Stitcher(config.stitcher)

    def run(self, plate_folder):
        """
        Run the full plate processing workflow.

        Args:
            plate_folder (str or Path): Path to the plate folder

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            plate_folder = Path(plate_folder)
            logger.info(f"Processing plate folder: {plate_folder}")

            # 1. Detect and preprocess Z-stacks
            has_zstack = self.zstack_processor.detect_z_stacks(plate_folder)
            logger.info(f"Z-stack detection result: {has_zstack}")

            # TODO: Implement the full workflow
            # 2. Select best focus or create projections if needed
            # 3. Stitch reference plane/projection
            # 4. Optionally stitch all Z-planes
            # 5. Save outputs

            # For now, return True to indicate success
            return True

        except Exception as e:
            logger.error(f"Error in PlateProcessor.run: {e}", exc_info=True)
            return False