from ezstitcher.core.config import PlateConfig
from ezstitcher.core.zstack_processor import ZStackProcessor
from ezstitcher.core.stitcher import Stitcher
from ezstitcher.core.focus_analyzer import FocusAnalyzer
from ezstitcher.core.image_preprocessor import ImagePreprocessor

class PlateProcessor:
    """
    High-level orchestrator for processing a microscopy plate.
    Coordinates Z-stack handling, stitching, and output management.
    """
    def __init__(self, config: PlateConfig):
        self.config = config
        self.zstack_processor = ZStackProcessor(config.zstack)
        self.focus_analyzer = FocusAnalyzer(config.focus)
        self.image_preprocessor = ImagePreprocessor(config.stitching.preprocessing_funcs)
        self.stitcher = Stitcher(config.stitching)

    def run(self):
        """
        Run the full plate processing workflow.
        """
        # Placeholder for orchestration logic
        # 1. Detect and preprocess Z-stacks
        # 2. Select best focus or create projections
        # 3. Stitch reference plane/projection
        # 4. Optionally stitch all Z-planes
        # 5. Save outputs
        pass