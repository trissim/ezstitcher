from ezstitcher.core.config import FocusConfig

class FocusAnalyzer:
    """
    Provides focus metrics and best focus selection.
    """
    def __init__(self, config: FocusConfig):
        self.config = config

    def compute_focus_metrics(self, image_stack):
        """
        Compute focus metrics for a stack of images.
        """
        pass

    def select_best_focus(self, image_stack):
        """
        Select the best focus plane from a stack of images.
        """
        pass
