from ezstitcher.core.config import StitchingConfig

class Stitcher:
    """
    Handles 2D stitching of single planes or projections.
    Encapsulates stitching backend logic.
    """
    def __init__(self, config: StitchingConfig):
        self.config = config

    def stitch(self, image_dir: str, output_dir: str):
        """
        Perform stitching on images in the given directory.
        """
        pass
