class ImagePreprocessor:
    """
    Handles image normalization, filtering, and compositing.
    """
    def __init__(self, preprocessing_funcs=None):
        self.preprocessing_funcs = preprocessing_funcs or {}

    def preprocess(self, image, channel: str):
        """
        Apply preprocessing to a single image for a given channel.
        """
        func = self.preprocessing_funcs.get(channel)
        if func:
            return func(image)
        return image
