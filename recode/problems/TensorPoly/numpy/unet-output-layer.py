import numpy as np


def unet_output(features: np.ndarray, num_classes: int) -> np.ndarray:
    batch, H, W, _ = features.shape
    output = np.zeros((batch, H, W, num_classes))
    return output
