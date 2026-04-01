import numpy as np


def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    batch_size = image.shape[0]
    output_h = 55
    output_w = 55
    num_filters = 96
    return np.zeros((batch_size, output_h, output_w, num_filters))
