import numpy as np


def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    batch, h, w, c = x.shape
    reshaped_x = x.reshape(batch, h // 2, 2, w // 2, 2, c)
    return reshaped_x.max(axis=(2, 4))
