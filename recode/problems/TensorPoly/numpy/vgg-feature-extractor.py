import numpy as np


def conv_relu(x, out_channels):
    _, _, _, C = x.shape
    W_weights = np.random.randn(C, out_channels) * 0.1
    x = x @ W_weights
    return np.maximum(0, x)


def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))


def vgg_features(x: np.ndarray, config: list) -> np.ndarray:
    out = x

    for layer in config:
        if isinstance(layer, int):
            out = conv_relu(out, layer)
        elif layer == "M":
            out = maxpool_2x2(out)

    return out
