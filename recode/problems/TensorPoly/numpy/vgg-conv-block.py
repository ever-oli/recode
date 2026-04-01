import numpy as np


def vgg_conv_block(x: np.ndarray, num_convs: int, out_channels: int) -> np.ndarray:
    current_x = x

    for _ in range(num_convs):
        in_channels = current_x.shape[-1]
        limit = np.sqrt(2 / (3 * 3 * in_channels))
        weights = np.random.randn(3, 3, in_channels, out_channels) * limit
        bias = np.zeros(out_channels)

        padded_x = np.pad(current_x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
        batch, h, w, _ = current_x.shape
        out = np.zeros((batch, h, w, out_channels))

        for i in range(3):
            for j in range(3):
                window = padded_x[:, i:i + h, j:j + w, :]
                out += np.tensordot(window, weights[i, j], axes=([-1], [0]))

        out += bias
        current_x = np.maximum(0, out)

    return current_x
