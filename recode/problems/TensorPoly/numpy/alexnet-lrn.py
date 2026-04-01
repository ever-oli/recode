import numpy as np


def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    batch_size, h, w, c = x.shape
    squared_x = np.square(x)
    pad = n // 2
    padded_sq = np.pad(squared_x, ((0, 0), (0, 0), (0, 0), (pad, pad)), mode="constant")

    sum_sq = np.zeros_like(x)
    for i in range(n):
        sum_sq += padded_sq[:, :, :, i:i + c]

    scale = (k + alpha * sum_sq) ** beta
    return x / scale
