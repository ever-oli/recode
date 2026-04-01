import mlx.core as mx


def local_response_normalization(x: mx.array, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> mx.array:
    _, _, _, c = x.shape
    squared_x = x * x
    pad = n // 2
    padded_sq = mx.pad(squared_x, ((0, 0), (0, 0), (0, 0), (pad, pad)))

    sum_sq = mx.zeros_like(x)
    for i in range(n):
        sum_sq = sum_sq + padded_sq[:, :, :, i:i + c]

    scale = (k + alpha * sum_sq) ** beta
    return x / scale
