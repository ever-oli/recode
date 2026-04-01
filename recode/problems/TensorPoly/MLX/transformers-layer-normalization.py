import mlx.core as mx


def layer_norm(x: mx.array, gamma: mx.array, beta: mx.array, eps: float = 1e-6) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.var(x, axis=-1, keepdims=True)
    x_normalized = (x - mean) / mx.sqrt(variance + eps)
    return gamma * x_normalized + beta
