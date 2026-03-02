import mlx.core as mx


def relu(x: mx.array) -> mx.array:
    return mx.maximum(0, x)
