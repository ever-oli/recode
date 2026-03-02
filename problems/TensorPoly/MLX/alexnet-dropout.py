import mlx.core as mx


def dropout(x: mx.array, p: float = 0.5, training: bool = True) -> mx.array:
    if not training or p == 0:
        return x

    mask = mx.random.bernoulli(1 - p, shape=x.shape)
    return (x * mask) / (1 - p)
