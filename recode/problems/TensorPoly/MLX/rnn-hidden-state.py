import mlx.core as mx


def init_hidden(batch_size: int, hidden_dim: int) -> mx.array:
    return mx.zeros((batch_size, hidden_dim))
