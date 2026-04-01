import mlx.core as mx


def sigmoid(x):
    x_arr = mx.array(x, dtype=mx.float32)
    return 1.0 / (1.0 + mx.exp(-x_arr))
