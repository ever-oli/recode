import mlx.core as mx


def sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-mx.clip(x, -500, 500)))


def reset_gate(h_prev: mx.array, x_t: mx.array, W_r: mx.array, b_r: mx.array) -> mx.array:
    concat = mx.concatenate([h_prev, x_t], axis=-1)
    linear_transform = mx.matmul(concat, mx.transpose(W_r)) + b_r
    return sigmoid(linear_transform)
