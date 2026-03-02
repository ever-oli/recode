import mlx.core as mx


def sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-mx.clip(x, -500, 500)))


def input_gate(h_prev: mx.array, x_t: mx.array, W_i: mx.array, b_i: mx.array, W_c: mx.array, b_c: mx.array) -> tuple:
    concat = mx.concatenate([h_prev, x_t], axis=-1)
    i_t = sigmoid(mx.matmul(concat, mx.transpose(W_i)) + b_i)
    c_tilde = mx.tanh(mx.matmul(concat, mx.transpose(W_c)) + b_c)
    return i_t, c_tilde
