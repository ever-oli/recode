import mlx.core as mx


def sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-mx.clip(x, -500, 500)))


def output_gate(h_prev: mx.array, x_t: mx.array, C_t: mx.array, W_o: mx.array, b_o: mx.array) -> tuple:
    concat = mx.concatenate([h_prev, x_t], axis=-1)
    o_t = sigmoid(mx.matmul(concat, mx.transpose(W_o)) + b_o)
    h_t = o_t * mx.tanh(C_t)
    return o_t, h_t
