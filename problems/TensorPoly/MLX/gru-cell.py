import mlx.core as mx


def sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-mx.clip(x, -500, 500)))


def gru_cell(x_t: mx.array, h_prev: mx.array,
             W_r: mx.array, W_z: mx.array, W_h: mx.array,
             b_r: mx.array, b_z: mx.array, b_h: mx.array) -> mx.array:
    concat_gates = mx.concatenate([h_prev, x_t], axis=-1)
    r_t = sigmoid(mx.matmul(concat_gates, mx.transpose(W_r)) + b_r)
    z_t = sigmoid(mx.matmul(concat_gates, mx.transpose(W_z)) + b_z)

    gated_h = r_t * h_prev
    concat_cand = mx.concatenate([gated_h, x_t], axis=-1)
    h_tilde = mx.tanh(mx.matmul(concat_cand, mx.transpose(W_h)) + b_h)

    h_t = z_t * h_prev + (1 - z_t) * h_tilde
    return h_t
