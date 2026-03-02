import mlx.core as mx


def candidate_hidden(h_prev: mx.array, x_t: mx.array, r_t: mx.array, W_h: mx.array, b_h: mx.array) -> mx.array:
    gated_h = r_t * h_prev
    concat = mx.concatenate([gated_h, x_t], axis=-1)
    linear_transform = mx.matmul(concat, mx.transpose(W_h)) + b_h
    return mx.tanh(linear_transform)
