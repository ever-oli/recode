import mlx.core as mx


def rnn_cell(x_t: mx.array, h_prev: mx.array, W_xh: mx.array, W_hh: mx.array, b_h: mx.array) -> mx.array:
    input_term = mx.matmul(x_t, mx.transpose(W_xh))
    hidden_term = mx.matmul(h_prev, mx.transpose(W_hh))
    h_t = mx.tanh(input_term + hidden_term + b_h)
    return h_t
