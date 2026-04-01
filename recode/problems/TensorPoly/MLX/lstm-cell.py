import mlx.core as mx


def sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-mx.clip(x, -500, 500)))


def lstm_cell(x_t: mx.array, h_prev: mx.array, C_prev: mx.array,
              W_f: mx.array, W_i: mx.array, W_c: mx.array, W_o: mx.array,
              b_f: mx.array, b_i: mx.array, b_c: mx.array, b_o: mx.array) -> tuple:
    concat = mx.concatenate([h_prev, x_t], axis=-1)
    f_t = sigmoid(mx.matmul(concat, mx.transpose(W_f)) + b_f)
    i_t = sigmoid(mx.matmul(concat, mx.transpose(W_i)) + b_i)
    c_tilde = mx.tanh(mx.matmul(concat, mx.transpose(W_c)) + b_c)
    o_t = sigmoid(mx.matmul(concat, mx.transpose(W_o)) + b_o)

    C_t = f_t * C_prev + i_t * c_tilde
    h_t = o_t * mx.tanh(C_t)
    return h_t, C_t
