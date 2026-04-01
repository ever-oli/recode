import mlx.core as mx


def rnn_forward(X: mx.array, h_0: mx.array, W_xh: mx.array, W_hh: mx.array, b_h: mx.array) -> tuple:
    batch_size, time_steps, _ = X.shape
    h_current = h_0
    h_all_list = []

    for t in range(time_steps):
        x_t = X[:, t, :]
        h_current = mx.tanh(mx.matmul(x_t, mx.transpose(W_xh)) + mx.matmul(h_current, mx.transpose(W_hh)) + b_h)
        h_all_list.append(h_current)

    h_all = mx.stack(h_all_list, axis=1)
    h_final = h_current
    return h_all, h_final
