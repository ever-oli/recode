import numpy as np


def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    batch_size, time_steps, _ = X.shape
    hidden_dim = h_0.shape[1]

    h_all_list = []
    h_current = h_0

    for t in range(time_steps):
        x_t = X[:, t, :]
        h_current = np.tanh(x_t @ W_xh.T + h_current @ W_hh.T + b_h)
        h_all_list.append(h_current)

    h_all = np.stack(h_all_list, axis=1)
    h_final = h_current
    return h_all, h_final
