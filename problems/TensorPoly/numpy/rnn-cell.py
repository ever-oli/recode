import numpy as np


def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    input_term = x_t @ W_xh.T
    hidden_term = h_prev @ W_hh.T
    h_t = np.tanh(input_term + hidden_term + b_h)
    return h_t
