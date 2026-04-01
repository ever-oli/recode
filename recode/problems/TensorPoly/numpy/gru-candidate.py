import numpy as np


def candidate_hidden(h_prev: np.ndarray, x_t: np.ndarray, r_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    gated_h = r_t * h_prev
    concat = np.concatenate([gated_h, x_t], axis=-1)
    linear_transform = concat @ W_h.T + b_h
    return np.tanh(linear_transform)
