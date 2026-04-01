import numpy as np


def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    dtanh = (1 - np.square(h_t)) * dh_next
    dW_hh = dtanh.T @ h_prev
    dh_prev = dtanh @ W_hh
    return dh_prev, dW_hh
