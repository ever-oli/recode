import numpy as np


def hidden_update(h_prev: np.ndarray, h_tilde: np.ndarray, z_t: np.ndarray) -> np.ndarray:
    keep_old = z_t * h_prev
    use_new = (1 - z_t) * h_tilde
    return keep_old + use_new
