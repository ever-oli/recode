import numpy as np


def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    spectral_norm = np.linalg.norm(W_hh, ord=2)
    norms = [1.0]
    current_norm = 1.0

    for _ in range(T - 1):
        current_norm *= spectral_norm
        norms.append(current_norm)

    return norms
