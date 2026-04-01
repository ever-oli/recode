import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def update_gate(h_prev: np.ndarray, x_t: np.ndarray, W_z: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    concat = np.concatenate([h_prev, x_t], axis=-1)
    linear_transform = concat @ W_z.T + b_z
    return sigmoid(linear_transform)
