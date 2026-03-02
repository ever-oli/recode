import numpy as np


def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    grad = np.array(x, copy=True)

    for F_grad in reversed(gradients_F):
        F_mat = np.array(F_grad)
        dim = F_mat.shape[-1]
        grad = grad @ (np.eye(dim) + F_mat)

    return grad


def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    grad = np.array(x, copy=True)

    for F_grad in reversed(gradients_F):
        F_mat = np.array(F_grad)
        grad = grad @ F_mat

    return grad
