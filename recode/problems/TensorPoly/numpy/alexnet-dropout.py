import numpy as np


def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    if not training or p == 0:
        return x

    mask = np.random.binomial(1, 1 - p, size=x.shape)
    return (x * mask) / (1 - p)
