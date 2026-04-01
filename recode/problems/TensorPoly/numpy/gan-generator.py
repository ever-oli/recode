import numpy as np


def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    _, noise_dim = z.shape

    W1 = np.random.randn(noise_dim, 128) * 0.02
    b1 = np.zeros(128)
    W2 = np.random.randn(128, output_dim) * 0.02
    b2 = np.zeros(output_dim)

    h1 = np.maximum(0, np.matmul(z, W1) + b1)
    output = np.tanh(np.matmul(h1, W2) + b2)
    return output
