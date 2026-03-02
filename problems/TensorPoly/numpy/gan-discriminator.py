import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def discriminator(x: np.ndarray) -> np.ndarray:
    _, input_dim = x.shape

    W1 = np.random.randn(input_dim, 256) * 0.02
    b1 = np.zeros(256)
    W2 = np.random.randn(256, 128) * 0.02
    b2 = np.zeros(128)
    W3 = np.random.randn(128, 1) * 0.02
    b3 = np.zeros(1)

    h1 = np.maximum(0.2 * (np.matmul(x, W1) + b1), np.matmul(x, W1) + b1)
    h2 = np.maximum(0.2 * (np.matmul(h1, W2) + b2), np.matmul(h1, W2) + b2)
    logits = np.matmul(h2, W3) + b3
    probs = sigmoid(logits)
    return probs
