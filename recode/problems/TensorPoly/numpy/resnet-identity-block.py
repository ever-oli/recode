import numpy as np


def relu(x):
    return np.maximum(0, x)


class IdentityBlock:
    def __init__(self, channels: int):
        self.channels = channels
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        identity = x
        out = np.matmul(x, self.W1)
        out = relu(out)
        out = np.matmul(out, self.W2)
        return out + identity
