import numpy as np


def relu(x):
    return np.maximum(0, x)


class BottleneckBlock:
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int):
        self.in_ch = in_channels
        self.bn_ch = bottleneck_channels
        self.out_ch = out_channels

        self.W1 = np.random.randn(in_channels, bottleneck_channels) * 0.01
        self.W2 = np.random.randn(bottleneck_channels, bottleneck_channels) * 0.01
        self.W3 = np.random.randn(bottleneck_channels, out_channels) * 0.01

        self.Ws = np.random.randn(in_channels, out_channels) * 0.01 if in_channels != out_channels else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        identity = x
        out = relu(np.matmul(x, self.W1))
        out = relu(np.matmul(out, self.W2))
        out = np.matmul(out, self.W3)

        if self.Ws is not None:
            identity = np.matmul(identity, self.Ws)

        out = relu(out + identity)
        return out
