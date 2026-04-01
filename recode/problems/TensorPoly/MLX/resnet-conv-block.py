import mlx.core as mx


def relu(x: mx.array) -> mx.array:
    return mx.maximum(0, x)


class ConvBlock:
    """
    Convolutional Block with projection shortcut.
    """

    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W1 = mx.random.normal(shape=(in_channels, out_channels)) * 0.01
        self.W2 = mx.random.normal(shape=(out_channels, out_channels)) * 0.01
        self.Ws = mx.random.normal(shape=(in_channels, out_channels)) * 0.01

    def forward(self, x: mx.array) -> mx.array:
        main = mx.matmul(x, self.W1)
        main = relu(main)
        main = mx.matmul(main, self.W2)

        shortcut = mx.matmul(x, self.Ws)
        out = relu(main + shortcut)
        return out
