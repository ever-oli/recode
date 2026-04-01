import mlx.core as mx


def relu(x: mx.array) -> mx.array:
    return mx.maximum(0, x)


class IdentityBlock:
    def __init__(self, channels: int):
        self.channels = channels
        self.W1 = mx.random.normal(shape=(channels, channels)) * 0.01
        self.W2 = mx.random.normal(shape=(channels, channels)) * 0.01

    def forward(self, x: mx.array) -> mx.array:
        identity = x
        out = mx.matmul(x, self.W1)
        out = relu(out)
        out = mx.matmul(out, self.W2)
        return out + identity
