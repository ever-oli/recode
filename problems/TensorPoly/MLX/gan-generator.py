import mlx.core as mx


def generator(z: mx.array, output_dim: int) -> mx.array:
    _, noise_dim = z.shape

    W1 = mx.random.normal(shape=(noise_dim, 128)) * 0.02
    b1 = mx.zeros((128,))
    W2 = mx.random.normal(shape=(128, output_dim)) * 0.02
    b2 = mx.zeros((output_dim,))

    h1 = mx.matmul(z, W1) + b1
    h1 = mx.maximum(0, h1)
    output = mx.matmul(h1, W2) + b2
    return mx.tanh(output)
