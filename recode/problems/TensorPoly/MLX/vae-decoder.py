import mlx.core as mx


def vae_decoder(z: mx.array, output_dim: int) -> mx.array:
    _, latent_dim = z.shape
    hidden_dim = 256

    w_h = mx.random.normal(shape=(latent_dim, hidden_dim)) * 0.01
    b_h = mx.zeros((hidden_dim,))
    h = mx.maximum(0, mx.matmul(z, w_h) + b_h)

    w_out = mx.random.normal(shape=(hidden_dim, output_dim)) * 0.01
    b_out = mx.zeros((output_dim,))
    logits = mx.matmul(h, w_out) + b_out

    x_hat = 1 / (1 + mx.exp(-logits))
    return x_hat
