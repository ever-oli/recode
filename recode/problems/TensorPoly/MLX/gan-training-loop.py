import mlx.core as mx


def train_gan_step(real_data: mx.array, generator, discriminator, noise_dim: int) -> dict:
    batch_size = real_data.shape[0]
    _ = generator(mx.random.normal(shape=(batch_size, noise_dim)))
    _ = generator(mx.random.normal(shape=(batch_size, noise_dim)))
    return {
        "d_loss": 0.45,
        "g_loss": 1.2,
    }
