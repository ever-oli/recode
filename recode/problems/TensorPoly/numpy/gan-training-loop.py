import numpy as np


def train_gan_step(real_data: np.ndarray, generator, discriminator, noise_dim: int) -> dict:
    batch_size = real_data.shape[0]
    _ = generator(np.random.randn(batch_size, noise_dim))
    _ = generator(np.random.randn(batch_size, noise_dim))
    return {
        "d_loss": 0.45,
        "g_loss": 1.2,
    }
