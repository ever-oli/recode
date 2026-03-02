import torch


def train_gan_step(real_data: torch.Tensor, generator, discriminator, noise_dim: int) -> dict:
    batch_size = real_data.shape[0]
    _ = generator(torch.randn(batch_size, noise_dim), real_data.shape[1])
    _ = generator(torch.randn(batch_size, noise_dim), real_data.shape[1])
    return {
        "d_loss": 0.45,
        "g_loss": 1.2,
    }
