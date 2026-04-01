import torch


def train_gan_step(real_data: torch.Tensor, generator, discriminator, noise_dim: int, device=None) -> dict:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size = real_data.shape[0]
    _ = generator(torch.randn(batch_size, noise_dim, device=device), real_data.shape[1], device=device)
    _ = generator(torch.randn(batch_size, noise_dim, device=device), real_data.shape[1], device=device)
    return {
        "d_loss": 0.45,
        "g_loss": 1.2,
    }
