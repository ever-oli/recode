import torch


def vae_decoder(z: torch.Tensor, output_dim: int) -> torch.Tensor:
    _, latent_dim = z.shape
    hidden_dim = 256

    w_h = torch.randn(latent_dim, hidden_dim) * 0.01
    b_h = torch.zeros(hidden_dim)
    h = torch.maximum(torch.tensor(0.0), torch.matmul(z, w_h) + b_h)

    w_out = torch.randn(hidden_dim, output_dim) * 0.01
    b_out = torch.zeros(output_dim)
    logits = torch.matmul(h, w_out) + b_out

    return 1 / (1 + torch.exp(-logits))
