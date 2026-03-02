import torch


def vae_decoder(z: torch.Tensor, output_dim: int, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    z = z.to(device)
    _, latent_dim = z.shape
    hidden_dim = 256

    w_h = torch.randn(latent_dim, hidden_dim, device=device) * 0.01
    b_h = torch.zeros(hidden_dim, device=device)
    h = torch.maximum(torch.tensor(0.0, device=device), torch.matmul(z, w_h) + b_h)

    w_out = torch.randn(hidden_dim, output_dim, device=device) * 0.01
    b_out = torch.zeros(output_dim, device=device)
    logits = torch.matmul(h, w_out) + b_out

    return 1 / (1 + torch.exp(-logits))
