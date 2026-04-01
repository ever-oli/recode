import torch


def generator(z: torch.Tensor, output_dim: int, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    z = z.to(device)
    _, noise_dim = z.shape

    W1 = torch.randn(noise_dim, 128, device=device) * 0.02
    b1 = torch.zeros(128, device=device)
    W2 = torch.randn(128, output_dim, device=device) * 0.02
    b2 = torch.zeros(output_dim, device=device)

    h1 = torch.maximum(torch.tensor(0.0, device=device), torch.matmul(z, W1) + b1)
    output = torch.tanh(torch.matmul(h1, W2) + b2)
    return output
