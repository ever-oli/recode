import torch


def relu(x: torch.Tensor, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = x.to(device)
    return torch.maximum(torch.tensor(0.0, device=device), x)
