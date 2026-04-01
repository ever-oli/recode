import torch


def relu(x: torch.Tensor, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    return torch.maximum(torch.tensor(0.0, device=device), x)
