import torch


def init_hidden(batch_size: int, hidden_dim: int, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.zeros((batch_size, hidden_dim), device=device)
