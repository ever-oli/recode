import torch


def init_hidden(batch_size: int, hidden_dim: int, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.zeros((batch_size, hidden_dim), device=device)
