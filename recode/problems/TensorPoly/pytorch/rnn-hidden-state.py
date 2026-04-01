import torch


def init_hidden(batch_size: int, hidden_dim: int) -> torch.Tensor:
    return torch.zeros((batch_size, hidden_dim))
