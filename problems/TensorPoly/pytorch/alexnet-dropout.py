import torch


def dropout(x: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    if not training or p == 0:
        return x

    mask = torch.bernoulli(torch.full_like(x, 1 - p))
    return (x * mask) / (1 - p)
