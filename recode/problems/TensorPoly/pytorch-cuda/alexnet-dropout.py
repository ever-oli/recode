import torch


def dropout(x: torch.Tensor, p: float = 0.5, training: bool = True, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    if not training or p == 0:
        return x

    mask = torch.bernoulli(torch.full_like(x, 1 - p))
    return (x * mask) / (1 - p)
