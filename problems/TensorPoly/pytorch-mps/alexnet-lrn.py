import torch


def local_response_normalization(x: torch.Tensor, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = x.to(device)
    _, _, _, c = x.shape
    squared_x = x * x
    pad = n // 2
    padded_sq = torch.nn.functional.pad(squared_x, (pad, pad, 0, 0, 0, 0, 0, 0))

    sum_sq = torch.zeros_like(x)
    for i in range(n):
        sum_sq = sum_sq + padded_sq[:, :, :, i:i + c]

    scale = (k + alpha * sum_sq) ** beta
    return x / scale
