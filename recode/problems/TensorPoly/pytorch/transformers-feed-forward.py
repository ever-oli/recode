import torch


def feed_forward(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
                 W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    hidden = torch.matmul(x, W1) + b1
    relu_out = torch.maximum(torch.tensor(0.0, dtype=hidden.dtype), hidden)
    return torch.matmul(relu_out, W2) + b2
