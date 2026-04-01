import torch


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    x_normalized = (x - mean) / torch.sqrt(variance + eps)
    return gamma * x_normalized + beta
