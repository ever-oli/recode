import torch


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(mu)
    return mu + std * epsilon
