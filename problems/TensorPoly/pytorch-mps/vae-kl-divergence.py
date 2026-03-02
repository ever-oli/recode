import torch


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> float:
    var = torch.exp(log_var)
    kl_element = 1 + log_var - mu ** 2 - var
    batch_kl = -0.5 * torch.sum(kl_element, dim=1)
    return float(torch.mean(batch_kl).item())
