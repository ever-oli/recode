import torch


def hidden_update(h_prev: torch.Tensor, h_tilde: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    keep_old = z_t * h_prev
    use_new = (1 - z_t) * h_tilde
    return keep_old + use_new
