import torch


def candidate_hidden(h_prev: torch.Tensor, x_t: torch.Tensor, r_t: torch.Tensor, W_h: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    gated_h = r_t * h_prev
    concat = torch.cat([gated_h, x_t], dim=-1)
    linear_transform = torch.matmul(concat, W_h.T) + b_h
    return torch.tanh(linear_transform)
